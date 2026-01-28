import os
import sys
import torch
import pandas as pd
import logging
from datetime import datetime
from datasets import load_dataset

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from quantization_impact_study import (
    compute_metrics, 
    generate_plots, 
    load_sae_checkpoint,
    get_decoder_weights
)
from functional_eval import run_eval, get_activations
from utils_loading import load_hooked_model
from run_pipeline import resolve_model_path, setup_logging

# --- Configuration for DeepSeek test ---
MODELS_DIR = "/experiment/models"
CHECKPOINTS_DIR = "/experiment/SAE/checkpoints"
TEST_FAMILY = "DeepSeek-R1-1.5b"
TEST_METHOD = "fp8"
TEST_BASELINE = "bfloat16"
NUM_SAMPLES = 5

def verify_deepseek():
    timestamp = datetime.now().strftime("test_deepseek_%Y%m%d_%H%M%S")
    run_dir = os.path.join("/experiment/SAE/analysis/results", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    logger = setup_logging(run_dir)
    logger.info("=== Starting DeepSeek Functional Verification ===")
    
    device = "cuda:2" # Using GPU 2
    
    # 1. Load Data
    logger.info("1. Loading Wikitext sample...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="test")
    texts = [x["text"] for x in dataset if len(x["text"]) > 100][:NUM_SAMPLES]

    # 2. Locate SAE Checkpoints
    base_sae_path = os.path.join(CHECKPOINTS_DIR, "DeepSeek-R1-1.5b-bfloat16/qg701fza/final_1000001536")
    target_sae_path = os.path.join(CHECKPOINTS_DIR, "DeepSeek-R1-1.5b-fp8/e1yu9tbs/final_1000001536")
    
    logger.info(f"2. Loading SAEs...\n   Base: {base_sae_path}\n   Target: {target_sae_path}")
    
    sae_base = load_sae_checkpoint(base_sae_path, device=device)
    sae_target = load_sae_checkpoint(target_sae_path, device=device)
    
    if not sae_base or not sae_target:
        logger.error("FAILED: Could not load SAE checkpoints.")
        return

    def get_hook(sae):
        h = getattr(sae.cfg, 'hook_name', getattr(sae.cfg, 'hook_point', None))
        if h is None and hasattr(sae.cfg, 'metadata'):
            m = sae.cfg.metadata
            if isinstance(m, dict): h = m.get('hook_name', m.get('hook_point'))
            else: h = getattr(m, 'hook_name', getattr(m, 'hook_point', None))
        return h or 'Unknown'

    logger.info(f"   Base SAE Hook: {get_hook(sae_base)}")
    logger.info(f"   Target SAE Hook: {get_hook(sae_target)}")

    # 3. Geometric Metrics
    logger.info("3. Computing Geometric Metrics...")
    methods_dict = {TEST_BASELINE: base_sae_path, TEST_METHOD: target_sae_path}
    df_geo, mappings = compute_metrics(TEST_FAMILY, methods_dict, TEST_BASELINE, device)
    
    logger.info(f"   Geometric Recall (Mean): {df_geo.iloc[1]['Recall_Mean']:.4f}")

    # 4. Functional Verification
    logger.info("4. Loading Model and Running Functional Eval...")
    # FORCE BASELINE MODEL FOR BOTH to isolate weight issues
    model_path = resolve_model_path(MODELS_DIR, TEST_FAMILY, TEST_BASELINE, logger)
    logger.info(f"   Resolved Model Path (FORCED BASELINE): {model_path}")
    
    model = load_hooked_model(model_path, device=device)
    
    # Extract activations explicitly for stats
    print("Extracting activations...")
    # Manual extraction of RAW activations (d_in)
    all_raw_acts = []
    
    with torch.no_grad():
        for i in range(0, len(texts), 4):
            batch_texts = texts[i:i+4]
            tokens = model.to_tokens(batch_texts)
            _, cache = model.run_with_cache(tokens, stop_at_layer=model.cfg.n_layers)
            
            # Find hook
            hook_name = getattr(sae_base.cfg, "hook_name", "blocks.21.hook_resid_post")
            if hook_name not in cache:
                # Fallback search
                for k in cache.keys():
                    if k.endswith("hook_resid_post") and cache[k].shape[-1] == 1536:
                        hook_name = k
                        break
            
            raw = cache[hook_name] # [batch, pos, d_in]
            flat = raw.reshape(-1, raw.shape[-1])
            all_raw_acts.append(flat.cpu()) # CPU to save memory
            
            del tokens, cache, raw, flat
            torch.cuda.empty_cache()
            
    acts = torch.cat(all_raw_acts, dim=0).to(device)
    print(f"Extracted Acts Shape: {acts.shape}")
    
    # Run Eval (Jaccard)
    # ... (Keep existing)
    idx_map = mappings[TEST_METHOD]["idx_base_to_target"]
    # We can reuse 'acts' for run_eval? No, run_eval takes texts.
    res = run_eval(model, sae_base, sae_target, texts, idx_map, device)
    
    if not res:
        logger.error("FAILED: Functional evaluation returned None.")
        return
        
    # 4. Pass through SAEs (Stats)
    print("Running SAE Forward Pass...")
    with torch.no_grad():
        # Get reconstructions (forward pass)
        recon_base = sae_base(acts)
        recon_target = sae_target(acts)

    # Stats: Compute MSE and FVU
    act_mean = acts.mean().item()
    act_std = acts.std().item()
    act_var = acts.var().item()
    
    print(f"\n   Activations Statistics: Mean={act_mean:.6f}, Std={act_std:.6f}, Var={act_var:.6f}")
    
    mse_base = (acts - recon_base).pow(2).mean().item()
    fvu_base = mse_base / act_var if act_var > 0 else 0.0
    
    mse_target = (acts - recon_target).pow(2).mean().item()
    fvu_target = mse_target / act_var if act_var > 0 else 0.0
    
    print(f"   Base SAE   -> MSE: {mse_base:.6f}, FVU: {fvu_base:.4f}")
    print(f"   Target SAE -> MSE: {mse_target:.6f}, FVU: {fvu_target:.4f}")
    
    if fvu_base > 0.8:
        print("   [WARNING] High FVU (>0.8) indicates SAEs are treating inputs as random noise.")

    # 5. Compute Jaccard
    logger.info(f"   Jaccard (Mean): {res['mean_jaccard']:.4f}")
    logger.info(f"   L0 (Base): {res['l0_base']:.2f}")
    logger.info(f"   L0 (Target): {res['l0_target']:.2f}")

    if res['mean_jaccard'] > 0.1:
        logger.info("=== SUCCESS: Jaccard is non-zero! (Meaning weights match activations) ===")
    else:
        logger.error("=== FAILURE: Jaccard is still near zero! (Meaning weights DON'T match activations) ===")

if __name__ == "__main__":
    verify_deepseek()