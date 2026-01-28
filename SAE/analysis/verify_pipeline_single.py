import os
import sys
import torch
import pandas as pd
import logging
from datetime import datetime
from datasets import load_dataset

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from quantization_impact_study import (
    compute_metrics, 
    generate_plots, 
    load_sae_checkpoint,
    get_decoder_weights
)
from functional_eval import run_eval
from utils_loading import load_hooked_model
from run_pipeline import resolve_model_path, setup_logging

# --- Configuration for single test ---
MODELS_DIR = "/experiment/models"
CHECKPOINTS_DIR = "/experiment/SAE/checkpoints"
TEST_FAMILY = "google_gemma-3-1b-it"
TEST_METHOD = "gptq"
TEST_BASELINE = "bfloat16"
NUM_SAMPLES = 10 # Small sample for quick test

def verify_single_run():
    timestamp = datetime.now().strftime("test_%Y%m%d_%H%M%S")
    run_dir = os.path.join("/experiment/SAE/analysis/results", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    logger = setup_logging(run_dir)
    logger.info("=== Starting Single-Shot Pipeline Verification ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = "cuda:2" # Using GPU 2 as seen in logs
    
    # 1. Load Data
    logger.info("1. Loading Wikitext sample...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="test")
    texts = [x["text"] for x in dataset if len(x["text"]) > 100][:NUM_SAMPLES]

    # 2. Locate SAE Checkpoints
    # We simulate what find_checkpoints does
    base_sae_path = os.path.join(CHECKPOINTS_DIR, f"{TEST_FAMILY}_{TEST_BASELINE}_sae_gemma3-1b/checkpoint_244000.pt")
    target_sae_path = os.path.join(CHECKPOINTS_DIR, f"{TEST_FAMILY}_{TEST_METHOD}_sae_gemma3-1b/checkpoint_244000.pt")
    
    logger.info(f"2. Loading SAEs...\n   Base: {base_sae_path}\n   Target: {target_sae_path}")
    
    sae_base = load_sae_checkpoint(base_sae_path, device=device)
    sae_target = load_sae_checkpoint(target_sae_path, device=device)
    
    if not sae_base or not sae_target:
        logger.error("FAILED: Could not load SAE checkpoints.")
        return

    # 3. Geometric Metrics
    logger.info("3. Computing Geometric Metrics...")
    methods_dict = {TEST_BASELINE: base_sae_path, TEST_METHOD: target_sae_path}
    df_geo, mappings = compute_metrics(TEST_FAMILY, methods_dict, TEST_BASELINE, device)
    
    if df_geo.empty or TEST_METHOD not in mappings:
        logger.error("FAILED: Geometric analysis produced no results.")
        return
    
    logger.info(f"   Recall (Mean): {df_geo.iloc[1]['Recall_Mean']:.4f}")

    # 4. Functional Verification
    logger.info("4. Loading Model and Running Functional Eval...")
    model_path = resolve_model_path(MODELS_DIR, TEST_FAMILY, TEST_METHOD, logger)
    logger.info(f"   Resolved Model Path: {model_path}")
    
    model = load_hooked_model(model_path, device=device)
    
    idx_map = mappings[TEST_METHOD]["idx_base_to_target"]
    res = run_eval(model, sae_base, sae_target, texts, idx_map, device)
    
    if not res:
        logger.error("FAILED: Functional evaluation returned None.")
        return
        
    logger.info(f"   Jaccard (Mean): {res['mean_jaccard']:.4f}")

    # 5. Plotting
    logger.info("5. Generating Plots...")
    df_func = pd.DataFrame([{ 
        "Model": TEST_FAMILY,
        "Method": TEST_METHOD,
        "Baseline": TEST_BASELINE,
        "Jaccard_Mean": res["mean_jaccard"],
        "L0_Ratio": res["l0_ratio"]
    }])
    df_combined = pd.merge(df_geo, df_func, on=["Model", "Method", "Baseline"], how="left")
    generate_plots(TEST_FAMILY, df_combined, run_dir)
    
    logger.info(f"=== SUCCESS ===\nResults saved to: {run_dir}")

if __name__ == "__main__":
    verify_single_run()
