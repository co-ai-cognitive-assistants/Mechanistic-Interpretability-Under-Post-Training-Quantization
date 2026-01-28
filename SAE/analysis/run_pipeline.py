import os
import sys
import pandas as pd
import torch
import gc
import logging
import subprocess
from datetime import datetime
from datasets import load_dataset

# Add SAE root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from quantization_impact_study import (
    find_checkpoints, 
    compute_metrics, 
    generate_plots, 
    load_sae_checkpoint,
    get_decoder_weights
)
from functional_eval import run_eval
from utils_loading import load_hooked_model
import aggregate_interpretability as agg

# --- Configuration ---
CHECKPOINTS_DIR = "/experiment/SAE/checkpoints"
MODELS_DIR = "/experiment/models"
BASE_OUTPUT_DIR = "/experiment/SAE/analysis/results"
NUM_SAMPLES = 50 

# --- Helpers ---

def setup_logging(run_dir):
    log_file = os.path.join(run_dir, "analysis_log.txt")
    
    # Create custom logger
    logger = logging.getLogger("SAE_Analysis")
    logger.setLevel(logging.INFO)
    logger.handlers = [] # Clear existing
    
    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s')) # Clean format for console
    logger.addHandler(ch)
    
    return logger

def get_best_device(logger):
    """Finds the GPU with the most free memory."""
    if not torch.cuda.is_available():
        return "cpu"
    
    try:
        # Run nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, text=True
        )
        free_mem = [int(x) for x in result.stdout.strip().split('\n')]
        best_gpu = free_mem.index(max(free_mem))
        logger.info(f"Auto-detected GPU {best_gpu} with {max(free_mem)} MiB free.")
        return f"cuda:{best_gpu}"
    except Exception as e:
        logger.warning(f"Failed to auto-detect best GPU: {e}. Defaulting to cuda:0")
        return "cuda:0"

def validate_sae(sae, logger, name="SAE"):
    """Checks if SAE weights are valid (non-zero, non-NaN)."""
    if sae is None:
        return False
    
    try:
        w = get_decoder_weights(sae, "cpu")
        if w is None:
            logger.error(f"  [FAIL] {name}: Could not extract decoder weights.")
            return False
            
        if torch.isnan(w).any():
            logger.error(f"  [FAIL] {name}: Weights contain NaNs.")
            return False
            
        if w.sum().abs().item() < 1e-6:
            logger.error(f"  [FAIL] {name}: Weights are effectively zero (Dead SAE).")
            return False
            
        return True
    except Exception as e:
        logger.error(f"  [FAIL] {name}: Validation check crashed: {e}")
        return False

def flush_memory(logger):
    gc.collect()
    torch.cuda.empty_cache()
    # logger.info("  [System] Memory Flushed.")

def resolve_model_path(models_dir, family_name, method, logger):
    """
    Robustly resolves the path to a model directory given a family name and quantization method.
    Handles fuzzy family matching and subdirectory organization.
    """
    # 1. Identify the Model Family Directory
    family_dir = os.path.join(models_dir, family_name)
    
    if not os.path.exists(family_dir):
        # Fuzzy match family
        def clean(s): return s.lower().replace("-", " ").replace("_", " ").split()
        target_tokens = set(clean(family_name))
        
        candidates = []
        for d in os.listdir(models_dir):
            d_path = os.path.join(models_dir, d)
            if not os.path.isdir(d_path): continue
            
            d_tokens = set(clean(d))
            common = target_tokens.intersection(d_tokens)
            score = len(common) / len(target_tokens) if target_tokens else 0
            if score >= 0.6: 
                candidates.append((score, d_path))
        
        if candidates:
            # Sort by score desc, then shortest name
            candidates.sort(key=lambda x: (-x[0], len(x[1])))
            family_dir = candidates[0][1]
            logger.info(f"    Resolved family '{family_name}' to '{os.path.basename(family_dir)}'")
        else:
            return family_dir # Fallback

    # 2. Identify the Method Specific Weights
    # Check direct subdirectory (e.g., models/gemma/int4)
    method_subdir = os.path.join(family_dir, method)
    if os.path.exists(method_subdir) and os.path.isdir(method_subdir):
        return method_subdir
        
    # Check if any subdirectory contains the method name (case-insensitive)
    try:
        subdirs = [d for d in os.listdir(family_dir) if os.path.isdir(os.path.join(family_dir, d))]
        for d in subdirs:
            if method.lower() == d.lower() or method.lower() in d.lower():
                return os.path.join(family_dir, d)
    except Exception:
        pass
             
    return family_dir

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_OUTPUT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    logger = setup_logging(run_dir)
    logger.info(f"=== Starting Integrated SAE Analysis Pipeline ===")
    logger.info(f"Output Directory: {run_dir}")
    
    device = get_best_device(logger)
    
    # 1. Prepare Data
    logger.info("Loading Wikitext for functional verification...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="test")
        texts = [x["text"] for x in dataset if len(x["text"]) > 100][:NUM_SAMPLES]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 2. Scan Checkpoints
    checkpoints_map = find_checkpoints(CHECKPOINTS_DIR)
    if not checkpoints_map:
        logger.error("No checkpoints found.")
        return

    # 3. Process each Model Family
    for model_name, methods in checkpoints_map.items():
        logger.info(f"\n>>> Analyzing Family: {model_name}")
        model_output_dir = os.path.join(run_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Identify Baseline
        baselines = ["float32", "bfloat16", "float16"]
        baseline_method = next((b for b in baselines if b in methods), None)
        
        if not baseline_method:
            logger.warning(f"  Skipping {model_name}: No baseline found.")
            continue
            
        # A. Geometric Analysis
        logger.info("  [Step 1] Geometric Metrics...")
        
        # Pre-validate Baseline SAE
        base_path = methods[baseline_method]
        base_sae = load_sae_checkpoint(base_path, device=device)
        if not validate_sae(base_sae, logger, f"Baseline ({baseline_method})"):
             logger.error("  Aborting family analysis due to broken baseline.")
             continue
        
        # Clean dictionary of valid methods only
        valid_methods = {baseline_method: base_path}
        for m, p in methods.items():
            if m == baseline_method: continue
            
            # Check if exists
            if not os.path.exists(p):
                logger.warning(f"  Skipping {m}: Path missing.")
                continue
                
            # Quick load check (lightweight)
            # We delay full validation to compute_metrics to avoid double loading, 
            # but compute_metrics doesn't do deep validation.
            # Let's trust compute_metrics to run, but filter results later.
            valid_methods[m] = p

        df_geo, mappings = compute_metrics(model_name, valid_methods, baseline_method, device)
        
        # Post-Process Geometric Results: Filter Failures
        if df_geo.empty:
            logger.warning("  No valid geometric results.")
            continue
            
        # Check for Broken Rows
        valid_indices = []
        for idx, row in df_geo.iterrows():
            # Criteria for "Broken": Recall is near zero implies total misalignment or dead weights
            if row["Recall_Mean"] < 0.001:
                logger.warning(f"  [FLAG] {row['Method']} flagged as BROKEN (Recall < 0.001). Excluding.")
                continue
            valid_indices.append(idx)
            
        df_geo = df_geo.loc[valid_indices].copy()
        
        if df_geo.empty:
            logger.warning("  All methods failed validation. Skipping functional.")
            continue

        # B. Functional Analysis
        logger.info("  [Step 2] Functional Verification...")

        # Baseline SAE is already loaded in `base_sae` (CPU)
        base_sae.to(device)
        base_sae_path = methods[baseline_method]

        # Load baseline model ONCE for base_sae activations
        baseline_model_path = resolve_model_path(MODELS_DIR, model_name, baseline_method, logger)
        logger.info(f"")
        logger.info(f"    ┌─ BASELINE ({baseline_method}) ─────────────────────────────")
        logger.info(f"    │ Model: {baseline_model_path}")
        logger.info(f"    │ SAE:   {base_sae_path}")
        logger.info(f"    └────────────────────────────────────────────────────────")
        baseline_model = load_hooked_model(baseline_model_path, device=device)

        functional_results = []

        for method, target_sae_path in methods.items():
            if method == baseline_method: continue

            # Skip if geometric failed
            if method not in df_geo["Method"].values:
                continue

            logger.info(f"")
            logger.info(f"  > Verifying: {baseline_method} vs {method}")

            target_model = None
            target_sae = None

            try:
                # 1. Resolve Target Model Path (quantized version)
                target_model_path = resolve_model_path(MODELS_DIR, model_name, method, logger)

                # 2. Log the comparison setup clearly
                logger.info(f"    ┌─ COMPARISON ──────────────────────────────────────────")
                logger.info(f"    │ BASE:")
                logger.info(f"    │   Model: {baseline_model_path}")
                logger.info(f"    │   SAE:   {base_sae_path}")
                logger.info(f"    │ TARGET ({method}):")
                logger.info(f"    │   Model: {target_model_path}")
                logger.info(f"    │   SAE:   {target_sae_path}")
                logger.info(f"    └────────────────────────────────────────────────────────")

                # 3. Load Target Model (with dequantization matching train_saelens.py)
                logger.info(f"    Loading target model...")
                target_model = load_hooked_model(target_model_path, device=device)

                # 4. Load Target SAE
                logger.info(f"    Loading target SAE...")
                target_sae = load_sae_checkpoint(target_sae_path, device=device)
                if not validate_sae(target_sae, logger, f"Target ({method})"):
                    continue
                target_sae.to(device)

                # 5. Run Eval with BOTH models
                #    - base_sae gets activations from baseline_model
                #    - target_sae gets activations from target_model (dequantized)
                if method in mappings:
                    idx_map = mappings[method]["idx_base_to_target"]
                    sim_vals = mappings[method]["sim_base_to_target"]

                    logger.info(f"    Running functional eval...")
                    res = run_eval(baseline_model, base_sae, target_sae, texts, idx_map, device,
                                   model_target=target_model)
                    
                    if res:
                        # Sanity Check Result
                        if res["l0_target"] < 0.1:
                            logger.warning(f"    [FLAG] {method}: Target SAE is effectively dead (L0 < 0.1).")

                        # Log result summary
                        logger.info(f"    ┌─ RESULT ───────────────────────────────────────────────")
                        logger.info(f"    │ Jaccard:  {res['mean_jaccard']:.4f}")
                        logger.info(f"    │ L0 Base:  {res['l0_base']:.2f}  |  L0 Target: {res['l0_target']:.2f}")
                        logger.info(f"    │ L0 Ratio: {res['l0_ratio']:.4f}")
                        logger.info(f"    └────────────────────────────────────────────────────────")

                        # Compute geometric similarity (mean decoder cosine similarity)
                        geometric_sim_mean = sim_vals.mean().item()

                        functional_results.append({
                            "Model": model_name,
                            "Method": method,
                            "Baseline": baseline_method,
                            "Geometric_Similarity": geometric_sim_mean,
                            "Jaccard_Mean": res["mean_jaccard"],
                            "Activation_Corr": res["mean_activation_corr"],
                            "Weighted_Jaccard": res["weighted_jaccard"],
                            "L0_Ratio": res["l0_ratio"]
                        })

                        # Save Details
                        df_det = pd.DataFrame({
                            "BaseFeatureIdx": range(len(res["jaccard_scores"])),
                            "GeometricSim": sim_vals.cpu().numpy(),
                            "FunctionalJaccard": res["jaccard_scores"]
                        })
                        df_det.to_csv(os.path.join(model_output_dir, f"details_{method}.csv"), index=False)

                        # Save model activation similarity if available
                        if res.get("model_activation_similarity"):
                            model_act_data = []
                            for layer_idx, stats in res["model_activation_similarity"].items():
                                model_act_data.append({
                                    "Layer": layer_idx,
                                    "Mean_Abs_Diff": stats["mean_abs_diff"],
                                    "Max_Diff": stats["max_diff"],
                                    "Correlation": stats["correlation"]
                                })
                            df_model_act = pd.DataFrame(model_act_data)
                            df_model_act.to_csv(
                                os.path.join(model_output_dir, f"model_activation_similarity_{method}.csv"),
                                index=False
                            )
                            logger.info(f"    Model Activation Correlation: {model_act_data[-1]['Correlation']:.4f} (layer {model_act_data[-1]['Layer']})")
                    else:
                        logger.warning(f"    [FAIL] {method}: Functional eval returned None.")
                else:
                    logger.warning(f"    [SKIP] {method}: No geometric mapping found.")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"    [OOM] GPU Out of Memory for {method}. Skipping.")
                    torch.cuda.empty_cache()
                else:
                    logger.error(f"    [ERROR] {method}: {e}")
            except Exception as e:
                logger.error(f"    [ERROR] {method}: {e}")
            finally:
                # CRITICAL: Always flush target model/SAE memory (baseline stays loaded)
                if target_model: del target_model
                if target_sae: del target_sae
                target_model = None
                target_sae = None
                flush_memory(logger)

        # Cleanup Baseline Model and SAE
        del baseline_model
        del base_sae
        flush_memory(logger)

        # Merge and Visualize
        if functional_results:
            df_func = pd.DataFrame(functional_results)
            df_combined = pd.merge(df_geo, df_func, on=["Model", "Method", "Baseline"], how="left")
        else:
            df_combined = df_geo
            
        generate_plots(model_name, df_combined, model_output_dir)

    # 4. Global Aggregation
    logger.info("\n=== Finalizing Global Statistics ===")
    try:
        agg_df = agg.load_all_stats(run_dir)
        if not agg_df.empty:
            agg.plot_global_comparison(agg_df, "Recall_GT_0.9", run_dir, "Feature Recall")
            agg.plot_global_comparison(agg_df, "Rank_Retention", run_dir, "Rank Retention")
            if "Jaccard_Mean" in agg_df.columns:
                agg.plot_global_comparison(agg_df, "Jaccard_Mean", run_dir, "Functional Consistency (Jaccard)")
        else:
            logger.warning("No data found for global aggregation.")
    except Exception as e:
        logger.error(f"Global aggregation failed: {e}")

    logger.info(f"\nPipeline Complete. Log saved to: {os.path.join(run_dir, 'analysis_log.txt')}")

if __name__ == "__main__":
    main()