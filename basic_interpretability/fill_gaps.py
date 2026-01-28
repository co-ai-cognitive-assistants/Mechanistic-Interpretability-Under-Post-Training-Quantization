"""
Fill in missing method/model combinations in the interpretability analysis.
Identifies gaps and runs analysis only for missing combinations.

Usage for background execution:
    # In tmux:
    python basic_interpretability/fill_gaps.py --device cuda:0 2>&1 | tee fill_gaps_output.log

    # Or with nohup:
    nohup python basic_interpretability/fill_gaps.py --device cuda:0 > fill_gaps_output.log 2>&1 &
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import gc
from datetime import datetime
from tqdm import tqdm

# Setup Logging with timestamps
log_file = os.path.join(os.path.dirname(__file__), "fill_gaps.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Disable tqdm progress bars for cleaner logs in background mode
# Set TQDM_DISABLE=1 environment variable to disable, or use --no-progress flag

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from basic_interpretability.loader import load_hf_dequantized
from basic_interpretability.run_comprehensive_analysis import (
    get_dataset_samples,
    run_logit_lens_analysis,
    run_attention_analysis,
    compute_kl,
    perform_sanity_check,
    cleanup_memory
)

MODELS_DIR = "/experiment/models"
RESULTS_DIR = "/experiment/basic_interpretability/results"

# All methods we want coverage for (excluding gguf which isn't supported)
TARGET_METHODS = ['float16', 'fp8', 'int8', 'int4', 'gptq', 'awq', 'hqq']


def get_existing_methods(family_name: str, tier: str) -> set:
    """Get methods already analyzed for a family."""
    if tier == "tier1":
        csv_path = os.path.join(RESULTS_DIR, f"run_{family_name}", "tier1", "logit_lens_kl.csv")
    else:
        csv_path = os.path.join(RESULTS_DIR, f"run_{family_name}", "tier2", "attention_entropy.csv")

    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path)
    return set(df['method'].unique())


def get_available_methods(family_name: str) -> set:
    """Get methods available in models directory."""
    family_path = os.path.join(MODELS_DIR, family_name)
    if not os.path.exists(family_path):
        return set()
    return set(os.listdir(family_path)) - {'bfloat16', 'bf16', 'gguf'}


def identify_gaps() -> dict:
    """Identify all missing method/model combinations."""
    gaps = {}

    for family_name in os.listdir(MODELS_DIR):
        family_path = os.path.join(MODELS_DIR, family_name)
        if not os.path.isdir(family_path):
            continue

        # Check for baseline
        if not os.path.exists(os.path.join(family_path, 'bfloat16')):
            continue

        available = get_available_methods(family_name)
        existing_t1 = get_existing_methods(family_name, "tier1")
        existing_t2 = get_existing_methods(family_name, "tier2")

        # Find gaps
        missing_t1 = (available & set(TARGET_METHODS)) - existing_t1
        missing_t2 = (available & set(TARGET_METHODS)) - existing_t2 - {'bfloat16'}

        if missing_t1 or missing_t2:
            gaps[family_name] = {
                'tier1': list(missing_t1),
                'tier2': list(missing_t2),
                'available': list(available)
            }

    return gaps


def run_missing_analysis(family_name: str, missing_methods: list, tier: str,
                         samples: list, device: str, base_data: dict = None):
    """Run analysis for missing methods."""
    family_path = os.path.join(MODELS_DIR, family_name)
    run_dir = os.path.join(RESULTS_DIR, f"run_{family_name}")

    results = []

    for method in missing_methods:
        method_path = os.path.join(family_path, method)
        if not os.path.exists(method_path):
            logger.warning(f"Method path not found: {method_path}")
            continue

        logger.info(f"  Running {tier} for {method}...")
        model = None
        try:
            model = load_hf_dequantized(method_path, device=device)

            passed, reason = perform_sanity_check(model)
            if not passed:
                logger.warning(f"  Sanity check failed for {method}: {reason}")
                del model
                model = None
                cleanup_memory()
                continue

            if tier == "tier1":
                _, target_probs = run_logit_lens_analysis(
                    model, samples,
                    reference_indices=base_data['indices']
                )
                kl_df = compute_kl(base_data['probs'], target_probs)
                kl_df["method"] = method
                results.append(kl_df)
            else:
                attn_df = run_attention_analysis(model, samples)
                if attn_df is not None:
                    attn_df["method"] = method
                    results.append(attn_df)

            del model
            model = None
            cleanup_memory()

        except Exception as e:
            logger.error(f"  Error on {method}: {e}")
            if model is not None:
                del model
            cleanup_memory()

    return results


def append_results(family_name: str, tier: str, new_results: list):
    """Append new results to existing CSV."""
    if not new_results:
        return

    run_dir = os.path.join(RESULTS_DIR, f"run_{family_name}")
    os.makedirs(os.path.join(run_dir, tier), exist_ok=True)

    if tier == "tier1":
        csv_path = os.path.join(run_dir, "tier1", "logit_lens_kl.csv")
    else:
        csv_path = os.path.join(run_dir, "tier2", "attention_entropy.csv")

    new_df = pd.concat(new_results, ignore_index=True)

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved {len(new_results)} new method(s) to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Fill gaps in interpretability analysis")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tier", type=str, choices=["tier1", "tier2", "both"], default="both")
    parser.add_argument("--families", type=str, nargs="*", help="Specific families to process")
    parser.add_argument("--methods", type=str, nargs="*", help="Specific methods to process")
    parser.add_argument("--dry-run", action="store_true", help="Only show gaps, don't run")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    args = parser.parse_args()

    # Disable tqdm if requested (cleaner for background logs)
    if args.no_progress:
        os.environ["TQDM_DISABLE"] = "1"

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"FILL GAPS STARTED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {args.device}, Tier: {args.tier}")
    logger.info("=" * 60)

    logger.info("\n=== Identifying gaps in interpretability analysis ===")
    gaps = identify_gaps()

    # Filter by families if specified
    if args.families:
        gaps = {k: v for k, v in gaps.items() if k in args.families}

    # Print summary
    total_t1 = sum(len(v['tier1']) for v in gaps.values())
    total_t2 = sum(len(v['tier2']) for v in gaps.values())

    logger.info(f"\nFound {len(gaps)} families with gaps:")
    logger.info(f"  Total tier1 (logit lens) gaps: {total_t1}")
    logger.info(f"  Total tier2 (attention) gaps: {total_t2}")

    for family, info in sorted(gaps.items()):
        logger.info(f"\n{family}:")
        if info['tier1']:
            logger.info(f"  tier1 missing: {info['tier1']}")
        if info['tier2']:
            logger.info(f"  tier2 missing: {info['tier2']}")

    if args.dry_run:
        logger.info("\n[DRY RUN] No analysis performed.")
        return

    # Load samples once
    samples = get_dataset_samples()

    # Progress tracking
    completed_t1 = 0
    completed_t2 = 0
    failed_t1 = []
    failed_t2 = []
    family_count = 0
    total_families = len(gaps)

    # Process each family
    for family_name, info in sorted(gaps.items()):
        family_count += 1
        logger.info("\n" + "=" * 60)
        logger.info(f">>> [{family_count}/{total_families}] Processing {family_name}")
        logger.info(f"    Time elapsed: {datetime.now() - start_time}")

        # Filter methods if specified
        t1_methods = info['tier1']
        t2_methods = info['tier2']
        if args.methods:
            t1_methods = [m for m in t1_methods if m in args.methods]
            t2_methods = [m for m in t2_methods if m in args.methods]

        if not t1_methods and not t2_methods:
            continue

        # Load baseline for logit lens reference
        base_data = None
        if t1_methods and args.tier in ["tier1", "both"]:
            baseline_path = os.path.join(MODELS_DIR, family_name, "bfloat16")
            logger.info(f"  Loading baseline from {baseline_path}")

            try:
                base_model = load_hf_dequantized(baseline_path, device=args.device)
                base_indices, base_probs = run_logit_lens_analysis(base_model, samples)
                base_data = {'indices': base_indices, 'probs': base_probs}
                del base_model
                cleanup_memory()
            except Exception as e:
                logger.error(f"  Failed to load baseline: {e}")
                continue

        # Run tier1
        if t1_methods and args.tier in ["tier1", "both"] and base_data:
            logger.info(f"  Running tier1 for: {t1_methods}")
            results = run_missing_analysis(
                family_name, t1_methods, "tier1",
                samples, args.device, base_data
            )
            if results:
                append_results(family_name, "tier1", results)
                completed_t1 += len(results)
                logger.info(f"  [OK] Completed {len(results)} tier1 method(s)")
            failed_count = len(t1_methods) - len(results)
            if failed_count > 0:
                failed_t1.extend([(family_name, m) for m in t1_methods[-failed_count:]])

        # Run tier2
        if t2_methods and args.tier in ["tier2", "both"]:
            logger.info(f"  Running tier2 for: {t2_methods}")
            results = run_missing_analysis(
                family_name, t2_methods, "tier2",
                samples, args.device
            )
            if results:
                append_results(family_name, "tier2", results)
                completed_t2 += len(results)
                logger.info(f"  [OK] Completed {len(results)} tier2 method(s)")
            failed_count = len(t2_methods) - len(results)
            if failed_count > 0:
                failed_t2.extend([(family_name, m) for m in t2_methods[-failed_count:]])

        cleanup_memory()

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("\n" + "=" * 60)
    logger.info("FILL GAPS COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duration:   {duration}")
    logger.info("")
    logger.info(f"Tier1 (logit lens): {completed_t1}/{total_t1} completed")
    logger.info(f"Tier2 (attention):  {completed_t2}/{total_t2} completed")

    if failed_t1:
        logger.info(f"\nFailed tier1: {failed_t1}")
    if failed_t2:
        logger.info(f"\nFailed tier2: {failed_t2}")

    logger.info("\n" + "=" * 60)
    logger.info("Run 'python basic_interpretability/reaggregate_data.py' to update global CSVs")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
