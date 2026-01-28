#!/usr/bin/env python3
"""
Compute Jaccard similarity between two SAE checkpoints using real activations.
Each SAE is run on its corresponding model (matching training setup).

Usage:
    # Self-comparison (sanity check - should yield Jaccard = 1.0)
    python test_self_jaccard.py --model <model_path> --base <sae_path>

    # Compare two SAEs (e.g., BF16 vs FP8)
    python test_self_jaccard.py \
        --model_base <bf16_model> --base <bf16_sae> \
        --model_target <fp8_model> --target <fp8_sae>

Examples:
    # Self-comparison
    python test_self_jaccard.py \
        --model /experiment/models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/fp8 \
        --base /experiment/SAE/checkpoints/DeepSeek-R1-1.5b-fp8/e1yu9tbs/final_1000001536

    # Compare BF16 baseline vs FP8 quantized
    python test_self_jaccard.py \
        --model_base /experiment/models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/bfloat16 \
        --base /experiment/SAE/checkpoints/DeepSeek-R1-1.5b-bfloat16/*/final_* \
        --model_target /experiment/models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/fp8 \
        --target /experiment/SAE/checkpoints/DeepSeek-R1-1.5b-fp8/*/final_*
"""

import sys
import os
import argparse
import torch
import numpy as np
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from quantization_impact_study import load_sae_checkpoint, get_decoder_weights, compute_similarity_matrix
from functional_eval import get_pre_topk_activations, compute_pre_topk_jaccard
from utils_loading import load_hooked_model


def compare_saes(model_base_path, path_base, model_target_path=None, path_target=None, num_samples=20):
    """
    Compare two SAEs using real model activations.

    Each SAE is evaluated on its corresponding model to ensure activations match training.
    """

    self_comparison = (path_target is None or path_target == path_base)
    if self_comparison:
        path_target = path_base
        model_target_path = model_base_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load SAEs
    print(f"\n{'='*60}")
    print("LOADING SAEs")
    print(f"{'='*60}")
    print(f"Base SAE:   {path_base}")
    sae_base = load_sae_checkpoint(path_base, device=device)
    if sae_base is None:
        print("Failed to load base SAE!")
        return

    # Print SAE config info
    if hasattr(sae_base, 'cfg'):
        cfg = sae_base.cfg
        hook = getattr(cfg, 'hook_name', getattr(cfg, 'hook_point', 'unknown'))
        d_in = getattr(cfg, 'd_in', 'unknown')
        d_sae = getattr(cfg, 'd_sae', 'unknown')
        print(f"  Config: hook={hook}, d_in={d_in}, d_sae={d_sae}")

    if self_comparison:
        sae_target = sae_base
        print("Target SAE: (same as base - self comparison)")
    else:
        print(f"Target SAE: {path_target}")
        sae_target = load_sae_checkpoint(path_target, device=device)
        if sae_target is None:
            print("Failed to load target SAE!")
            return

        # Print target SAE config info
        if hasattr(sae_target, 'cfg'):
            cfg = sae_target.cfg
            hook = getattr(cfg, 'hook_name', getattr(cfg, 'hook_point', 'unknown'))
            d_in = getattr(cfg, 'd_in', 'unknown')
            d_sae = getattr(cfg, 'd_sae', 'unknown')
            print(f"  Config: hook={hook}, d_in={d_in}, d_sae={d_sae}")

    # 2. Compute geometric mapping
    print(f"\n{'='*60}")
    print("GEOMETRIC ANALYSIS")
    print(f"{'='*60}")
    w_base = get_decoder_weights(sae_base, device)
    w_target = get_decoder_weights(sae_target, device)
    print(f"Base decoder shape:   {w_base.shape}")
    print(f"Target decoder shape: {w_target.shape}")

    if w_base.shape != w_target.shape:
        print("ERROR: SAE dimensions don't match!")
        return

    sim_vals, idx_base_to_target, _, _ = compute_similarity_matrix(w_base, w_target)
    print(f"Mean geometric similarity: {sim_vals.mean().item():.6f}")
    print(f"Min geometric similarity:  {sim_vals.min().item():.6f}")

    if self_comparison:
        is_identity = torch.all(idx_base_to_target == torch.arange(len(idx_base_to_target), device=idx_base_to_target.device)).item()
        print(f"Mapping is identity:       {is_identity}")

    # 3. Load Models
    print(f"\n{'='*60}")
    print("LOADING MODELS")
    print(f"{'='*60}")
    print(f"Base model path:   {model_base_path}")
    model_base = load_hooked_model(model_base_path, device=device)

    if self_comparison:
        model_target = model_base
        print("Target model:      (same as base)")
    else:
        print(f"Target model path: {model_target_path}")
        model_target = load_hooked_model(model_target_path, device=device)

    # 4. Load Data
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="test")
    texts = [x["text"] for x in dataset if len(x["text"]) > 100][:num_samples]
    print(f"Loaded {len(texts)} text samples from wikitext")

    # 5. Get Real Activations (each SAE on its corresponding model)
    print(f"\n{'='*60}")
    print("EXTRACTING ACTIVATIONS")
    print(f"{'='*60}")

    # First, let's verify the raw model activations are actually different
    if not self_comparison:
        print("Checking raw model activations (before SAE)...")

        # Get SAE's target layer from config
        sae_d_in = sae_base.cfg.d_in if hasattr(sae_base, 'cfg') else 1536

        with torch.no_grad():
            test_tokens = model_base.to_tokens(texts[:2])
            _, cache_base = model_base.run_with_cache(test_tokens, stop_at_layer=model_base.cfg.n_layers)
            _, cache_target = model_target.run_with_cache(test_tokens, stop_at_layer=model_target.cfg.n_layers)

            # Check multiple layers to see how differences compound
            layers_to_check = [0, model_base.cfg.n_layers // 2, int(model_base.cfg.n_layers * 0.75)]
            for layer_idx in layers_to_check:
                key = f"blocks.{layer_idx}.hook_resid_post"
                if key in cache_base:
                    act_base = cache_base[key]
                    act_target = cache_target[key]
                    diff = (act_base - act_target).abs().mean().item()
                    max_diff = (act_base - act_target).abs().max().item()
                    corr = torch.corrcoef(torch.stack([act_base.flatten(), act_target.flatten()]))[0,1].item()
                    print(f"  Layer {layer_idx} ({key}):")
                    print(f"    Mean abs diff: {diff:.6f}, Max diff: {max_diff:.4f}")
                    print(f"    Correlation: {corr:.6f}")

            del cache_base, cache_target

    print("\nRunning base model + base SAE encoder...")
    pre_topk_base, encoded_base = get_pre_topk_activations(model_base, sae_base, texts)
    print(f"  Shape: {pre_topk_base.shape} (tokens x features)")

    if self_comparison:
        pre_topk_target = pre_topk_base
        encoded_target = encoded_base
        print("Using same activations for target (self-comparison)")
    else:
        print("Running target model + target SAE encoder...")
        pre_topk_target, encoded_target = get_pre_topk_activations(model_target, sae_target, texts)
        print(f"  Shape: {pre_topk_target.shape}")

    # 6. Diagnostic: Check activation distributions
    print(f"\n{'='*60}")
    print("ACTIVATION DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"Base pre-topk stats:")
    print(f"  mean: {pre_topk_base.mean().item():.4f}, std: {pre_topk_base.std().item():.4f}")
    print(f"  min:  {pre_topk_base.min().item():.4f}, max: {pre_topk_base.max().item():.4f}")
    print(f"  >0:   {(pre_topk_base > 0).float().mean().item()*100:.2f}%")
    print(f"Target pre-topk stats:")
    print(f"  mean: {pre_topk_target.mean().item():.4f}, std: {pre_topk_target.std().item():.4f}")
    print(f"  min:  {pre_topk_target.min().item():.4f}, max: {pre_topk_target.max().item():.4f}")
    print(f"  >0:   {(pre_topk_target > 0).float().mean().item()*100:.2f}%")

    # Check if activations are correlated at all
    flat_base = pre_topk_base.flatten()
    flat_target = pre_topk_target.flatten()
    # Sample for speed
    sample_idx = torch.randperm(len(flat_base))[:100000]
    corr = np.corrcoef(flat_base[sample_idx].numpy(), flat_target[sample_idx].numpy())[0, 1]
    print(f"Raw activation correlation (sampled): {corr:.4f}")

    # 7. Compute Jaccard
    print(f"\n{'='*60}")
    print("COMPUTING JACCARD")
    print(f"{'='*60}")

    per_feature_jaccard, global_jaccard = compute_pre_topk_jaccard(
        pre_topk_base, pre_topk_target, idx_base_to_target.cpu()
    )

    # 7. Additional Stats
    l0_base = (encoded_base > 0).float().sum(dim=1).mean().item()
    l0_target = (encoded_target > 0).float().sum(dim=1).mean().item()

    # Count active features (for filtering dead features in per-feature stats)
    active_base = (pre_topk_base > 0).any(dim=0).cpu().numpy()
    active_target = (pre_topk_target > 0).any(dim=0).cpu().numpy()
    active_either = active_base | active_target
    n_active = active_either.sum()

    # Per-feature Jaccard for active features only
    active_jaccard = per_feature_jaccard[active_either]

    # Results
    print(f"\n{'='*60}")
    if self_comparison:
        print("SELF-COMPARISON RESULTS")
    else:
        print("SAE COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Tokens analyzed:     {pre_topk_base.shape[0]}")
    print(f"Total features:      {pre_topk_base.shape[1]}")
    print(f"Active features:     {n_active} ({100*n_active/pre_topk_base.shape[1]:.1f}%)")
    print(f"")
    print(f"Global Jaccard:              {global_jaccard:.6f}")
    print(f"Per-feature mean (all):      {np.mean(per_feature_jaccard):.6f}")
    print(f"Per-feature mean (active):   {np.mean(active_jaccard):.6f}")
    print(f"Per-feature min (active):    {np.min(active_jaccard):.6f}")
    print(f"Per-feature max (active):    {np.max(active_jaccard):.6f}")
    print(f"Per-feature std (active):    {np.std(active_jaccard):.6f}")
    print(f"")
    print(f"L0 (base):           {l0_base:.2f}")
    print(f"L0 (target):         {l0_target:.2f}")
    print(f"L0 ratio:            {l0_target/l0_base if l0_base > 0 else 0:.4f}")
    print(f"{'='*60}")

    if self_comparison:
        # For self-comparison, check active features only
        if global_jaccard == 1.0 and np.allclose(active_jaccard, 1.0):
            print("✓ SANITY CHECK PASSED: Self-comparison yields Jaccard = 1.0 (active features)")
        else:
            print("✗ SANITY CHECK FAILED: Expected all 1.0 for active features")
            non_one = np.where(~np.isclose(active_jaccard, 1.0))[0]
            if len(non_one) > 0:
                print(f"  Active features with Jaccard != 1.0: {len(non_one)}")
                print(f"  Example indices: {non_one[:10]}...")
                print(f"  Example values: {active_jaccard[non_one[:10]]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare SAEs using real activations (each SAE on its corresponding model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # For self-comparison (single model)
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model (for self-comparison, use this instead of --model_base)")

    # For cross-comparison (two models)
    parser.add_argument("--model_base", type=str, default=None,
                        help="Path to base model (e.g., BF16)")
    parser.add_argument("--model_target", type=str, default=None,
                        help="Path to target model (e.g., FP8)")

    # SAE paths
    parser.add_argument("--base", type=str, required=True,
                        help="Path to base SAE checkpoint")
    parser.add_argument("--target", type=str, default=None,
                        help="Path to target SAE checkpoint (optional for self-comparison)")

    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of text samples to use")

    args = parser.parse_args()

    # Resolve model paths
    if args.model:
        # Self-comparison mode: use same model for both
        model_base_path = args.model
        model_target_path = args.model if args.target is None else args.model_target
    elif args.model_base:
        model_base_path = args.model_base
        model_target_path = args.model_target
    else:
        parser.error("Must specify either --model (for self-comparison) or --model_base (for cross-comparison)")

    compare_saes(model_base_path, args.base, model_target_path, args.target, args.num_samples)
