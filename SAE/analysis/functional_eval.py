import torch
import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
from tqdm import tqdm

# Add experiment root to path for basic_interpretability.loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from basic_interpretability.loader import load_model_for_lens
    from SAE.analysis.quantization_impact_study import load_sae_checkpoint, compute_similarity_matrix, get_decoder_weights
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running from /experiment/SAE/analysis or similar.")
    sys.exit(1)


def get_hook_name(sae):
    """Extract hook name from SAE config."""
    hook_name = getattr(sae.cfg, "hook_name", getattr(sae.cfg, "hook_point", None))
    if hook_name is None and hasattr(sae.cfg, "metadata"):
        metadata = sae.cfg.metadata
        if isinstance(metadata, dict):
            hook_name = metadata.get("hook_name", metadata.get("hook_point"))
        else:
            hook_name = getattr(metadata, "hook_name", getattr(metadata, "hook_point", None))
    return hook_name


def get_pre_topk_activations(model, sae, texts, batch_size=4):
    """
    Runs text through Model -> SAE encoder and captures PRE-TOPK feature activations.

    For TopK SAEs, the encode() method applies sparsity selection which is too sensitive
    for Jaccard comparison. This function returns the raw encoder output (before TopK),
    which represents the "activation strength" of each feature.

    Returns:
        pre_topk: Tensor [N_tokens, d_sae] - raw encoder outputs (before TopK/ReLU)
        encoded: Tensor [N_tokens, d_sae] - sparse encoded outputs (after TopK/ReLU)
    """
    model.eval()
    sae.eval()

    all_pre_topk = []
    all_encoded = []

    sae_d_in = sae.cfg.d_in if hasattr(sae, "cfg") else getattr(sae, "d_in", None)
    hook_name = get_hook_name(sae)

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # 1. Run Model
            tokens = model.to_tokens(batch_texts)
            _, cache = model.run_with_cache(tokens, stop_at_layer=model.cfg.n_layers)

            # Find target activation
            target_act = None
            if hook_name and hook_name in cache:
                target_act = cache[hook_name]
            else:
                # Search for matching dimension
                possible_keys = sorted(
                    [k for k in cache.keys() if sae_d_in == cache[k].shape[-1]],
                    key=lambda k: ("resid_post" not in k, "resid" not in k, k)
                )
                if possible_keys:
                    target_act = cache[possible_keys[0]]

            if target_act is None:
                print("Warning: Could not find matching activations.")
                target_act = list(cache.values())[-1]

            # 2. Flatten: [batch*pos, d_in]
            flat_act = target_act.reshape(-1, target_act.shape[-1])

            # 3. Compute PRE-TopK activations (raw encoder output)
            apply_b_dec = getattr(sae.cfg, "apply_b_dec_to_input", True)
            if apply_b_dec:
                x = flat_act - sae.b_dec
            else:
                x = flat_act

            pre_topk = x @ sae.W_enc + sae.b_enc  # [N, d_sae]
            all_pre_topk.append(pre_topk.cpu())

            # 4. Also get the actual encoded output (for L0 stats)
            if hasattr(sae, "encode"):
                encoded = sae.encode(flat_act)
                all_encoded.append(encoded.cpu())

            del tokens, cache, target_act, flat_act, pre_topk
            if 'encoded' in dir():
                del encoded
            torch.cuda.empty_cache()

    return torch.cat(all_pre_topk, dim=0), torch.cat(all_encoded, dim=0)


def compute_pre_topk_jaccard(pre_topk_base, pre_topk_target, mapping_indices, threshold=0.0):
    """
    Computes Jaccard similarity using PRE-TopK activations with a threshold.

    This measures the overlap of features that WOULD activate (with ReLU/threshold),
    independent of TopK selection. This is robust to the sensitivity of TopK-32.

    Args:
        pre_topk_base: [N_tokens, d_sae] raw encoder outputs for base SAE
        pre_topk_target: [N_tokens, d_sae] raw encoder outputs for target SAE
        mapping_indices: [d_base] indices mapping base features to target features
        threshold: activation threshold (default 0.0 = ReLU-like)

    Returns:
        per_feature_jaccard: [d_sae] Jaccard score for each feature
        global_jaccard: scalar, mean Jaccard across all tokens
    """
    # Apply threshold to get binary activations
    base_active = pre_topk_base > threshold  # [N, d_sae]
    target_active = pre_topk_target > threshold

    # Global Jaccard (across all features and tokens)
    intersection = (base_active & target_active).sum(dim=1).float()
    union = (base_active | target_active).sum(dim=1).float()
    global_jaccard = (intersection / union.clamp(min=1)).mean().item()

    # Per-feature Jaccard (for matched pairs)
    n_features = pre_topk_base.shape[1]
    per_feature_jaccard = np.zeros(n_features)

    for i in range(n_features):
        target_idx = mapping_indices[i].item() if hasattr(mapping_indices[i], 'item') else mapping_indices[i]
        b = base_active[:, i]
        t = target_active[:, target_idx]

        inter = (b & t).sum().item()
        uni = (b | t).sum().item()
        per_feature_jaccard[i] = inter / uni if uni > 0 else 0.0

    return per_feature_jaccard, global_jaccard


def compute_activation_correlation(pre_topk_base, pre_topk_target, mapping_indices, n_samples=2000):
    """
    Computes Spearman correlation of activation magnitudes for matched feature pairs.

    This measures whether features that fire strongly in the base SAE also fire
    strongly in the target SAE. Unlike Jaccard, this captures magnitude relationships.

    Args:
        pre_topk_base: [N_tokens, d_sae] raw encoder outputs for base SAE
        pre_topk_target: [N_tokens, d_sae] raw encoder outputs for target SAE
        mapping_indices: [d_base] indices mapping base features to target features
        n_samples: number of features to sample (for speed)

    Returns:
        correlations: [n_samples] Spearman correlations for sampled features
        mean_correlation: scalar, mean correlation
    """
    n_features = pre_topk_base.shape[1]
    sample_idx = np.random.choice(n_features, min(n_samples, n_features), replace=False)

    correlations = []
    for i in sample_idx:
        target_idx = mapping_indices[i].item() if hasattr(mapping_indices[i], 'item') else mapping_indices[i]

        base_acts = pre_topk_base[:, i].numpy()
        target_acts = pre_topk_target[:, target_idx].numpy()

        # Spearman correlation (rank-based, robust to outliers)
        corr, _ = spearmanr(base_acts, target_acts)
        if not np.isnan(corr):
            correlations.append(corr)

    return np.array(correlations), np.mean(correlations) if correlations else 0.0


def compute_weighted_jaccard(pre_topk_base, pre_topk_target, mapping_indices):
    """
    Computes magnitude-weighted Jaccard similarity.

    Instead of binary overlap, this weights by activation magnitude:
    WeightedJaccard = sum(min(a, b)) / sum(max(a, b))

    This captures "how much" features overlap, not just "whether" they overlap.
    """
    # Apply ReLU to get positive activations only
    relu_base = torch.relu(pre_topk_base)
    relu_target = torch.relu(pre_topk_target)

    # Reindex target to match base features
    relu_target_matched = relu_target[:, mapping_indices]

    min_acts = torch.minimum(relu_base, relu_target_matched)
    max_acts = torch.maximum(relu_base, relu_target_matched)

    weighted_jaccard = (min_acts.sum(dim=1) / max_acts.sum(dim=1).clamp(min=1e-6)).mean().item()
    return weighted_jaccard


def compute_model_activation_similarity(model_base, model_target, texts, layer_indices=None, device="cuda"):
    """
    Computes similarity between raw model activations (before SAE encoding).

    This isolates the effect of quantization on model representations from
    SAE training effects.

    Args:
        model_base: Baseline model (e.g., BF16)
        model_target: Target model (e.g., FP8 quantized)
        texts: List of text samples
        layer_indices: Which layers to compare. If None, uses [0, n//2, 3n//4]
        device: Device to use

    Returns:
        Dict with per-layer statistics: mean_abs_diff, max_diff, correlation
    """
    model_base.eval()
    model_target.eval()

    n_layers = model_base.cfg.n_layers
    if layer_indices is None:
        layer_indices = [0, n_layers // 2, int(n_layers * 0.75)]

    results = {}

    with torch.no_grad():
        for layer_idx in layer_indices:
            all_base_acts = []
            all_target_acts = []

            for text in texts[:10]:  # Use subset for speed
                tokens = model_base.to_tokens([text])

                # Get activations from base model
                _, cache_base = model_base.run_with_cache(tokens, stop_at_layer=layer_idx + 1)
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                if hook_name not in cache_base:
                    # Try alternative naming
                    possible = [k for k in cache_base.keys() if f"{layer_idx}" in k and "resid" in k]
                    hook_name = possible[0] if possible else list(cache_base.keys())[-1]

                act_base = cache_base[hook_name].flatten(0, 1)  # [batch*seq, d_model]

                # Get activations from target model
                tokens_t = model_target.to_tokens([text])
                _, cache_target = model_target.run_with_cache(tokens_t, stop_at_layer=layer_idx + 1)
                if hook_name not in cache_target:
                    possible = [k for k in cache_target.keys() if f"{layer_idx}" in k and "resid" in k]
                    hook_name = possible[0] if possible else list(cache_target.keys())[-1]

                act_target = cache_target[hook_name].flatten(0, 1)

                # Align shapes (take min length)
                min_len = min(act_base.shape[0], act_target.shape[0])
                all_base_acts.append(act_base[:min_len].cpu())
                all_target_acts.append(act_target[:min_len].cpu())

                del cache_base, cache_target
                torch.cuda.empty_cache()

            # Concatenate all activations
            base_acts = torch.cat(all_base_acts, dim=0).float()
            target_acts = torch.cat(all_target_acts, dim=0).float()

            # Compute statistics
            diff = (base_acts - target_acts).abs()
            mean_abs_diff = diff.mean().item()
            max_diff = diff.max().item()

            # Correlation (flatten and compute)
            base_flat = base_acts.flatten()
            target_flat = target_acts.flatten()
            correlation = torch.corrcoef(torch.stack([base_flat, target_flat]))[0, 1].item()

            results[layer_idx] = {
                "mean_abs_diff": mean_abs_diff,
                "max_diff": max_diff,
                "correlation": correlation
            }

    return results


def run_eval(model, sae_base, sae_target, texts, mapping_indices, device="cuda", model_target=None):
    """
    Runs functional evaluation comparing base and target SAEs.

    Computes:
    - Pre-TopK Jaccard: Overlap of features that would activate (ReLU-like)
    - Activation Correlation: Spearman correlation of activation magnitudes
    - Weighted Jaccard: Magnitude-weighted overlap
    - L0 statistics: Sparsity of encoded outputs
    - Model activation similarity: Raw model output comparison (if model_target provided)

    Args:
        model: Model for base SAE (and target SAE if model_target is None)
        sae_base: Base SAE (typically trained on baseline precision, e.g., BF16)
        sae_target: Target SAE (typically trained on quantized model)
        texts: List of text samples
        mapping_indices: Feature mapping from geometric analysis
        device: Device to use
        model_target: Optional separate model for target SAE. If provided, base SAE
                      uses `model` and target SAE uses `model_target`. This ensures
                      each SAE receives activations from the model it was trained on.
    """
    # Compute model activation similarity if we have two different models
    model_act_similarity = None
    if model_target is not None:
        print("    Computing model activation similarity...")
        try:
            model_act_similarity = compute_model_activation_similarity(
                model, model_target, texts, device=device
            )
        except Exception as e:
            print(f"    Warning: Model activation comparison failed: {e}")

    print("    Extracting activations (Base SAE)...")
    pre_topk_base, encoded_base = get_pre_topk_activations(model, sae_base, texts)

    # Use separate model for target if provided, otherwise use same model
    target_model = model_target if model_target is not None else model
    print("    Extracting activations (Target SAE)...")
    pre_topk_target, encoded_target = get_pre_topk_activations(target_model, sae_target, texts)

    # Ensure mapping_indices is on CPU for indexing
    if hasattr(mapping_indices, 'cpu'):
        mapping_indices = mapping_indices.cpu()

    print("    Computing functional metrics...")

    # 1. Pre-TopK Jaccard (main metric)
    per_feature_jaccard, global_jaccard = compute_pre_topk_jaccard(
        pre_topk_base, pre_topk_target, mapping_indices
    )

    # 2. Activation Magnitude Correlation
    correlations, mean_correlation = compute_activation_correlation(
        pre_topk_base, pre_topk_target, mapping_indices
    )

    # 3. Weighted Jaccard
    weighted_jaccard = compute_weighted_jaccard(
        pre_topk_base, pre_topk_target, mapping_indices
    )

    # 4. L0 Sparsity (on encoded outputs)
    l0_base = (encoded_base > 0).float().sum(dim=1).mean().item()
    l0_target = (encoded_target > 0).float().sum(dim=1).mean().item()
    l0_ratio = l0_target / l0_base if l0_base > 0 else 0

    # 5. Dead feature detection
    acts_base_dead = (pre_topk_base.sum(dim=0) <= 0).numpy()
    acts_target_dead = (pre_topk_target[:, mapping_indices].sum(dim=0) <= 0).numpy()

    return {
        # Primary metrics
        "mean_jaccard": global_jaccard,  # Pre-TopK Jaccard (renamed for compatibility)
        "jaccard_scores": per_feature_jaccard,
        "mean_activation_corr": mean_correlation,
        "activation_correlations": correlations,
        "weighted_jaccard": weighted_jaccard,

        # Sparsity metrics
        "l0_base": l0_base,
        "l0_target": l0_target,
        "l0_ratio": l0_ratio,

        # Dead features
        "acts_base_dead": acts_base_dead,
        "acts_target_dead": acts_target_dead,

        # Statistics
        "n_tokens": pre_topk_base.shape[0],
        "n_features": pre_topk_base.shape[1],

        # Model activation similarity (isolates quantization from SAE training)
        "model_activation_similarity": model_act_similarity,
    }

def main():
    parser = argparse.ArgumentParser(description="Functional Consistency Analysis for SAEs")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name (or local path)")
    parser.add_argument("--base_sae", type=str, required=True, help="Path to Baseline SAE checkpoint")
    parser.add_argument("--target_sae", type=str, required=True, help="Path to Target (Quantized) SAE checkpoint")
    parser.add_argument("--output_dir", type=str, default="functional_results")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of text sequences to run")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Functional Analysis for {args.model_name} ---")

    # 1. Load Data
    print("Loading Dataset (wikitext)...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="test")
    texts = [x["text"] for x in dataset if len(x["text"]) > 100][:args.num_samples]

    # 2. Load SAEs
    print("Loading SAEs...")
    sae_base = load_sae_checkpoint(args.base_sae)
    sae_target = load_sae_checkpoint(args.target_sae)

    if not sae_base or not sae_target:
        print("Failed to load SAEs.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_base.to(device)
    sae_target.to(device)

    # 3. Geometric Matching (Prerequisite)
    print("Computing Geometric Matching...")
    w_base = get_decoder_weights(sae_base, device)
    w_target = get_decoder_weights(sae_target, device)

    sim_vals, mapping_indices, _, _ = compute_similarity_matrix(w_base, w_target)

    # 4. Load Model
    print(f"Loading Model: {args.model_name}...")
    try:
        from basic_interpretability.loader import load_model_for_lens
        model = load_model_for_lens(args.model_name, device=device)
    except Exception as e:
        print(f"Model load failed: {e}")
        sys.exit(1)

    # 5. Run Eval
    results = run_eval(model, sae_base, sae_target, texts, mapping_indices, device)

    if results:
        # Save Detailed Stats (per-feature)
        df = pd.DataFrame({
            "BaseFeatureIdx": range(len(results["jaccard_scores"])),
            "TargetMatchIdx": mapping_indices.cpu().numpy() if hasattr(mapping_indices, 'cpu') else mapping_indices,
            "GeometricSim": sim_vals.cpu().numpy() if hasattr(sim_vals, 'cpu') else sim_vals,
            "PreTopK_Jaccard": results["jaccard_scores"],
            "IsDead_Base": results["acts_base_dead"],
            "IsDead_Target": results["acts_target_dead"]
        })

        out_file = os.path.join(args.output_dir, f"{os.path.basename(args.model_name)}_functional_stats.csv")
        df.to_csv(out_file, index=False)
        print(f"Saved detailed stats to {out_file}")

        # Save Summary
        high_sim_mask = df["GeometricSim"] > 0.9
        summary = {
            "Model": args.model_name,
            "N_Tokens": results["n_tokens"],
            "N_Features": results["n_features"],
            # Primary functional metrics
            "PreTopK_Jaccard": results["mean_jaccard"],
            "PreTopK_Jaccard_HighSim": np.mean(results["jaccard_scores"][high_sim_mask]) if high_sim_mask.any() else 0,
            "Activation_Correlation": results["mean_activation_corr"],
            "Weighted_Jaccard": results["weighted_jaccard"],
            # Sparsity
            "L0_Base": results["l0_base"],
            "L0_Target": results["l0_target"],
            "L0_Ratio": results["l0_ratio"],
        }

        summary_file = os.path.join(args.output_dir, "summary.csv")
        pd.DataFrame([summary]).to_csv(
            summary_file, mode='a',
            header=not os.path.exists(summary_file),
            index=False
        )

        # Print summary
        print("\n" + "=" * 50)
        print("FUNCTIONAL CONSISTENCY SUMMARY")
        print("=" * 50)
        print(f"  Pre-TopK Jaccard:        {results['mean_jaccard']:.4f}")
        print(f"  Activation Correlation:  {results['mean_activation_corr']:.4f}")
        print(f"  Weighted Jaccard:        {results['weighted_jaccard']:.4f}")
        print(f"  L0 Ratio:                {results['l0_ratio']:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
