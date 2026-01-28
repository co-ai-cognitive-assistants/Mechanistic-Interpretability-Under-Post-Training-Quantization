import os
import sys
import glob
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
from datasets import load_dataset
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig

# Setup Logging
log_file = os.path.join(os.path.dirname(__file__), "interpretability_run.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from basic_interpretability.loader import load_hf_dequantized
except ImportError:
    # Fallback if running from root
    from basic_interpretability.loader import load_hf_dequantized

# ==========================================
# 1. UTILS & DISCOVERY
# ==========================================

def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()

def scan_models(models_dir: str) -> dict:
    """
    Scans the models directory for families and variants.
    """
    families = {}
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return {}
        
    for family_name in os.listdir(models_dir):
        family_path = os.path.join(models_dir, family_name)
        if not os.path.isdir(family_path):
            continue
            
        variants = {}
        for variant_name in os.listdir(family_path):
            variant_path = os.path.join(family_path, variant_name)
            if os.path.isdir(variant_path):
                variants[variant_name] = variant_path
        
        if not variants:
            continue
            
        baseline_key = None
        for cand in ["bfloat16", "bf16", "base"]:
            if cand in variants:
                baseline_key = cand
                break
        
        if not baseline_key:
            logger.warning(f"No clear baseline (bfloat16/bf16/base) found for {family_name}. Skipping.")
            continue
            
        baseline_path = variants.pop(baseline_key)
        
        families[family_name] = {
            "baseline": {"name": baseline_key, "path": baseline_path},
            "targets": variants,
            "size_bytes": sum(os.path.getsize(os.path.join(baseline_path, f)) for f in os.listdir(baseline_path) if os.path.isfile(os.path.join(baseline_path, f)))
        }
    
    return dict(sorted(families.items(), key=lambda item: item[1]["size_bytes"]))

def get_dataset_samples(dataset_name="wikitext", num_samples=20, seq_len=128, seed=42):
    """Loads dataset samples."""
    logger.info(f"Loading {num_samples} samples from {dataset_name}...")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        ds = ds.filter(lambda x: len(x["text"]) > 100)
        indices = np.random.RandomState(seed).permutation(len(ds))[:num_samples]
        return [ds[int(i)]["text"] for i in indices]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return ["The capital of France is Paris."] * num_samples

def perform_sanity_check(model):
    """Checks if model produces valid logits."""
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    test_text = "The quick brown fox jumps over the lazy dog."
    try:
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return False, "Logits contain NaN or Inf"
            
        probs = F.softmax(logits[0, -1, :], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        if entropy > 12.0: # Highly uniform noise
            return False, f"High Entropy ({entropy:.2f})"
            
        return True, "Passed"
    except Exception as e:
        return False, str(e)

# ==========================================
# 2. ANALYSIS CORE (Unified HF Hooks)
# ==========================================

def run_logit_lens_analysis(model, samples, seq_len=128, reference_indices=None):
    """Unified Logit Lens using HF Hooks."""
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    
    # Identify target components
    inner = None
    layers = None
    ln_final = None
    
    if hasattr(model, "model"):
        inner = model.model
    elif hasattr(model, "transformer"):
        inner = model.transformer
    elif hasattr(model, "backbone"):
        inner = model.backbone
    
    if inner is not None:
        if hasattr(inner, "layers"):
            layers = inner.layers
        elif hasattr(inner, "h"):
            layers = inner.h
        elif hasattr(inner, "blocks"):
            layers = inner.blocks
        
        if hasattr(inner, "norm"):
            ln_final = inner.norm
        elif hasattr(inner, "ln_f"):
            ln_final = inner.ln_f
    
    # Fallback: check model itself if inner didn't yield layers
    if layers is None:
        if hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "h"):
            layers = model.h
    
    if layers is None:
        # Last ditch effort for specific architectures or simply fail with debug info
        attrs = dir(inner) if inner else dir(model)
        raise AttributeError(f"Unsupported model structure for hooks. Available attributes: {attrs[:20]}...")

    unembed = model.lm_head
    n_layers = len(layers)
    K = 10
    
    all_indices = []
    all_probs = []
    
    for i, text in enumerate(tqdm(samples, desc="Logit Lens", leave=False)):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).to(model.device)
        curr_seq_len = inputs.input_ids.shape[1]
        
        sample_indices = np.zeros((n_layers, curr_seq_len, K), dtype=np.int32)
        sample_probs = np.zeros((n_layers, curr_seq_len, K), dtype=np.float32)
        
        layer_activations = {}
        hooks = []
        def get_hook(l):
            def hook(module, input, output):
                # Standard HF DecoderLayer output is (hidden_states, ...)
                # Some models might return just hidden_states
                if isinstance(output, tuple):
                    layer_activations[l] = output[0].detach()
                else:
                    layer_activations[l] = output.detach()
            return hook
            
        for l in range(n_layers):
            hooks.append(layers[l].register_forward_hook(get_hook(l)))
            
        with torch.no_grad():
            model(**inputs)
            
        for h in hooks: h.remove()
        
        ref_idxs_list = reference_indices[i] if reference_indices is not None else None
        
        # Run projection in no_grad to avoid gradient tracking on parameters
        with torch.no_grad():
            for l in range(n_layers):
                resid = layer_activations[l]
                # Apply final norm if it exists, otherwise skip (some models might integrate it differently)
                if ln_final:
                    normed = ln_final(resid)
                else:
                    normed = resid
                
                logits = unembed(normed)
                probs = F.softmax(logits, dim=-1) # [1, seq, vocab]
                
                if ref_idxs_list is not None:
                    target_ids = torch.tensor(ref_idxs_list[l], device=probs.device).unsqueeze(0).long()
                    selected_probs = torch.gather(probs, -1, target_ids)
                    sample_probs[l] = selected_probs.squeeze(0).float().cpu().numpy()
                else:
                    topk = torch.topk(probs, K, dim=-1)
                    sample_indices[l] = topk.indices.squeeze(0).cpu().numpy()
                    sample_probs[l] = topk.values.squeeze(0).float().cpu().numpy()
                
        all_indices.append(sample_indices)
        all_probs.append(sample_probs)
        
    return all_indices, all_probs

def run_attention_analysis(model, samples):
    """Unified Attention Entropy using HF output_attentions."""
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    config = model.config.text_config if hasattr(model.config, "text_config") else model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    
    head_entropies = torch.zeros((n_layers, n_heads), device="cpu")
    count = 0
    
    for i, text in enumerate(tqdm(samples[:10], desc="Attn Entropy", leave=False)):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
        if outputs.attentions is None:
            logger.warning("Model did not return attentions. Ensure attn_implementation='eager'.")
            return None

        for l in range(n_layers):
            attn = outputs.attentions[l] # [1, head, Q, K]
            eps = 1e-9
            entropy = -torch.sum(attn * torch.log(attn + eps), dim=-1).mean(dim=[0, 2])
            head_entropies[l] += entropy.cpu()
        count += 1
        
    avg = head_entropies / count
    stats = []
    flat = avg.flatten().numpy()
    stats.append({"scope": "global", "layer": -1, "mean": np.mean(flat), "std": np.std(flat)})
    for l in range(n_layers):
        stats.append({"scope": "layer", "layer": l, "mean": np.mean(avg[l].numpy())})
    return pd.DataFrame(stats)

def compute_kl(base_probs, target_probs):
    eps = 1e-9
    num_layers = base_probs[0].shape[0]
    stats = []
    for l in range(num_layers):
        all_vals = []
        for b_p, t_p in zip(base_probs, target_probs):
            p = b_p[l] / (b_p[l].sum(axis=-1, keepdims=True) + eps)
            q = t_p[l] / (t_p[l].sum(axis=-1, keepdims=True) + eps)
            kl = np.sum(p * np.log((p + eps) / (q + eps)), axis=-1)
            all_vals.extend(kl.flatten())
        stats.append({"layer": l, "kl_mean": np.mean(all_vals), "kl_std": np.std(all_vals)})
    return pd.DataFrame(stats)

# ==========================================
# 3. ORCHESTRATION
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--output_dir", type=str, default="basic_interpretability/results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    families = scan_models(args.models_dir)
    samples = get_dataset_samples()
    
    for fam_name, info in families.items():
        run_dir = os.path.join(args.output_dir, f"run_{fam_name}")
        os.makedirs(run_dir, exist_ok=True)
        t1_out = os.path.join(run_dir, "tier1", "logit_lens_kl.csv")
        if os.path.exists(t1_out) and not args.force:
            logger.info(f"Skipping {fam_name}, results exist.")
            continue
            
        logger.info(f"\n>>> Analyzing Family: {fam_name}")
        base_model = None
        target_model = None
        try:
            # Baseline
            base_model = load_hf_dequantized(info["baseline"]["path"], device=args.device)
            base_indices, base_probs = run_logit_lens_analysis(base_model, samples)
            base_attn = run_attention_analysis(base_model, samples)
            if base_attn is not None:
                base_attn["method"] = info["baseline"]["name"]
            
            del base_model
            base_model = None
            cleanup_memory()
            
            t1_results = []
            t2_results = []
            if base_attn is not None:
                t2_results.append(base_attn)
            
            # Targets
            for target_name, target_path in info["targets"].items():
                logger.info(f"--- Target: {target_name} ---")
                target_model = None
                try:
                    target_model = load_hf_dequantized(target_path, device=args.device)
                    passed, reason = perform_sanity_check(target_model)
                    if not passed:
                        logger.warning(f"Sanity Check Failed for {target_name}: {reason}")
                        del target_model
                        target_model = None
                        cleanup_memory()
                        continue
                        
                    _, target_probs = run_logit_lens_analysis(target_model, samples, reference_indices=base_indices)
                    kl_df = compute_kl(base_probs, target_probs)
                    kl_df["method"] = target_name
                    t1_results.append(kl_df)
                    
                    attn_df = run_attention_analysis(target_model, samples)
                    if attn_df is not None:
                        attn_df["method"] = target_name
                        t2_results.append(attn_df)
                    
                    del target_model
                    target_model = None
                    cleanup_memory()
                except Exception as e:
                    logger.error(f"Error on {target_name}: {e}")
                    if target_model is not None:
                        del target_model
                        target_model = None
                    cleanup_memory()
            
            # Save
            if t1_results:
                os.makedirs(os.path.join(run_dir, "tier1"), exist_ok=True)
                pd.concat(t1_results).to_csv(t1_out, index=False)
            if t2_results:
                os.makedirs(os.path.join(run_dir, "tier2"), exist_ok=True)
                pd.concat(t2_results).to_csv(os.path.join(run_dir, "tier2", "attention_entropy.csv"), index=False)
                
        except Exception as e:
            logger.error(f"Family {fam_name} failed: {e}")
            if base_model is not None:
                del base_model
            if target_model is not None:
                del target_model
            cleanup_memory()

if __name__ == "__main__":
    main()