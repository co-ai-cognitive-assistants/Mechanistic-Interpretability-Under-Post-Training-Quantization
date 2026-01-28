import torch
import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy.linalg import svdvals
from sae_lens import SAE, MatryoshkaBatchTopKTrainingSAE

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

# Add SAE directory to path to handle imports of custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set plotting style
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="whitegrid", context="paper")

class LegacySAE(torch.nn.Module):
    """Wrapper for SAEs that can't be loaded via SAELens (deprecated or custom)."""
    def __init__(self, state_dict, device="cpu", cfg=None):
        super().__init__()
        self.device = device

        # 1. Determine Weights
        if "model_state_dict" in state_dict:
            sd = state_dict["model_state_dict"]
            # Map keys: encoder.weight [d_sae, d_in], decoder_weight [d_sae, d_in]
            # SAELens internal W_enc is [d_in, d_sae]
            self.W_enc = torch.nn.Parameter(sd["encoder.weight"].T.float().to(device))
            self.b_enc = torch.nn.Parameter(sd["encoder.bias"].float().to(device))
            self.W_dec = torch.nn.Parameter(sd["decoder_weight"].float().to(device))
            self.b_dec = torch.nn.Parameter(sd["b_dec"].float().to(device))
        else:
            # Standard SAELens keys: W_enc [d_in, d_sae], W_dec [d_sae, d_in]
            self.W_enc = torch.nn.Parameter(state_dict["W_enc"].float().to(device))
            self.b_enc = torch.nn.Parameter(state_dict["b_enc"].float().to(device))
            self.W_dec = torch.nn.Parameter(state_dict["W_dec"].float().to(device))
            self.b_dec = torch.nn.Parameter(state_dict["b_dec"].float().to(device))

        # 2. Setup Metadata
        self.d_in = self.W_enc.shape[0]
        self.d_sae = self.W_enc.shape[1]

        # 3. Extract SAE config - check multiple sources
        from types import SimpleNamespace
        extracted_cfg = {"d_in": self.d_in, "d_sae": self.d_sae, "architecture": "standard"}

        # Try to extract from embedded checkpoint config (old .pt format)
        if "config" in state_dict and hasattr(state_dict["config"], "sae"):
            sae_cfg = state_dict["config"].sae
            # Extract relevant fields from the embedded SAE config
            for attr in ["architecture", "activation_fn", "sparsity_k", "k", "apply_b_dec_to_input"]:
                if hasattr(sae_cfg, attr):
                    extracted_cfg[attr] = getattr(sae_cfg, attr)
            print(f"    Extracted SAE config from checkpoint: arch={extracted_cfg.get('architecture')}, "
                  f"activation_fn={extracted_cfg.get('activation_fn')}, k={extracted_cfg.get('sparsity_k', extracted_cfg.get('k'))}")

        # Override with explicitly provided cfg
        if cfg:
            if isinstance(cfg, dict):
                extracted_cfg.update(cfg)
            else:
                for attr in dir(cfg):
                    if not attr.startswith('_'):
                        extracted_cfg[attr] = getattr(cfg, attr)

        self.cfg = SimpleNamespace(**extracted_cfg)

    def _is_topk(self):
        """Check if this SAE uses TopK activation."""
        arch = str(getattr(self.cfg, "architecture", "")).lower()
        act_fn = str(getattr(self.cfg, "activation_fn", "")).lower()
        return "topk" in arch or "topk" in act_fn

    def _get_k(self):
        """Get the k value for TopK activation."""
        return getattr(self.cfg, "sparsity_k", getattr(self.cfg, "k", 32))

    def encode(self, x):
        # x: [batch, d_in]
        apply_b_dec = getattr(self.cfg, "apply_b_dec_to_input", True)
        if apply_b_dec:
            x = x - self.b_dec

        sae_out = x @ self.W_enc + self.b_enc

        if self._is_topk():
            k = self._get_k()
            topk_vals, topk_indices = torch.topk(sae_out, k=min(k, sae_out.shape[-1]), dim=-1)
            res = torch.zeros_like(sae_out)
            res.scatter_(-1, topk_indices, torch.relu(topk_vals))
            return res
        else:
            return torch.relu(sae_out)

    def forward(self, x):
        acts = self.encode(x)
        return acts @ self.W_dec + self.b_dec

    def to(self, device):
        self.device = device
        super().to(device)
        return self

def load_sae_checkpoint(path: str, device: str = "cpu") -> Optional[torch.nn.Module]:
    """Loads an SAE from a directory or a .pt checkpoint using sae_lens or manual fallback."""
    
    # 1. Check for Directory vs File
    is_dir = os.path.isdir(path)
    if is_dir:
        cfg_path = os.path.join(path, "cfg.json")
    else:
        cfg_path = os.path.join(os.path.dirname(path), "cfg.json")
    
    # 2. Try Native SAELens Loader (with Matryoshka detection)
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r') as f:
                cfg_data = json.load(f)
            
            # Detect Architecture
            arch = str(cfg_data.get("architecture", "")).lower()
            if "matryoshka" in arch:
                SAEClass = MatryoshkaBatchTopKTrainingSAE
                print(f"  Loading Matryoshka SAE from {os.path.basename(path)}...")
            else:
                SAEClass = SAE

            if is_dir:
                sae = SAEClass.load_from_pretrained(path, device=device)
                print(f"    Loaded SAE weights sum: {sae.W_enc.sum().item():.4f}")
                return sae
            else:
                # Load directory base then override with specific checkpoint weights
                sae = SAEClass.load_from_pretrained(os.path.dirname(path), device=device)
                
                state_dict = torch.load(path, map_location=device, weights_only=False)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                
                # Clean prefix if needed
                if isinstance(state_dict, dict):
                    if any(k.startswith("sae.") for k in state_dict.keys()):
                        state_dict = {k.replace("sae.", ""): v for k, v in state_dict.items()}
                    
                    sae.load_state_dict(state_dict, strict=False)
                return sae
        except Exception as e:
            print(f"    Native load failed: {e}. Attempting legacy fallback...")

    # 3. Legacy Fallback (Handles standalone .pt or broken folders)
    try:
        state_dict = None
        cfg = None
        
        # Load Weights
        if is_dir:
            w_path_st = os.path.join(path, "sae_weights.safetensors")
            w_path_pt = os.path.join(path, "sae_weights.pt")
            if os.path.exists(w_path_st) and load_safetensors:
                state_dict = load_safetensors(w_path_st)
            elif os.path.exists(w_path_pt):
                state_dict = torch.load(w_path_pt, map_location="cpu", weights_only=False)
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
            
        # Load Config if available
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)

        if state_dict:
            return LegacySAE(state_dict, device=device, cfg=cfg)
            
    except Exception as e:
        print(f"    Legacy load failed for {path}: {e}")
        
    return None

def find_checkpoints(root_dir: str) -> Dict[str, Dict[str, str]]:
    """Scans directory for checkpoints."""
    print(f"Scanning {root_dir} for checkpoints...")
    checkpoints = {}
    subdirs = glob.glob(os.path.join(root_dir, "*"))
    
    for subdir in subdirs:
        if not os.path.isdir(subdir):
            continue
        dirname = os.path.basename(subdir)
        methods = ["bfloat16", "float16", "fp32", "int4", "int8", "fp8", "awq", "gptq", "hqq", "gguf"]
        
        detected_method = "unknown"
        for m in methods:
            # Check for delimiters _ or - around the method name
            if f"_{m}_" in dirname or f"-{m}_" in dirname or dirname.endswith(f"_{m}") or dirname.endswith(f"-{m}"):
                detected_method = m
                break
        
        if detected_method != "unknown":
            # Split by either _method_ or -method_
            if f"_{detected_method}_" in dirname:
                model_name = dirname.split(f"_{detected_method}_")[0]
            elif f"-{detected_method}_" in dirname:
                model_name = dirname.split(f"-{detected_method}_")[0]
            else:
                model_name = dirname.rsplit(f"_{detected_method}", 1)[0] if dirname.endswith(f"_{detected_method}") else dirname.rsplit(f"-{detected_method}", 1)[0]
        else:
            model_name = dirname
            
        # Search for .pt files
        pt_files = glob.glob(os.path.join(subdir, "**/checkpoint_*.pt"), recursive=True)
        
        # Search for SAELens directories (containing cfg.json)
        # We look for directories that have cfg.json
        # This might find 'final_1000001536' folders
        saelens_dirs = []
        for root, dirs, files in os.walk(subdir):
            if "cfg.json" in files and ("sae_weights.safetensors" in files or "sae_weights.pt" in files):
                saelens_dirs.append(root)
        
        all_candidates = pt_files + saelens_dirs
        
        if not all_candidates:
            continue
            
        # Helper to extract step number
        def get_step(path):
            name = os.path.basename(path)
            # Try checkpoint_123.pt
            if name.startswith("checkpoint_") and name.endswith(".pt"):
                try:
                    return int(name.split("_")[-1].split(".")[0])
                except:
                    pass
            # Try final_123 directory
            if name.startswith("final_"):
                try:
                    return int(name.split("_")[-1])
                except:
                    pass
            # Try simple number directory
            try:
                return int(name)
            except:
                pass
            return 0

        latest_ckpt = max(all_candidates, key=get_step)
            
        if model_name not in checkpoints:
            checkpoints[model_name] = {}
        checkpoints[model_name][detected_method] = latest_ckpt
        
    return checkpoints

def get_decoder_weights(sae, device="cpu"):
    # Check for W_dec (sae_lens standard)
    w = None
    if hasattr(sae, "W_dec"):
        w = sae.W_dec
    elif hasattr(sae, "decoder_weight"):
        w = sae.decoder_weight
        
    if w is not None:
        if isinstance(w, torch.Tensor):
            return w.detach().to(device)
        # If it's a Parameter, it's also a Tensor
        return w.data.detach().to(device)
        
    return None

def compute_similarity_matrix(w1, w2, chunk_size=1024):
    """
    Computes max similarity bidirectionally.
    Returns:
        sim_1_to_2: Best match in w2 for each vector in w1 (Recall-ish)
        sim_2_to_1: Best match in w1 for each vector in w2 (Precision-ish)
        idx_1_to_2: Indices of best matches
        idx_2_to_1: Indices of best matches
    """
    w1_norm = torch.nn.functional.normalize(w1, p=2, dim=1)
    w2_norm = torch.nn.functional.normalize(w2, p=2, dim=1)
    
    d_sae = w1.shape[0]
    
    sim_1_to_2 = []
    idx_1_to_2 = []
    
    # 1 -> 2
    for i in range(0, d_sae, chunk_size):
        end = min(i + chunk_size, d_sae)
        chunk = w1_norm[i:end]
        sim_chunk = torch.matmul(chunk, w2_norm.T)
        batch_max, batch_idx = torch.max(sim_chunk, dim=1)
        sim_1_to_2.append(batch_max.cpu())
        idx_1_to_2.append(batch_idx.cpu())

    sim_1_to_2 = torch.cat(sim_1_to_2)
    idx_1_to_2 = torch.cat(idx_1_to_2)
    
    # 2 -> 1 (Inverse)
    # We can optimize this by reusing calculations if memory allows, but chunking is safer
    d_sae_2 = w2.shape[0]
    sim_2_to_1 = []
    idx_2_to_1 = []
    
    for i in range(0, d_sae_2, chunk_size):
        end = min(i + chunk_size, d_sae_2)
        chunk = w2_norm[i:end]
        sim_chunk = torch.matmul(chunk, w1_norm.T)
        batch_max, batch_idx = torch.max(sim_chunk, dim=1)
        sim_2_to_1.append(batch_max.cpu())
        idx_2_to_1.append(batch_idx.cpu())

    sim_2_to_1 = torch.cat(sim_2_to_1)
    idx_2_to_1 = torch.cat(idx_2_to_1)
    
    return sim_1_to_2, idx_1_to_2, sim_2_to_1, idx_2_to_1

def compute_spectral_stats(weights):
    """Computes effective rank and singular value stats."""
    # Move to CPU for SVD to save GPU memory
    w_cpu = weights.float().cpu().numpy()
    
    # Sanitize inputs (NaN/Inf check)
    if not np.isfinite(w_cpu).all():
        print("    Warning: Weights contain NaNs or Infs! Patching with zeros for SVD.")
        w_cpu = np.nan_to_num(w_cpu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # SVD
    try:
        s = svdvals(w_cpu)
    except Exception as e:
        print(f"    SVD Calculation failed: {e}")
        return {
            "singular_values": np.zeros(10), # Dummy
            "effective_rank": 0.0,
            "spectral_ratio": 0.0
        }
    
    # Effective Rank (Entropy)
    s_norm = s / np.sum(s)
    # Avoid log(0)
    s_norm = s_norm[s_norm > 1e-10]
    
    if len(s_norm) == 0:
        entropy = 0
    else:
        entropy = -np.sum(s_norm * np.log(s_norm))
        
    effective_rank = np.exp(entropy)
    
    return {
        "singular_values": s,
        "effective_rank": effective_rank,
        "spectral_ratio": s[0] / s[-1] if (len(s) > 0 and s[-1] > 1e-6) else 0
    }

def compute_metrics(model_name: str, methods_dict: Dict[str, str], baseline_method: str, device: str = "cpu") -> Tuple[pd.DataFrame, Dict[str, torch.Tensor]]:
    """Computes geometric and spectral metrics for a group of SAEs."""
    print(f"  Baseline: {baseline_method}")
    base_sae = load_sae_checkpoint(methods_dict[baseline_method])
    if not base_sae: 
        return pd.DataFrame(), {}

    base_sae.to(device)
    w_base = get_decoder_weights(base_sae, device)
    
    # Analyze Baseline Spectrum
    base_spectral = compute_spectral_stats(w_base)
    
    stats = []
    # Store mappings: {method: (idx_recall, idx_precision)}
    mappings = {}
    
    # Add baseline entry
    stats.append({
        "Model": model_name,
        "Method": f"{baseline_method} (Base)",
        "Baseline": baseline_method,
        "Recall_Mean": 1.0,
        "Precision_Mean": 1.0,
        "Effective_Rank": base_spectral["effective_rank"],
        "Rank_Retention": 1.0,
        "Unique_Utilization": 1.0,
        "SingularValues": base_spectral["singular_values"] # Store for plotting later
    })

    for method, path in methods_dict.items():
        if method == baseline_method:
            continue
            
        target_sae = load_sae_checkpoint(path)
        if not target_sae: continue
        target_sae.to(device)
        w_target = get_decoder_weights(target_sae, device)
        
        print(f"    Comparing {method} vs {baseline_method}...")
        
        # 1. Similarity Analysis
        sim_recall, idx_recall, sim_precision, idx_precision = compute_similarity_matrix(w_base, w_target)
        
        # Store mappings for functional eval
        mappings[method] = {
            "idx_base_to_target": idx_recall, 
            "idx_target_to_base": idx_precision,
            "sim_base_to_target": sim_recall
        }
        
        # 2. Spectral Analysis
        target_spectral = compute_spectral_stats(w_target)
        
        # 3. Collapse/Split Analysis
        unique_targets_hit = torch.unique(idx_recall).numel()
        d_sae = w_base.shape[0]

        # Calculate AUC-like metrics for robustness
        recall_thresholds = [0.7, 0.8, 0.9, 0.95, 0.99]
        recall_at_k = {}
        for t in recall_thresholds:
            recall_at_k[f"Recall_GT_{t}"] = (sim_recall > t).float().mean().item()
        
        # Metrics
        entry = {
            "Model": model_name,
            "Method": method,
            "Baseline": baseline_method,
            
            # Consistency (Recall)
            "Recall_Mean": sim_recall.mean().item(),
            **recall_at_k,
            
            # Hallucination (Precision)
            "Precision_Mean": sim_precision.mean().item(),
            "Precision_GT_0.9": (sim_precision > 0.9).float().mean().item(),
            
            # Structural
            "Effective_Rank": target_spectral["effective_rank"],
            "Rank_Retention": target_spectral["effective_rank"] / base_spectral["effective_rank"],
            
            # Topology
            "Dead_Features_Base": (sim_recall < 0.1).float().sum().item(),
            "Dead_Features_Target": (sim_precision < 0.1).float().sum().item(),
            "Unique_Utilization": unique_targets_hit / d_sae,
            
            # Stored for plotting (remove before saving CSV if large)
            "SingularValues": target_spectral["singular_values"]
        }
        stats.append(entry)
        
        # Free memory
        del target_sae, w_target
        torch.cuda.empty_cache()

    return pd.DataFrame(stats), mappings

def generate_plots(model_name: str, df_stats: pd.DataFrame, output_dir: str):
    """Generates visualization plots for the analyzed model."""
    if df_stats.empty: return

    # Drop SingularValues from CSV version
    csv_cols = [c for c in df_stats.columns if c != "SingularValues"]
    df_stats[csv_cols].to_csv(os.path.join(output_dir, f"{model_name}_advanced_stats.csv"), index=False)

    # Filter for plotting
    df_plot = df_stats[df_stats["Recall_Mean"].notna() & (df_stats["Method"] != df_stats["Baseline"] + " (Base)")].copy()

    if df_plot.empty: return

    sns.set_palette("viridis")

    # 1. GEOMETRIC vs FUNCTIONAL GAP - Key figure for the paper
    if "Geometric_Similarity" in df_plot.columns and "Jaccard_Mean" in df_plot.columns:
        plt.figure(figsize=(10, 6))

        methods = df_plot["Method"].tolist()
        x = np.arange(len(methods))
        width = 0.35

        geo_vals = df_plot["Geometric_Similarity"].values
        jac_vals = df_plot["Jaccard_Mean"].values

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, geo_vals, width, label='Geometric Similarity', color='#2ecc71', edgecolor='black')
        bars2 = ax.bar(x + width/2, jac_vals, width, label='Functional Jaccard', color='#e74c3c', edgecolor='black')

        # Add value labels
        for bar, val in zip(bars1, geo_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar, val in zip(bars2, jac_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Quantization Method', fontsize=12)
        ax.set_title(f'Geometric vs Functional Similarity\n{model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        # Add annotation about the gap
        if len(geo_vals) > 0 and len(jac_vals) > 0:
            gap = geo_vals[0] - jac_vals[0]
            ax.annotate(f'Gap: {gap:.2f}', xy=(0, (geo_vals[0] + jac_vals[0])/2),
                       xytext=(0.5, 0.5), fontsize=12, color='purple',
                       arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7))

        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_geometric_vs_functional.png"), dpi=300)
        plt.close()

    # 2. Recall vs Precision Scatter (kept for reference)
    plt.figure(figsize=(10, 7))
    min_x = df_plot["Recall_Mean"].min()
    min_y = df_plot["Precision_Mean"].min()
    margin = 0.002

    sns.scatterplot(
        data=df_plot,
        x="Recall_Mean",
        y="Precision_Mean",
        hue="Method",
        style="Method",
        s=200,
        alpha=0.9,
        edgecolor="w"
    )

    for i in range(df_plot.shape[0]):
        row = df_plot.iloc[i]
        plt.text(
            row.Recall_Mean + 0.0002,
            row.Precision_Mean + 0.0002,
            row.Method,
            fontsize=10,
            weight='bold'
        )

    plt.title(f"Interpretability Trade-off: Recall vs Precision\n({model_name})", fontsize=14, weight='bold', pad=20)
    plt.xlabel("Recall (Base Features Found)", fontsize=12)
    plt.ylabel("Precision (Quantized Features Real)", fontsize=12)
    plt.xlim(min_x - margin, df_plot["Recall_Mean"].max() + margin)
    plt.ylim(min_y - margin, df_plot["Precision_Mean"].max() + margin)

    sns.despine()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_precision_recall.png"), dpi=300)
    plt.close()
    
    # 2. Spectral Decay
    plt.figure(figsize=(12, 7))
    
    # Find baseline row
    base_row = df_stats[df_stats["Method"].str.contains("(Base)")]
    if not base_row.empty and "SingularValues" in base_row.columns:
        base_vals = base_row.iloc[0]["SingularValues"]
        plt.plot(base_vals, 'k--', linewidth=2, label="Baseline", alpha=0.7)
    
    for _, row in df_plot.iterrows():
        if "SingularValues" in row and hasattr(row["SingularValues"], "__len__"):
            plt.plot(row["SingularValues"], linewidth=1.5, label=row["Method"], alpha=0.8)
    
    plt.yscale('log')
    plt.title(f"Singular Value Spectrum (Effective Rank Analysis)\n{model_name}", fontsize=14, weight='bold', pad=20)
    plt.ylabel("Singular Value (Log Scale)", fontsize=12)
    plt.xlabel("Index (Feature Component)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_spectral_decay.png"), dpi=300)
    plt.close()
    
    # 3. Rank Retention Bar Chart
    plt.figure(figsize=(10, 6))
    min_rank = df_plot["Rank_Retention"].min()
    max_rank = df_plot["Rank_Retention"].max()
    y_pad = (max_rank - min_rank) * 0.2 if max_rank != min_rank else 0.01
    
    ax = sns.barplot(
        data=df_plot, 
        x="Method", 
        y="Rank_Retention", 
        hue="Method",
        palette="magma",
        edgecolor=".2",
        legend=False
    )
    
    for container in ax.containers:
        ax.bar_label(container, fmt="{:.5f}", padding=3, fontsize=9)
    
    plt.title(f"Effective Rank Retention (Structure Preservation)\n{model_name}", fontsize=14, weight='bold', pad=20)
    plt.ylabel("Rank Retention Ratio (Target / Base)", fontsize=12)
    plt.xlabel("Quantization Method", fontsize=12)
    plt.ylim(min_rank - y_pad, max_rank + y_pad)
    plt.axhline(1.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label="Baseline (1.0)")
    
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_rank_retention.png"), dpi=300)
    plt.close()

    # 4. Functional Consistency (multiple metrics)
    # Support both old "Jaccard_Mean" and new "PreTopK_Jaccard" / "Activation_Corr" columns
    func_metrics = []
    for col_name, label in [
        ("PreTopK_Jaccard", "Pre-TopK Jaccard"),
        ("Jaccard_Mean", "Pre-TopK Jaccard"),  # Backwards compatibility
        ("Activation_Corr", "Activation Correlation"),
        ("Weighted_Jaccard", "Weighted Jaccard"),
    ]:
        if col_name in df_stats.columns and not df_plot[col_name].isna().all():
            func_metrics.append((col_name, label))

    # Remove duplicates (prefer new names)
    seen_labels = set()
    unique_metrics = []
    for col, label in func_metrics:
        if label not in seen_labels:
            unique_metrics.append((col, label))
            seen_labels.add(label)

    if unique_metrics:
        # Create grouped bar chart for all functional metrics
        plt.figure(figsize=(12, 6))

        # Reshape data for grouped bars
        plot_data = []
        for col, label in unique_metrics:
            for _, row in df_plot.iterrows():
                plot_data.append({
                    "Method": row["Method"],
                    "Metric": label,
                    "Value": row[col]
                })

        plot_df = pd.DataFrame(plot_data)

        ax = sns.barplot(
            data=plot_df,
            x="Method",
            y="Value",
            hue="Metric",
            palette="viridis",
            edgecolor=".2"
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="{:.3f}", padding=3, fontsize=8)

        plt.title(f"Functional Consistency Metrics\n({model_name})", fontsize=14, weight='bold', pad=20)
        plt.ylabel("Score", fontsize=12)
        plt.xlabel("Quantization Method", fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc='upper left')
        sns.despine()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_functional_consistency.png"), dpi=300)
        plt.close()

    # 5. Model Activation Similarity (for appendix) - load from CSV if exists
    model_act_files = glob.glob(os.path.join(output_dir, "model_activation_similarity_*.csv"))
    if model_act_files:
        plt.figure(figsize=(10, 6))

        all_data = []
        for f in model_act_files:
            method = os.path.basename(f).replace("model_activation_similarity_", "").replace(".csv", "")
            df_act = pd.read_csv(f)
            for _, row in df_act.iterrows():
                all_data.append({
                    "Method": method,
                    "Layer": f"Layer {int(row['Layer'])}",
                    "Correlation": row["Correlation"]
                })

        if all_data:
            df_act_all = pd.DataFrame(all_data)

            ax = sns.barplot(
                data=df_act_all,
                x="Layer",
                y="Correlation",
                hue="Method",
                palette="coolwarm",
                edgecolor=".2"
            )

            for container in ax.containers:
                ax.bar_label(container, fmt="{:.4f}", padding=3, fontsize=9)

            plt.title(f"Model Activation Similarity (BF16 vs Quantized)\n{model_name}", fontsize=14, weight='bold', pad=20)
            plt.ylabel("Pearson Correlation", fontsize=12)
            plt.xlabel("Layer", fontsize=12)
            plt.ylim(0.99, 1.001)  # Zoom in since correlations are very high
            plt.axhline(1.0, color='green', linestyle='--', alpha=0.7, label='Perfect (1.0)')
            plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc='upper left')
            sns.despine()
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_model_activation_similarity.png"), dpi=300)
            plt.close()

def analyze_model_group(model_name: str, methods_dict: Dict[str, str], output_dir: str):
    """Legacy wrapper for standalone execution."""
    print(f"\nAnalyzing Model: {model_name}")
    
    baselines = ["float32", "bfloat16", "float16"]
    baseline_method = None
    for b in baselines:
        if b in methods_dict:
            baseline_method = b
            break
            
    if not baseline_method:
        print(f"Skipping {model_name}: No baseline method found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, _ = compute_metrics(model_name, methods_dict, baseline_method, device)
    if not df.empty:
        generate_plots(model_name, df, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE quantization impact.")
    parser.add_argument("--checkpoints_dir", type=str, default="/experiment/SAE/checkpoints", help="Root directory of checkpoints")
    parser.add_argument("--output_dir", type=str, default="/experiment/SAE/analysis/results", help="Output directory")
    parser.add_argument("--model_name", type=str, default=None, help="Optional: Specific model name to analyze (e.g. 'google_gemma-3-1b-it')")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    
    print(f"Output Directory: {run_output_dir}")
    
    checkpoints_map = find_checkpoints(args.checkpoints_dir)
    if not checkpoints_map:
        print("No checkpoints found!")
        return

    # Filter by model name if specified
    if args.model_name:
        if args.model_name in checkpoints_map:
            print(f"Filtering for model: {args.model_name}")
            checkpoints_map = {args.model_name: checkpoints_map[args.model_name]}
        else:
            print(f"Error: Model '{args.model_name}' not found in checkpoints.")
            print(f"Available models: {list(checkpoints_map.keys())}")
            return
        
    for model_name, methods in checkpoints_map.items():
        analyze_model_group(model_name, methods, run_output_dir)
        
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
