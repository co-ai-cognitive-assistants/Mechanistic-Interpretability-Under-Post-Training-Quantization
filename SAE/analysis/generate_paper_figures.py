#!/usr/bin/env python3
"""
Generate publication-quality figures for the SAE analysis section of the paper.

This script creates figures that illustrate:
1. Geometric vs Functional similarity gap (main finding)
2. Model activation similarity (shows quantization doesn't change activations much)
3. SAE metrics comparison across quantization methods

Usage:
    python generate_paper_figures.py --results_dir <path_to_results> --output_dir <figures_dir>
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (8, 5)
sns.set_theme(style="whitegrid", context="paper")


def load_results(results_dir):
    """Load all analysis results from the results directory."""
    # Find the most recent run
    run_dirs = sorted(glob.glob(os.path.join(results_dir, "20*")))
    if not run_dirs:
        print(f"No results found in {results_dir}")
        return None, None, None

    latest_run = run_dirs[-1]
    print(f"Loading results from: {latest_run}")

    # Find model subdirectories
    model_dirs = [d for d in glob.glob(os.path.join(latest_run, "*"))
                  if os.path.isdir(d) and not d.endswith(".txt")]

    all_stats = []
    all_details = []
    all_model_act = []

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)

        # Load advanced stats
        stats_file = glob.glob(os.path.join(model_dir, "*_advanced_stats.csv"))
        if stats_file:
            df = pd.read_csv(stats_file[0])
            df["Model"] = model_name
            all_stats.append(df)

        # Load per-feature details
        detail_files = glob.glob(os.path.join(model_dir, "details_*.csv"))
        for f in detail_files:
            method = os.path.basename(f).replace("details_", "").replace(".csv", "")
            df = pd.read_csv(f)
            df["Model"] = model_name
            df["Method"] = method
            all_details.append(df)

        # Load model activation similarity
        act_files = glob.glob(os.path.join(model_dir, "model_activation_similarity_*.csv"))
        for f in act_files:
            method = os.path.basename(f).replace("model_activation_similarity_", "").replace(".csv", "")
            df = pd.read_csv(f)
            df["Model"] = model_name
            df["Method"] = method
            all_model_act.append(df)

    df_stats = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    df_details = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    df_model_act = pd.concat(all_model_act, ignore_index=True) if all_model_act else pd.DataFrame()

    return df_stats, df_details, df_model_act


def fig_geometric_vs_functional(df_stats, df_details, output_dir, model_filter=None):
    """
    Figure: Geometric vs Functional Similarity Gap

    This is the main figure showing that high geometric similarity (decoder cosine)
    does not translate to high functional similarity (activation Jaccard).
    """
    if df_stats.empty:
        print("No stats data for geometric vs functional figure")
        return

    # Filter to non-baseline methods
    df = df_stats[~df_stats["Method"].str.contains("Base", na=False)].copy()

    if model_filter:
        df = df[df["Model"].str.contains(model_filter, case=False)]

    if df.empty:
        print("No data after filtering")
        return

    # Check if we have the needed columns
    # Recall_Mean IS the mean geometric similarity (mean of decoder cosine similarities)
    geo_col = "Recall_Mean"
    jac_col = "Jaccard_Mean"

    if jac_col not in df.columns:
        print(f"Missing {jac_col} column")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort methods for consistent ordering
    method_order = ['fp8', 'int4', 'awq', 'gptq', 'hqq']
    methods = [m for m in method_order if m in df["Method"].values]
    if not methods:
        methods = df["Method"].unique().tolist()

    x = np.arange(len(methods))
    width = 0.35

    # Get values for each method
    geo_vals = [df[df["Method"] == m][geo_col].values[0] if len(df[df["Method"] == m]) > 0 else 0 for m in methods]
    jac_vals = [df[df["Method"] == m][jac_col].values[0] if len(df[df["Method"] == m]) > 0 else 0 for m in methods]

    # Create bars
    bars1 = ax.bar(x - width/2, geo_vals, width, label='Geometric Similarity\n(Decoder Cosine)',
                   color='#27ae60', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, jac_vals, width, label='Functional Jaccard\n(Activation Overlap)',
                   color='#c0392b', edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, val in zip(bars1, geo_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, jac_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.set_xlabel('Quantization Method', fontsize=12)
    ax.set_title('Geometric vs Functional Similarity Gap\n(DeepSeek-R1-1.5B)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods], fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.0)

    # Add gap annotation for the first method
    if len(geo_vals) > 0 and len(jac_vals) > 0:
        # Find FP8 index for annotation
        fp8_idx = methods.index('fp8') if 'fp8' in methods else 0
        g, j = geo_vals[fp8_idx], jac_vals[fp8_idx]
        gap = g - j
        mid = (g + j) / 2

        # Draw gap indicator
        ax.annotate('', xy=(fp8_idx + width/2, j + 0.01), xytext=(fp8_idx - width/2, g - 0.01),
                   arrowprops=dict(arrowstyle='<->', color='purple', lw=2.5))
        ax.text(fp8_idx, mid + 0.05, f'Gap: {gap:.2f}', ha='center', va='center',
               fontsize=12, color='purple', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='purple', alpha=0.9))

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sae_geometric_vs_functional.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "sae_geometric_vs_functional.pdf"), bbox_inches='tight')
    plt.close()
    print(f"Saved: sae_geometric_vs_functional.png/pdf")


def fig_model_activation_similarity(df_model_act, output_dir, model_filter=None):
    """
    Figure: Model Activation Similarity (Appendix)

    Shows that quantized models produce near-identical activations to baseline,
    proving the low Jaccard is due to SAE training, not model differences.
    """
    if df_model_act.empty:
        print("No model activation data")
        return

    df = df_model_act.copy()
    if model_filter:
        df = df[df["Model"].str.contains(model_filter, case=False)]

    if df.empty:
        print("No data after filtering")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Create layer labels
    df["Layer_Label"] = df["Layer"].apply(lambda x: f"Layer {int(x)}")

    # Plot grouped bars
    methods = df["Method"].unique()
    layers = sorted(df["Layer"].unique())
    x = np.arange(len(layers))
    width = 0.8 / len(methods)

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        method_data = df[df["Method"] == method]
        correlations = [method_data[method_data["Layer"] == l]["Correlation"].values[0]
                       if len(method_data[method_data["Layer"] == l]) > 0 else 0
                       for l in layers]
        bars = ax.bar(x + i * width - width * len(methods) / 2, correlations, width,
                     label=method.upper(), color=colors[i], edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, correlations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9, rotation=45)

    ax.set_ylabel('Correlation with Baseline')
    ax.set_xlabel('Layer')
    ax.set_title('Model Activation Similarity (BF16 vs Quantized)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {int(l)}" for l in layers])
    ax.legend(title="Method")

    # Zoom in on high correlation region
    ax.set_ylim(0.99, 1.002)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(len(layers) - 0.5, 1.0005, 'Perfect = 1.0', ha='right', va='bottom',
           fontsize=10, color='green')

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_activation_similarity.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "model_activation_similarity.pdf"), bbox_inches='tight')
    plt.close()
    print(f"Saved: model_activation_similarity.png/pdf")


def fig_jaccard_distribution(df_details, output_dir, model_filter=None):
    """
    Figure: Per-feature Jaccard distribution

    Shows the distribution of Jaccard scores across features, illustrating
    that most features have very low functional overlap.
    """
    if df_details.empty:
        print("No detail data for Jaccard distribution")
        return

    df = df_details.copy()
    if model_filter:
        df = df[df["Model"].str.contains(model_filter, case=False)]

    if df.empty or "FunctionalJaccard" not in df.columns:
        print("No Jaccard data after filtering")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    methods = df["Method"].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        method_data = df[df["Method"] == method]["FunctionalJaccard"].values
        ax.hist(method_data, bins=50, alpha=0.6, label=method.upper(),
               color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Per-Feature Jaccard Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Functional Jaccard Scores')
    ax.legend()

    # Add vertical lines for means
    for i, method in enumerate(methods):
        method_data = df[df["Method"] == method]["FunctionalJaccard"].values
        mean_val = np.mean(method_data)
        ax.axvline(mean_val, color=colors[i], linestyle='--', linewidth=2)
        ax.text(mean_val + 0.01, ax.get_ylim()[1] * 0.9 - i * ax.get_ylim()[1] * 0.08,
               f'{method}: μ={mean_val:.3f}', fontsize=10, color=colors[i])

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sae_jaccard_distribution.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "sae_jaccard_distribution.pdf"), bbox_inches='tight')
    plt.close()
    print(f"Saved: sae_jaccard_distribution.png/pdf")


def fig_rank_retention(df_stats, output_dir, model_filter=None):
    """
    Figure: Rank Retention

    Shows that effective rank is preserved despite different features,
    indicating SAEs learn equally complex representations.
    """
    if df_stats.empty or "Rank_Retention" not in df_stats.columns:
        print("No rank retention data")
        return

    df = df_stats[~df_stats["Method"].str.contains("Base", na=False)].copy()
    if model_filter:
        df = df[df["Model"].str.contains(model_filter, case=False)]

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    methods = df["Method"].unique()
    values = [df[df["Method"] == m]["Rank_Retention"].values[0] for m in methods]

    colors = ['#3498db' if v >= 0.99 else '#e74c3c' for v in values]
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
               f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Rank Retention (Target / Baseline)')
    ax.set_xlabel('Quantization Method')
    ax.set_title('Effective Rank Retention')

    # Set y-axis limits
    min_val = min(values) - 0.02
    max_val = max(values) + 0.02
    ax.set_ylim(min_val, max_val)

    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='Perfect (1.0)')
    ax.legend()

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sae_rank_retention.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "sae_rank_retention.pdf"), bbox_inches='tight')
    plt.close()
    print(f"Saved: sae_rank_retention.png/pdf")


def fig_geometric_jaccard_scatter(df_details, output_dir, model_filter=None):
    """
    Figure: Per-feature Geometric vs Functional scatter plot

    Shows that even features with high geometric similarity can have low Jaccard,
    illustrating the geometric-functional disconnect at the feature level.
    """
    if df_details.empty:
        print("No detail data for scatter plot")
        return

    df = df_details.copy()
    if model_filter:
        df = df[df["Model"].str.contains(model_filter, case=False)]

    if df.empty or "GeometricSim" not in df.columns or "FunctionalJaccard" not in df.columns:
        print("Missing required columns for scatter plot")
        return

    # Focus on FP8 for cleaner visualization
    df_fp8 = df[df["Method"] == "fp8"].copy() if "fp8" in df["Method"].values else df

    if df_fp8.empty:
        df_fp8 = df[df["Method"] == df["Method"].iloc[0]]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Sample if too many points
    if len(df_fp8) > 5000:
        df_plot = df_fp8.sample(5000, random_state=42)
    else:
        df_plot = df_fp8

    # Create scatter with density coloring
    scatter = ax.scatter(df_plot["GeometricSim"], df_plot["FunctionalJaccard"],
                        alpha=0.3, s=10, c='#3498db', edgecolors='none')

    # Add diagonal reference line (what you'd expect if they were correlated)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect correlation', alpha=0.7)

    # Add mean lines
    mean_geo = df_plot["GeometricSim"].mean()
    mean_jac = df_plot["FunctionalJaccard"].mean()
    ax.axvline(mean_geo, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax.axhline(mean_jac, color='orange', linestyle=':', linewidth=2, alpha=0.7)

    # Add annotation for the means
    ax.annotate(f'Mean Geometric: {mean_geo:.2f}', xy=(mean_geo, 0.9),
               fontsize=10, color='green', ha='left')
    ax.annotate(f'Mean Jaccard: {mean_jac:.3f}', xy=(0.1, mean_jac + 0.02),
               fontsize=10, color='orange')

    ax.set_xlabel('Geometric Similarity (Decoder Cosine)', fontsize=12)
    ax.set_ylabel('Functional Jaccard (Activation Overlap)', fontsize=12)
    ax.set_title('Per-Feature: Geometric vs Functional Similarity\n(FP8, DeepSeek-R1-1.5B)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left')

    # Add text annotation explaining the gap
    ax.text(0.7, 0.15, f'Even with geometric\nsim > 0.9, Jaccard\nis often < 0.5',
           fontsize=10, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sae_geometric_jaccard_scatter.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "sae_geometric_jaccard_scatter.pdf"), bbox_inches='tight')
    plt.close()
    print(f"Saved: sae_geometric_jaccard_scatter.png/pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures for SAE analysis")
    parser.add_argument("--results_dir", type=str, default="/experiment/SAE/analysis/results",
                       help="Directory containing analysis results")
    parser.add_argument("--output_dir", type=str, default="/experiment/691c50a6c3aab84fde1f53bb/figures",
                       help="Directory to save figures")
    parser.add_argument("--model_filter", type=str, default=None,
                       help="Filter to specific model (e.g., 'deepseek', 'gemma')")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    df_stats, df_details, df_model_act = load_results(args.results_dir)

    print("\nGenerating figures...")

    # Main paper figures
    fig_geometric_vs_functional(df_stats, df_details, args.output_dir, args.model_filter)
    fig_geometric_jaccard_scatter(df_details, args.output_dir, args.model_filter)
    fig_rank_retention(df_stats, args.output_dir, args.model_filter)

    # Appendix figures
    fig_model_activation_similarity(df_model_act, args.output_dir, args.model_filter)
    fig_jaccard_distribution(df_details, args.output_dir, args.model_filter)

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
