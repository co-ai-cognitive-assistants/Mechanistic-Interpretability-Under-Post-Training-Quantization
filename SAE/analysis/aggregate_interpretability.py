import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import argparse
from scipy import stats
import numpy as np

def load_all_stats(results_dir):
    all_files = glob.glob(os.path.join(results_dir, "**/*_advanced_stats.csv"), recursive=True)
    df_list = []
    for f in all_files:
        df = pd.read_csv(f)
        df_list.append(df)
    
    if not df_list:
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)

def plot_global_comparison(df, metric, output_dir, title):
    plt.figure(figsize=(12, 6))
    
    # Order methods by performance (median)
    order = df.groupby("Method")[metric].median().sort_values(ascending=False).index
    
    # Box Plot with individual points
    sns.boxplot(data=df, x="Method", y=metric, order=order, hue="Method", palette="viridis", showfliers=False, legend=False)
    sns.stripplot(data=df, x="Method", y=metric, order=order, color="black", alpha=0.6, jitter=0.2)
    
    plt.title(f"Global Analysis: {title}", fontsize=14, weight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"global_{metric}.png"), dpi=300)
    plt.close()

def perform_statistical_tests(df, metric, baseline="float16"):
    """
    Performs pairwise t-tests against the baseline method.
    """
    print(f"\nStatistical Significance ({metric}):")
    print(f"{'Comparison':<30} | {'t-stat':<10} | {'p-value':<10} | {'Significant?'}")
    print("-" * 70)
    
    methods = df["Method"].unique()
    
    # Pivot to get matched samples (rows=Model, cols=Method)
    pivot_df = df.pivot(index="Model", columns="Method", values=metric)
    
    if baseline not in pivot_df.columns:
        # Try to find a valid baseline
        options = ["bfloat16", "float32"]
        for opt in options:
            if opt in pivot_df.columns:
                baseline = opt
                break
    
    if baseline not in pivot_df.columns:
        print(f"Baseline {baseline} not found in data.")
        return

    base_scores = pivot_df[baseline]
    
    for method in methods:
        if method == baseline: continue
        if method not in pivot_df.columns: continue
        
        target_scores = pivot_df[method]
        
        # Align data (drop NaNs if a model is missing one method)
        valid_mask = base_scores.notna() & target_scores.notna()
        a = base_scores[valid_mask]
        b = target_scores[valid_mask]
        
        if len(a) < 2:
            print(f"{baseline} vs {method}: Not enough samples.")
            continue
            
        t_stat, p_val = stats.ttest_rel(a, b)
        is_sig = "*" if p_val < 0.05 else ""
        if p_val < 0.001: is_sig = "***"
        
        print(f"{baseline} vs {method:<20} | {t_stat:6.3f}     | {p_val:.2e}   | {is_sig}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Path to analysis output folder (containing model subfolders or CSVs)")
    parser.add_argument("--output_dir", type=str, default="global_analysis_results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = load_all_stats(args.results_dir)
    if df.empty:
        print("No data found.")
        return
    
    print(f"Loaded {len(df)} runs across {df['Model'].nunique()} models.")
    
    # 1. Global Metrics Visualization
    metrics = {
        "Recall_GT_0.9": "Feature Recall (Strict > 0.9)",
        "Precision_GT_0.9": "Feature Precision (Strict > 0.9)",
        "Rank_Retention": "Effective Rank Retention",
        "Unique_Utilization": "Feature Space Utilization"
    }
    
    for metric, title in metrics.items():
        if metric in df.columns:
            plot_global_comparison(df, metric, args.output_dir, title)
            perform_statistical_tests(df, metric)
    
    # 2. Correlation with Model Size (Hypothesis: Bigger models resist quantization better)
    # Extract size from name (rough heuristic)
    def extract_size(name):
        name_u = name.upper()
        if "70B" in name_u: return 70
        if "32B" in name_u: return 32
        if "14B" in name_u: return 14
        if "8B" in name_u: return 8
        if "7B" in name_u: return 7
        if "1.5B" in name_u: return 1.5
        if "1B" in name_u: return 1
        return np.nan

    df["Size_B"] = df["Model"].apply(extract_size)
    
    # Plot Trend: Size vs Recall for Int4 (example)
    target_method = "int4" # adjust as needed
    subset = df[df["Method"].str.contains(target_method, case=False, na=False)]
    
    if not subset.empty and subset["Size_B"].notna().sum() > 2:
        plt.figure(figsize=(8, 6))
        sns.regplot(data=subset, x="Size_B", y="Recall_GT_0.9", logx=True)
        plt.title(f"Does Model Scale Protect Interpretability? ({target_method})")
        plt.xlabel("Model Size (Billions)")
        plt.ylabel("Feature Recall")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "size_scaling_trend.png"))
        print("\nGenerated size scaling trend analysis.")

if __name__ == "__main__":
    main()
