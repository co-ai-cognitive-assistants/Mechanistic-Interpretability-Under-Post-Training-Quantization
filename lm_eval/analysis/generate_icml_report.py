import os
import json
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Configuration
RESULTS_DIR = "../../results"  # Relative to where the script is run (assuming inside lm_eval/analysis/)
OUTPUT_DIR = "figures"      # Subdirectory for plots
BENCHMARKS = ['arc_challenge', 'hellaswag', 'mmlu', 'winogrande']
QUANT_TYPES = ['bfloat16', 'float16', 'fp8', 'int8', 'int4', 'gguf', 'awq', 'gptq', 'hqq']
BASELINE_TYPES = ['bfloat16', 'float16']

# Set style for publication (ICML style-ish)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_filename(filename):
    basename = os.path.basename(filename)
    # Regex to capture model name, quantization, and timestamp
    # Pattern expects: {ModelName}_{Quantization}_temp_detail_{Timestamp}.json
    quants_pattern = "|".join(QUANT_TYPES + ['bf16']) 
    # More robust regex: look for the quantization keyword surrounded by underscores
    # and "temp_detail" following it.
    pattern = re.compile(rf"(.+)_({quants_pattern})_temp_detail_(.+)\.json")
    
    match = pattern.match(basename)
    if match:
        model_name = match.group(1)
        quant = match.group(2)
        timestamp_str = match.group(3)
        
        # Normalize quant names
        if quant == 'bf16': quant = 'bfloat16'
        
        # Clean model name prefixes to make charts cleaner
        for prefix in ['google_', 'meta-llama_', 'Qwen_', 'deepseek-ai_', 'openai_']:
            if model_name.startswith(prefix):
                model_name = model_name[len(prefix):]
        
        return model_name, quant, timestamp_str
    
    return None, None, None

def extract_model_size(model_name):
    # Look for patterns like 7B, 1.5B, 270m
    match = re.search(r'(\d+(?:\.\d+)?)([bBm])', model_name, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        unit = match.group(2).lower()
        if unit == 'm':
            return val / 1000.0 # Convert Millions to Billions
        return val
    return np.nan

def load_results(results_dir):
    data = []
    # Handle running from analysis/ or root
    search_path = results_dir
    if not os.path.exists(search_path):
        # Fallback if running from root
        if os.path.exists("results"):
            search_path = "results"
            
    print(f"Loading results from: {search_path}")
    files = glob.glob(os.path.join(search_path, "*.json"))
    
    print(f"Found {len(files)} JSON files.")
    
    for f in files:
        model, quant, timestamp = parse_filename(f)
        if not model:
            # Try a fallback parse if the strictly formatted one fails
            # Sometimes filenames might differ slightly
            continue
            
        try:
            with open(f, 'r') as fp:
                content = json.load(fp)
            
            res = content.get('results', {})
            if not res:
                continue

            row = {
                'model': model,
                'quantization': quant,
                'timestamp': timestamp,
                'file': f,
                'size_b': extract_model_size(model)
            }
            
            # Extract metrics
            has_metrics = False
            for bench in BENCHMARKS:
                if bench in res:
                    # Usually 'acc,none' or just 'acc'
                    val = res[bench].get('acc,none')
                    if val is None:
                        val = res[bench].get('acc')
                    
                    if val is not None:
                        row[bench] = val
                        has_metrics = True
                        
                        # Get stderr
                        stderr = res[bench].get('acc_stderr,none')
                        if stderr is None: stderr = res[bench].get('acc_stderr')
                        if stderr is not None:
                            row[f"{bench}_stderr"] = stderr
            
            if has_metrics:
                # Calculate simple average
                valid_scores = [row[b] for b in BENCHMARKS if b in row]
                row['average'] = np.mean(valid_scores) if valid_scores else 0
                data.append(row)
                
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No valid data found!")
        return df

    # Filter duplicates: keep latest timestamp for each (model, quantization)
    initial_count = len(df)
    df = df.sort_values('timestamp', ascending=False)
    df = df.drop_duplicates(subset=['model', 'quantization'], keep='first')
    
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"Duplicate handling: Dropped {dropped_count} older evaluation files. Keeping only the latest for each (model, method).")
    else:
        print("No duplicate evaluations found.")
    
    return df

def calculate_degradation(df):
    # We want to compare against a baseline (bfloat16 or float16)
    baselines = {}
    
    # First pass: find best baseline for each model
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        base_row = None
        # Prefer bfloat16, then float16
        for base_type in BASELINE_TYPES:
            found = model_df[model_df['quantization'] == base_type]
            if not found.empty:
                base_row = found.iloc[0]
                break
        
        if base_row is not None:
            baselines[model] = base_row['average']
            
    # Calculate deltas
    def get_delta(row):
        base_score = baselines.get(row['model'])
        if base_score:
            return (row['average'] - base_score) / base_score * 100.0
        return np.nan
    
    def get_normalized(row):
        base_score = baselines.get(row['model'])
        if base_score:
            return row['average'] / base_score
        return np.nan

    df['degradation_pct'] = df.apply(get_delta, axis=1)
    df['normalized_acc'] = df.apply(get_normalized, axis=1)
    
    return df

# --- Plotting Functions ---

def plot_model_comparison(df, model, output_path):
    """Bar chart comparing quantization methods for a single model."""
    subset = df[df['model'] == model].sort_values('average', ascending=False)
    if subset.empty: return
    
    plt.figure(figsize=(10, 6))
    
    # Create colors: highlight baseline
    palette = {}
    quants = subset['quantization'].unique()
    for q in quants:
        if q in BASELINE_TYPES:
            palette[q] = "#2ecc71" # Green for baseline
        else:
            palette[q] = "#3498db" # Blue for others
            
    # Highlight specific low-bit quants
    if 'int4' in palette: palette['int4'] = "#e74c3c" # Red for int4
    
    ax = sns.barplot(data=subset, x='quantization', y='average', hue='quantization', palette=palette, dodge=False)
    
    # Add error bars if stderr columns exist (approximate as mean of stderrs)
    # This is a bit tricky with seaborn barplot aggregation, but since we have 1 row per quant, we can do it manually
    for i, p in enumerate(ax.patches):
        if i < len(subset):
            row = subset.iloc[i]
            # Calculate composite stderr roughly (sqrt(sum(stderr^2))/N)
            stderrs = [row[f"{b}_stderr"] for b in BENCHMARKS if f"{b}_stderr" in row and pd.notna(row[f"{b}_stderr"])]
            if stderrs:
                composite_err = np.sqrt(np.sum(np.array(stderrs)**2)) / len(stderrs)
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                ax.errorbar(x, y, yerr=composite_err, fmt='none', c='black', capsize=5)

    plt.title(f"Average Accuracy by Quantization: {model}")
    plt.ylim(0, 1.05)
    plt.ylabel("Average Accuracy")
    plt.xlabel("Quantization Method")
    plt.xticks(rotation=45)
    plt.legend([],[], frameon=False) # Hide legend as x-axis is sufficient
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_radar_chart(df, model, output_path):
    """Radar chart for benchmark breakdown."""
    subset = df[df['model'] == model]
    if subset.empty: return

    # Benchmarks available for this model
    available_bench = [b for b in BENCHMARKS if b in subset.columns and subset[b].notna().any()]
    if len(available_bench) < 3: return

    categories = available_bench
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], [b.replace('_', ' ').title() for b in categories], size=10)
    
    # Scale max
    max_val = subset[categories].max().max()
    ax.set_ylim(0, max_val * 1.1)
    
    # Sort: Baseline first, then high acc to low acc to ensure visibility
    subset = subset.sort_values('average', ascending=False)
    
    # Define distinctive styles
    styles = {
        'bfloat16': {'color': 'black', 'ls': '-', 'lw': 2.5},
        'float16': {'color': 'black', 'ls': '--', 'lw': 2.5},
        'fp8': {'color': '#9b59b6', 'ls': '-', 'lw': 2},
        'int8': {'color': '#3498db', 'ls': '-', 'lw': 2},
        'int4': {'color': '#e74c3c', 'ls': '-', 'lw': 2},
        'gguf': {'color': '#f1c40f', 'ls': '-.', 'lw': 2},
        'awq': {'color': '#2ecc71', 'ls': ':', 'lw': 2},
        'gptq': {'color': '#16a085', 'ls': ':', 'lw': 2},
    }

    for idx, row in subset.iterrows():
        quant = row['quantization']
        values = row[categories].tolist()
        values += values[:1]
        
        style = styles.get(quant, {'color': 'gray', 'ls': '-', 'lw': 1})
        
        ax.plot(angles, values, linewidth=style['lw'], linestyle=style['ls'], color=style['color'], label=quant)
        # Only fill for baseline to reduce clutter
        if quant in BASELINE_TYPES:
            ax.fill(angles, values, color=style['color'], alpha=0.05)
        
    plt.title(f"Benchmark Profile: {model}", y=1.08, size=16)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_aggregated_impact(df, output_path):
    """Boxplot of degradation percentage."""
    # Exclude baselines from the boxplot as they are 0 by definition
    subset = df[~df['quantization'].isin(BASELINE_TYPES)].copy()
    
    if subset.empty: return

    # Sort quant types by median degradation
    order = subset.groupby('quantization')['degradation_pct'].median().sort_values(ascending=False).index
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=subset, x='quantization', y='degradation_pct', order=order, palette="coolwarm", hue='quantization', dodge=False)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.title("Performance Degradation Distribution by Quantization Method\n(Across All Models)")
    plt.ylabel("Accuracy Change relative to Baseline (%)")
    plt.xlabel("Quantization Method")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_size_vs_degradation(df, output_path):
    """Scatter plot of size vs degradation."""
    subset = df[~df['quantization'].isin(BASELINE_TYPES)].copy()
    if subset.empty: return
    
    plt.figure(figsize=(12, 8))
    
    # Use distinct markers for different quant types
    markers = {
        'fp8': 'o', 'int8': 's', 'int4': '^', 
        'gguf': 'D', 'awq': 'P', 'gptq': 'X', 'hqq': '*'
    }
    
    sns.scatterplot(
        data=subset, 
        x='size_b', 
        y='degradation_pct', 
        hue='quantization', 
        style='quantization',
        markers=markers,
        s=150, 
        alpha=0.8,
        palette='deep'
    )
    
    plt.xscale('log')
    plt.axhline(0, color='gray', linestyle='--')
    
    # Annotate points that are very low
    for i, row in subset.iterrows():
        if row['degradation_pct'] < -20: # Arbitrary threshold for "bad" 
            plt.text(row['size_b'], row['degradation_pct']-2, row['model'], fontsize=8, alpha=0.7)

    plt.title("Quantization Sensitivity vs Model Size")
    plt.xlabel("Model Size (Billions of Parameters) - Log Scale")
    plt.ylabel("Accuracy Change (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Method")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    print("Starting Comprehensive Analysis for ICML Report...")
    
    # Setup Output
    # Handle running location
    out_dir_path = OUTPUT_DIR
    if not os.path.exists("analysis") and os.path.basename(os.getcwd()) == "analysis":
        # We are inside analysis
        out_dir_path = OUTPUT_DIR
    elif os.path.exists("analysis"):
        # We are in root
        out_dir_path = os.path.join("analysis", OUTPUT_DIR)
        
    ensure_dir(out_dir_path)
    print(f"Output directory: {out_dir_path}")
    
    # Load
    df = load_results(RESULTS_DIR)
    if df.empty: return
    print(f"Loaded {len(df)} unique result records.")
    
    # Process
    df = calculate_degradation(df)
    
    # Individual Plots
    print("Generating individual model plots...")
    models = df['model'].unique()
    for model in models:
        safe_name = model.replace("/", "_")
        
        # Bar Comparison
        plot_model_comparison(df, model, os.path.join(out_dir_path, f"{safe_name}_bar.png"))
        
        # Radar Chart
        plot_radar_chart(df, model, os.path.join(out_dir_path, f"{safe_name}_radar.png"))

    # Aggregated Plots
    print("Generating aggregated plots...")
    plot_aggregated_impact(df, os.path.join(out_dir_path, "aggregated_degradation_boxplot.png"))
    plot_size_vs_degradation(df, os.path.join(out_dir_path, "size_vs_degradation_scatter.png"))
    
    # Summary Tables
    print("Saving summary tables...")
    summary = df[['model', 'quantization', 'average', 'degradation_pct', 'size_b', 'timestamp']].sort_values(['model', 'average'], ascending=[True, False])
    
    csv_path = os.path.join(os.path.dirname(out_dir_path), "quantization_summary.csv")
    summary.to_csv(csv_path, index=False)
    
    # LaTeX Table
    tex_path = os.path.join(os.path.dirname(out_dir_path), "summary_table.tex")
    
    # Pivot for cleaner LaTeX table: Model as rows, Quant as columns (showing Degradation or Avg)
    # Let's show Average Score
    pivot_df = summary.pivot(index='model', columns='quantization', values='average')
    # Add Size column back
    sizes = summary[['model', 'size_b']].drop_duplicates().set_index('model')
    pivot_df = pivot_df.join(sizes)
    # Reorder columns
    cols = ['size_b'] + [c for c in pivot_df.columns if c != 'size_b']
    pivot_df = pivot_df[cols]
    
    latex_table = pivot_df.to_latex(float_format="%.2f", na_rep="-", caption="Average Accuracy by Model and Quantization Method", label="tab:main_results")
    with open(tex_path, "w") as f:
        f.write(latex_table)

    print("Analysis Complete.")
    print(f"Figures saved to: {out_dir_path}")
    print(f"Data saved to: {csv_path}")

if __name__ == "__main__":
    main()
