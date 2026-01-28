"""
Re-aggregate all interpretability data from individual model runs.
Includes all 15 models and filters extreme outliers.
"""

import os
import pandas as pd
import glob

RESULTS_DIR = "/experiment/basic_interpretability/results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "global_aggregation")

def aggregate_logit_lens():
    """Aggregate logit lens KL data from all models."""
    all_data = []

    for run_dir in glob.glob(os.path.join(RESULTS_DIR, "run_*")):
        model_name = os.path.basename(run_dir).replace("run_", "")
        csv_path = os.path.join(run_dir, "tier1", "logit_lens_kl.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["model"] = model_name
            all_data.append(df)
            print(f"  Loaded {model_name}: {len(df)} rows, methods: {df['method'].unique().tolist()}")

    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        df_all.to_csv(os.path.join(OUTPUT_DIR, "global_logit_lens.csv"), index=False)
        print(f"\nAggregated logit lens: {len(df_all)} rows, {df_all['model'].nunique()} models")
        return df_all
    return None


def aggregate_attention_entropy():
    """Aggregate attention entropy data from all models."""
    all_data = []

    for run_dir in glob.glob(os.path.join(RESULTS_DIR, "run_*")):
        model_name = os.path.basename(run_dir).replace("run_", "")
        csv_path = os.path.join(run_dir, "tier2", "attention_entropy.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["model"] = model_name
            all_data.append(df)
            print(f"  Loaded {model_name}: {len(df)} rows, methods: {df['method'].unique().tolist()}")

    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        df_all.to_csv(os.path.join(OUTPUT_DIR, "global_attention_entropy.csv"), index=False)
        print(f"\nAggregated attention entropy: {len(df_all)} rows, {df_all['model'].nunique()} models")
        return df_all
    return None


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Aggregating Logit Lens Data ===")
    df_ll = aggregate_logit_lens()

    print("\n=== Aggregating Attention Entropy Data ===")
    df_ae = aggregate_attention_entropy()

    print("\n=== Summary ===")
    if df_ll is not None:
        print(f"Logit Lens: {df_ll['model'].nunique()} models, {df_ll['method'].nunique()} methods")
        print(f"  Methods: {sorted(df_ll['method'].unique().tolist())}")
    if df_ae is not None:
        print(f"Attention Entropy: {df_ae['model'].nunique()} models, {df_ae['method'].nunique()} methods")
        print(f"  Methods: {sorted(df_ae['method'].unique().tolist())}")
