import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from scipy.stats import norm, mannwhitneyu
from itertools import combinations
from statsmodels.sandbox.stats.multicomp import multipletests

DETECTION_ORIGINAL_FILE = os.environ.get('DETECTION_ORIGINAL_FILE', 'self_sync_outputs_Mistral7B/detection_results.json')
DETECTION_ATTACK_TRANSLATION_FILE = os.environ.get('DETECTION_ATTACK_TRANSLATION_FILE', 'self_sync_outputs_Mistral7B/detection_results_attack_translation.json')
DETECTION_ATTACK_T5_FILE = os.environ.get('DETECTION_ATTACK_T5_FILE', 'self_sync_outputs_Mistral7B/detection_results_attack_t5.json')

PLOT_TRADEOFF_SCATTER_PATH = os.environ.get('PLOT_TRADEOFF_SCATTER_PATH', 'self_sync_outputs_Mistral7B/zscore_vs_perplexity_scatter.png')
PLOT_ROBUSTNESS_BAR_PATH = os.environ.get('PLOT_ROBUSTNESS_BAR_PATH', 'self_sync_outputs_Mistral7B/robustness_zscore_barplot.png')
PLOT_STEP_PLOT_PATH = os.environ.get('PLOT_STEP_PLOT_PATH', 'self_sync_outputs_Mistral7B/zscore_vs_tokens_stepplot.png')
PLOT_U_HEATMAP_PATH = os.environ.get('PLOT_U_HEATMAP_PATH', 'self_sync_outputs_Mistral7B/mannwhitney_heatmap.png')
U_TEST_CSV_PATH = os.environ.get('U_TEST_CSV_PATH', 'self_sync_outputs_Mistral7B/mannwhitney_results.csv')

ZSCORE_INTERVALS = 20

config_name_mapping = {
    "baseline": "Human Text",
    "llm_baseline": "LLM Baseline",
    
    # Weak
    "llm_senso_only_weak": "Senso-Sync (Weak)",
    "llm_redgreen_only_weak": "RedGreen (Weak)",
    "llm_full_adaptive_weak": "Full Adaptive (Weak)",
    
    # Medium
    "llm_senso_only_medium": "Senso-Sync (Medium)",
    "llm_redgreen_only_medium": "RedGreen (Medium)",
    "llm_full_adaptive_medium": "Full Adaptive (Medium)",
    
    # Strong
    "llm_senso_only_strong": "Senso-Sync (Strong)",
    "llm_redgreen_only_strong": "RedGreen (Strong)",
    "llm_full_adaptive_strong": "Full Adaptive (Strong)",
}

config_order = [
    "baseline", "llm_baseline",
    "llm_senso_only_weak", "llm_redgreen_only_weak", "llm_full_adaptive_weak",
    "llm_senso_only_medium", "llm_redgreen_only_medium", "llm_full_adaptive_medium",
    "llm_senso_only_strong", "llm_redgreen_only_strong", "llm_full_adaptive_strong"
]

def load_zscore_data(filepath):
    """
    Loads detection data and returns a dict of 
    {config: [{'zscore_at_token': [list], 'final_zscore': float, 'perplexity': float}, ...]}
    """
    try:
        with open(filepath, 'r') as f: 
            data = json.load(f)
    except FileNotFoundError:
        print(f"  WARNING: Could not find file: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"  WARNING: Could not decode JSON from: {filepath}")
        return None

    results = {}
    for entry in data:
        detection = entry.get('detection', {})
        for config, scores in detection.items():
            if 'error' in scores:
                continue
            if config not in results:
                results[config] = []
            try:
                z_list = scores.get('zscore_at_token')
                pplx = float(scores.get('perplexity'))
                if z_list and len(z_list) > 0 and not np.isnan(pplx):
                    final_z = float(z_list[-1])
                    results[config].append({
                        'zscore_at_token': z_list,
                        'final_zscore': final_z,
                        'perplexity': pplx
                    })
            except:
                continue
    return results


print("  Loading detection data...")
original_data = load_zscore_data(DETECTION_ORIGINAL_FILE)
trans_data = load_zscore_data(DETECTION_ATTACK_TRANSLATION_FILE)
t5_data = load_zscore_data(DETECTION_ATTACK_T5_FILE)

if not original_data:
    print("  ERROR: Cannot proceed without original detection data.")
    sys.exit(1)

plot_data = []
step_plot_data = []

for config_key in config_order:
    if config_key not in config_name_mapping:
        continue
    config_label = config_name_mapping[config_key]
    
    # Iterate through all three conditions
    for condition, data in [('Original', original_data), ('Translation', trans_data), ('T5 Attack', t5_data)]:
        if data and config_key in data:
            final_zscores = [d['final_zscore'] for d in data[config_key]]
            pplxs = [d['perplexity'] for d in data[config_key]]
            
            if not final_zscores or not pplxs:
                continue
            
            # Aggregate data for Scatter/Bar plots
            plot_data.append({
                'Configuration': config_label,
                'Condition': condition,
                'Median Z-Score': np.median(final_zscores),
                'Median Perplexity': np.median(pplxs)
            })
            
            # Detailed token data for Step plots
            for d in data[config_key]:
                for i, z_val in enumerate(d['zscore_at_token']):
                    num_tokens = (i + 1) * ZSCORE_INTERVALS
                    step_plot_data.append({
                        'Configuration': config_label,
                        'Condition': condition,
                        'Num Opportunities': num_tokens,
                        'Z-Score': z_val
                    })

df_plot = pd.DataFrame(plot_data)
df_step_plot = pd.DataFrame(step_plot_data)


# Z-Score vs. Perplexity Trade-off Scatter Plot
print("  Generating Z-Score vs. Perplexity trade-off scatter plot...")

df_scatter = df_plot[df_plot["Condition"] == "Original"]
if df_scatter.empty:
    print("  Skipping trade-off plot: No 'Original' data.")
else:
    sns.set_theme(style="whitegrid", context="talk")
    custom_palette = {
        "Human Text": "#7f7f7f",
        "LLM Baseline": "#1f77b4",
        "Senso-Sync (Weak)": "#2ca02c",
        "RedGreen (Weak)": "#98df8a",
        "Full Adaptive (Weak)": "#006400",
        "Senso-Sync (Medium)": "#ff7f0e",
        "RedGreen (Medium)": "#ffbb78",
        "Full Adaptive (Medium)": "#cc6600",
        "Senso-Sync (Strong)": "#d62728",
        "RedGreen (Strong)": "#ff9896",
        "Full Adaptive (Strong)": "#8b0000"
    }

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.scatterplot(
        data=df_scatter, x="Median Perplexity", y="Median Z-Score",
        hue="Configuration", palette=custom_palette, s=250,
        edgecolor="black", linewidth=0.8, ax=ax
    )
    
    ax.axhline(y=norm.ppf(0.05), color="red", linestyle="--", linewidth=1.5, label="Detection Threshold (p=0.05)")
    ax.set_title("Watermark Trade-off: Detection Strength vs. Perplexity (Original Text)", fontsize=18, pad=18)
    ax.set_xlabel("Median Perplexity (Lower → better text quality)", fontsize=14, labelpad=10)
    ax.set_ylabel("Median Z-Score (Lower → stronger detection)", fontsize=14, labelpad=10)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Configuration", bbox_to_anchor=(1.02, 1),
              loc="upper left", borderaxespad=0.5, frameon=True, fontsize=11, title_fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(PLOT_TRADEOFF_SCATTER_PATH, dpi=300, bbox_inches="tight")
    print(f"  Scatter plot saved to '{PLOT_TRADEOFF_SCATTER_PATH}'")
    plt.close()

# Robustness Bar Plot
print("  Generating robustness bar plot...")

df_bar = df_plot[df_plot["Configuration"] != "Human Text"].copy()
if df_bar.empty:
    print("  Skipping robustness plot: No watermark data.")
else:
    sns.set_theme(style="whitegrid", context="talk")
    order = [config_name_mapping[c] for c in config_order if c in config_name_mapping and c != "baseline"]
    conditions = ["Original", "Translation", "T5 Attack"]

    # Jitter for visualization
    df_bar["Median Z-Score"] = df_bar["Median Z-Score"].astype(float)
    df_bar["Median Z-Score"] += (
        df_bar.groupby("Configuration")["Median Z-Score"]
              .transform(lambda x: np.linspace(-0.01, 0.01, len(x)))
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(
        data=df_bar, x="Configuration", y="Median Z-Score", hue="Condition",
        hue_order=conditions, order=order, palette=["#1f77b4", "#ff7f0e", "#2ca02c"],
        errorbar=None, dodge=True, width=0.8, edgecolor="black", linewidth=0.8, ax=ax
    )

    ax.axhline(y=norm.ppf(0.05), color="red", linestyle="--", linewidth=1.4, label="Detection Threshold (p=0.05)")
    ax.set_title("Watermark Robustness: Median Z-Score Under Paraphrase Attacks", fontsize=17, pad=18)
    ax.set_xlabel("Watermark Configuration", fontsize=13)
    ax.set_ylabel("Median Z-Score (Lower is stronger detection)", fontsize=13)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(title="Text Condition", bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0, frameon=True, fontsize=11, title_fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(PLOT_ROBUSTNESS_BAR_PATH, dpi=300, bbox_inches="tight")
    print(f"  Robustness bar plot saved to '{PLOT_ROBUSTNESS_BAR_PATH}'")
    plt.close()


# Z-Score vs. Tokens Step Plot
print("  Generating multi-condition Z-Score vs. Tokens step plots...")

if df_step_plot.empty:
    print("  Skipping step plots: No data found.")
else:
    sns.set_theme(style="whitegrid", context="talk")
    conditions = ["Original", "Translation", "T5 Attack"]
    fig, axes = plt.subplots(1, len(conditions), figsize=(22, 6), sharey=True)

    palette = {
        "Human Text": "#808080", "LLM Baseline": "#1f77b4",
        "Senso-Sync (Weak)": "#66c2a5", "RedGreen (Weak)": "#8da0cb", "Full Adaptive (Weak)": "#4daf4a",
        "Senso-Sync (Medium)": "#ff7f00", "RedGreen (Medium)": "#fdb462", "Full Adaptive (Medium)": "#d95f02",
        "Senso-Sync (Strong)": "#e41a1c", "RedGreen (Strong)": "#fb8072", "Full Adaptive (Strong)": "#a50f15",
    }

    for ax, cond in zip(axes, conditions):
        df_cond = df_step_plot[df_step_plot["Condition"] == cond]
        if df_cond.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        sns.lineplot(
            data=df_cond, x="Num Opportunities", y="Z-Score", hue="Configuration",
            estimator=np.median, errorbar=None, lw=2.0, alpha=0.95, palette=palette, ax=ax
        )
        ax.axhline(y=norm.ppf(0.05), color="red", linestyle="--", linewidth=1.3, label="Threshold (p=0.05)")
        ax.set_title(cond, fontsize=15, pad=12)
        ax.set_xlabel("Number of Watermark Opportunities", fontsize=12)
        if cond == "Original":
            ax.set_ylabel("Median Z-Score (Lower is stronger detection)", fontsize=12)
        else:
            ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(labelsize=10)
        ax.legend_.remove()

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Configuration", bbox_to_anchor=(1.01, 0.5),
               loc="center left", frameon=True, fontsize=10, title_fontsize=11)

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(PLOT_STEP_PLOT_PATH, dpi=300, bbox_inches="tight")
    print(f"  Step plots saved to '{PLOT_STEP_PLOT_PATH}'")
    plt.close()

# Mann–Whitney U Test & Heatmap
print("  Performing Mann–Whitney U tests and generating heatmap...")

u_results = []
for config_a, config_b in combinations(original_data.keys(), 2):
    zs_a = [d['final_zscore'] for d in original_data[config_a] if 'final_zscore' in d]
    zs_b = [d['final_zscore'] for d in original_data[config_b] if 'final_zscore' in d]
    
    if len(zs_a) > 1 and len(zs_b) > 1:
        try:
            stat, p = mannwhitneyu(zs_a, zs_b, alternative='two-sided')
            u_results.append({
                'Config A': config_name_mapping.get(config_a, config_a),
                'Config B': config_name_mapping.get(config_b, config_b),
                'U-Statistic': stat,
                'p-Value': p
            })
        except Exception as e:
            print(f"    Skipped {config_a} vs {config_b}: {e}")

if u_results:
    df_u = pd.DataFrame(u_results)
    reject, pvals_corr, _, _ = multipletests(df_u['p-Value'], method='bonferroni', alpha=0.05)
    df_u['Adjusted p-Value'] = pvals_corr
    df_u['Significant'] = reject
    df_u.to_csv(U_TEST_CSV_PATH, index=False)
    print(f"  Mann–Whitney U results saved to '{U_TEST_CSV_PATH}'")

    # Generate Heatmap
    configs = sorted(set(df_u['Config A']).union(df_u['Config B']))
    p_matrix = pd.DataFrame(index=configs, columns=configs, dtype=float)
    
    for _, row in df_u.iterrows():
        p_matrix.loc[row['Config A'], row['Config B']] = row['Adjusted p-Value']
        p_matrix.loc[row['Config B'], row['Config A']] = row['Adjusted p-Value']
    
    np.fill_diagonal(p_matrix.values, np.nan)

    sns.set_theme(style="ticks", context="paper")
    plt.figure(figsize=(10, 8))
    mask = np.isnan(p_matrix.values)
    
    sns.heatmap(
        p_matrix, mask=mask, cmap="viridis_r", annot=True, fmt=".3f",
        linewidths=0.4, cbar_kws={'label': 'Adjusted p-value (Bonferroni)'},
        vmin=0, vmax=1.0, annot_kws={'size': 7}
    )
    plt.title("Pairwise Mann–Whitney U Tests on Z-Scores (Original Condition)", fontsize=11, pad=8)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_U_HEATMAP_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Mann–Whitney U heatmap saved to '{PLOT_U_HEATMAP_PATH}'")

else:
    print("  No valid data for Mann–Whitney U tests.")

print("  All processing and plotting complete.")