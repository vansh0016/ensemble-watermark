import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, mannwhitneyu
import pandas as pd
import seaborn as sns
import os
from itertools import product, combinations
from statsmodels.sandbox.stats.multicomp import multipletests

DETECTION_ORIGINAL_FILE = os.environ.get('DETECTION_ORIGINAL_FILE', 'outputs/detection_results.json')
DETECTION_ATTACK_TRANSLATION_FILE = os.environ.get('DETECTION_ATTACK_TRANSLATION_FILE', 'outputs/detection_results_attack_translation.json')
DETECTION_ATTACK_T5_FILE = os.environ.get('DETECTION_ATTACK_T5_FILE', 'outputs/detection_results_attack_t5.json')

PLOT_CONFIG_COMPARISON_HEATMAP_PATH = os.environ.get('PLOT_CONFIG_COMPARISON_HEATMAP_PATH', 'outputs/config_comparison_heatmap.png')
PLOT_TRADEOFF_SCATTER_PATH = os.environ.get('PLOT_TRADEOFF_SCATTER_PATH', 'outputs/attack_tradeoff_scatter.png')
ALPHA = 0.05


def load_detection_data(filepath, data_type='z-score'):
    try:
        with open(filepath, 'r') as f: data = json.load(f)
    except FileNotFoundError: print(f"  WARNING: Could not find file: {filepath}"); return None
    except json.JSONDecodeError: print(f"  WARNING: Could not decode JSON from: {filepath}"); return None
    results = {}
    for entry in data:
        detection = entry.get('detection', {})
        for config, scores in detection.items():
            if 'error' in scores: continue
            value = np.nan
            if data_type == 'z-score':
                final_score = scores.get('final_score')
                value = prob_to_z_score(final_score)
            elif data_type == 'perplexity_increase':
                baseline_pplx = detection.get('llm_baseline', {}).get('perplexity')
                current_pplx = scores.get('perplexity')
                try:
                    baseline_pplx = float(baseline_pplx) if baseline_pplx is not None else np.nan
                    current_pplx = float(current_pplx) if current_pplx is not None else np.nan
                    if not np.isnan(baseline_pplx) and not np.isnan(current_pplx):
                         increase = current_pplx - baseline_pplx
                         value = increase if not (np.isinf(increase) or np.isnan(increase)) else np.nan
                    else: value = np.nan
                except: value = np.nan
            if not np.isnan(value):
                if config not in results: results[config] = []
                results[config].append(value)
    return results

def prob_to_z_score(prob):
    if prob is None: return np.nan
    try:
        prob = float(prob);
        if prob <= 0: prob = 1e-12
        elif prob >= 1: prob = 1 - 1e-12
        z = norm.ppf(prob); return np.clip(z, -10, 10)
    except: return np.nan

config_name_mapping = {
    "baseline": "Human", "llm_baseline": "LLM Baseline",
    "llm_senso_medium": "Senso", "llm_acro_medium": "Acro", "llm_redgreen_medium": "Red-Green",
    "llm_both_medium": "Senso+Acro", "llm_senso_redgreen_medium": "Senso+RG",
    "llm_acro_redgreen_medium": "Acro+RG", "llm_all_three_medium": "All Three"
}
config_order = [
    "baseline", "llm_baseline", "llm_senso_medium", "llm_acro_medium", "llm_redgreen_medium",
    "llm_both_medium", "llm_senso_redgreen_medium", "llm_acro_redgreen_medium", "llm_all_three_medium"
]
ordered_keys = [c for c in config_order if c in config_name_mapping]
ordered_labels = [config_name_mapping[c] for c in ordered_keys]

print("  Loading detection data for analysis...")
original_z = load_detection_data(DETECTION_ORIGINAL_FILE, 'z-score')
trans_z = load_detection_data(DETECTION_ATTACK_TRANSLATION_FILE, 'z-score')
t5_z = load_detection_data(DETECTION_ATTACK_T5_FILE, 'z-score')
original_pplx_inc = load_detection_data(DETECTION_ORIGINAL_FILE, 'perplexity_increase')
trans_pplx_inc = load_detection_data(DETECTION_ATTACK_TRANSLATION_FILE, 'perplexity_increase')
t5_pplx_inc = load_detection_data(DETECTION_ATTACK_T5_FILE, 'perplexity_increase')
if not original_z:
    print("  ERROR: Cannot proceed without original Z-score data for heatmap.")
    exit()

# 1. Configuration Comparison Heatmap
print("  Performing pairwise Mann-Whitney U tests between configurations (using original data)...")
pairwise_results = []
config_pairs = list(combinations(ordered_keys, 2))

for (config1_key, config2_key) in config_pairs:
    data1 = original_z.get(config1_key, [])
    data2 = original_z.get(config2_key, [])

    if len(data1) > 1 and len(data2) > 1:
        try:
            stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            pairwise_results.append({
                'config1': config1_key,
                'config2': config2_key,
                'p_value': p_value
            })
        except ValueError as e:
             print(f"    Skipping test between {config1_key} and {config2_key}: {e}")
             pairwise_results.append({'config1': config1_key, 'config2': config2_key, 'p_value': 1.0})
    else:
         pairwise_results.append({'config1': config1_key, 'config2': config2_key, 'p_value': np.nan})

if pairwise_results:
    p_values_only = [res['p_value'] for res in pairwise_results if not np.isnan(res['p_value'])]
    if p_values_only:
        reject, pvals_corrected, _, _ = multipletests(p_values_only, method='bonferroni', alpha=ALPHA)
        corrected_idx = 0
        for res in pairwise_results:
            if not np.isnan(res['p_value']):
                res['adjusted_p'] = pvals_corrected[corrected_idx]; corrected_idx += 1
            else: res['adjusted_p'] = np.nan
    else:
        for res in pairwise_results: res['adjusted_p'] = res['p_value'] # Use raw p if correction failed

    p_value_matrix = pd.DataFrame(index=ordered_labels, columns=ordered_labels, dtype=float)
    p_value_matrix[:] = np.nan

    for res in pairwise_results:
        label1 = config_name_mapping.get(res['config1'])
        label2 = config_name_mapping.get(res['config2'])
        if label1 and label2:
            p_value_matrix.loc[label1, label2] = res['adjusted_p']
            p_value_matrix.loc[label2, label1] = res['adjusted_p']

    np.fill_diagonal(p_value_matrix.values, np.nan)

    print("  Generating configuration comparison heatmap...")
    plt.figure(figsize=(10, 8))
    sns.set_theme()
    sns.set_style("ticks")

    mask = np.isnan(p_value_matrix.values)
    sns.heatmap(
        p_value_matrix,
        mask=mask,
        annot=True, fmt=".3f", cmap="viridis_r",
        linewidths=0.5,
        cbar_kws={'label': 'Adjusted p-value (Bonferroni)'},
        vmin=0, vmax=1.0
    )
    plt.title('Pairwise Comparison of Watermark Configurations (Original Z-Scores)\nMann-Whitney U Test with Bonferroni Correction')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOT_CONFIG_COMPARISON_HEATMAP_PATH, dpi=300)
    print(f"  Configuration comparison heatmap saved to '{PLOT_CONFIG_COMPARISON_HEATMAP_PATH}'")
    plt.close()
else:
    print("  Skipping configuration comparison heatmap: No pairwise results.")


# 2. Trade-off Scatter Plot
print("  Calculating median values for trade-off scatter plot...")
scatter_data = []
conditions = {'Original': (original_z, original_pplx_inc),
              'Translation': (trans_z, trans_pplx_inc),
              'T5 Attack': (t5_z, t5_pplx_inc)}

if not trans_z or not trans_pplx_inc:
    print("  Skipping Translation condition in scatter plot: Data missing.")
    conditions.pop('Translation', None)
if not t5_z or not t5_pplx_inc:
     print("  Skipping T5 Attack condition in scatter plot: Data missing.")
     conditions.pop('T5 Attack', None)


for config_key in config_order:
    if config_key not in config_name_mapping or config_key in ['baseline', 'llm_baseline']: continue
    config_label = config_name_mapping[config_key]
    for condition, (z_data, pplx_inc_data) in conditions.items():
        if z_data and pplx_inc_data and config_key in z_data and config_key in pplx_inc_data:
            z_scores = z_data[config_key]; pplx_increases = pplx_inc_data[config_key]
            if z_scores and pplx_increases:
                median_z = np.median(z_scores); median_pplx_inc = np.median(pplx_increases)
                if not np.isnan(median_z) and not np.isnan(median_pplx_inc):
                    scatter_data.append({'Configuration': config_label, 'Condition': condition,
                                         'Median Z-Score': median_z, 'Median Perplexity Increase': median_pplx_inc})

if not scatter_data:
    print("  Skipping trade-off scatter plot: No valid median data calculated.")
else:
    df_scatter = pd.DataFrame(scatter_data)
    print("  Generating trade-off scatter plot...")
    plt.figure(figsize=(10, 8)); sns.set_theme(style="whitegrid")
    sns.scatterplot(data=df_scatter, x='Median Perplexity Increase', y='Median Z-Score',
                    hue='Configuration', style='Condition', s=150, palette='tab10')
    z_score_threshold = norm.ppf(0.005)
    plt.axhline(y=z_score_threshold, color='red', linestyle='--', label=f'Detection Threshold (p = 0.005)')
    plt.title('Watermark Trade-off: Detection Strength vs. Perplexity Cost (Median Values)')
    plt.xlabel('Median Perplexity Increase (vs. LLM Baseline)'); plt.ylabel('Median Z-Score (Lower is Stronger Detection)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True); plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(PLOT_TRADEOFF_SCATTER_PATH, dpi=300)
    print(f"  Trade-off scatter plot saved to '{PLOT_TRADEOFF_SCATTER_PATH}'")
    plt.close()

print("  Analysis complete.")