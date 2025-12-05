# pipeline_scripts/generate_all_plots.py
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, mannwhitneyu
import pandas as pd
import seaborn as sns
import os
import sys
from itertools import combinations
from statsmodels.sandbox.stats.multicomp import multipletests

DETECTION_FILE = os.environ.get('DETECTION_FILE_PATH', 'outputs/detection_results.json')
PLOT_BOXPLOT_PATH_PREFIX = os.environ.get('PLOT_BOXPLOT_PATH_PREFIX', 'outputs/results_boxplot')
PLOT_HEATMAP_PATH = os.environ.get('PLOT_HEATMAP_PATH', 'outputs/config_comparison_heatmap.png')
PLOT_SCATTER_PATH = os.environ.get('PLOT_SCATTER_PATH', 'outputs/delta_tradeoff_scatter.png')
PLOT_STEP_PLOT_PATH = os.environ.get('PLOT_STEP_PLOT_PATH', 'outputs/detection_rate_vs_sentences.png')
ALPHA = 0.05

def load_detection_data_for_plotting(filepath):
    """Loads detection data and calculates z-score and perplexity increase."""
    try:
        with open(filepath, 'r') as f: data = json.load(f)
    except FileNotFoundError: print(f"  ERROR: Could not find file: {filepath}"); return None
    except json.JSONDecodeError: print(f"  WARNING: Could not decode JSON from: {filepath}"); return None
    
    results = {} 
    raw_sample_data = [] 
    
    for entry in data:
        detection = entry.get('detection', {})
        
        baseline_pplx = detection.get('llm_baseline', {}).get('perplexity')
        try:
            baseline_pplx = float(baseline_pplx) if baseline_pplx is not None else np.nan
            if np.isinf(baseline_pplx): baseline_pplx = np.nan
        except: baseline_pplx = np.nan
        
        for config, scores in detection.items():
            if 'error' in scores: continue
            if config not in results:
                results[config] = {'z_scores': [], 'pplx_increases': [], 'final_scores': []}
            
            # 1. Get Z-Score
            final_score = scores.get('final_score')
            z_score = prob_to_z_score(final_score)
            if not np.isnan(z_score):
                results[config]['z_scores'].append(float(z_score))
                results[config]['final_scores'].append(float(final_score))
                
            # 2. Get Perplexity Increase
            current_pplx = scores.get('perplexity')
            try:
                current_pplx = float(current_pplx) if current_pplx is not None else np.nan
                if np.isinf(current_pplx): current_pplx = np.nan
                if not np.isnan(baseline_pplx) and not np.isnan(current_pplx):
                     increase = current_pplx - baseline_pplx
                     if not (np.isinf(increase) or np.isnan(increase)):
                         results[config]['pplx_increases'].append(float(increase))
            except: pass

            # 3. Get Num Sentences for Step Plot
            num_sentences = scores.get('num_sentences')
            if not np.isnan(z_score) and num_sentences is not None:
                raw_sample_data.append({
                    'Config Key': config,
                    'Z-Score': float(z_score),
                    'Num Sentences': int(num_sentences)
                })
                
    return results, raw_sample_data

def prob_to_z_score(prob):
    if prob is None: return np.nan
    try:
        prob = float(prob);
        if prob <= 0: prob = 1e-12
        elif prob >= 1: prob = 1 - 1e-12
        z = norm.ppf(prob); return np.clip(z, -10, 10)
    except: return np.nan

config_name_mapping = {
    "baseline": ("Baseline", "Baseline"), "llm_baseline": ("LLM Baseline", "Baseline"),
    "llm_senso_weak": ("SENSO", "Weak Delta"), "llm_acro_weak": ("ACRO", "Weak Delta"),
    "llm_redgreen_weak": ("RedGreen", "Weak Delta"), "llm_both_weak": ("SENSO+ACRO", "Weak Delta"),
    "llm_senso_redgreen_weak": ("SENSO+RG", "Weak Delta"), "llm_acro_redgreen_weak": ("ACRO+RG", "Weak Delta"),
    "llm_all_three_weak": ("All Three", "Weak Delta"),
    "llm_senso_medium": ("SENSO", "Medium Delta"), "llm_acro_medium": ("ACRO", "Medium Delta"),
    "llm_redgreen_medium": ("RedGreen", "Medium Delta"), "llm_both_medium": ("SENSO+ACRO", "Medium Delta"),
    "llm_senso_redgreen_medium": ("SENSO+RG", "Medium Delta"), "llm_acro_redgreen_medium": ("ACRO+RG", "Medium Delta"),
    "llm_all_three_medium": ("All Three", "Medium Delta"),
    "llm_senso_strong": ("SENSO", "Strong Delta"), "llm_acro_strong": ("ACRO", "Strong Delta"),
    "llm_redgreen_strong": ("RedGreen", "Strong Delta"), "llm_both_strong": ("SENSO+ACRO", "Strong Delta"),
    "llm_senso_redgreen_strong": ("SENSO+RG", "Strong Delta"), "llm_acro_redgreen_strong": ("ACRO+RG", "Strong Delta"),
    "llm_all_three_strong": ("All Three", "Strong Delta"),
}
config_order_labels = ['Human', 'LLM Baseline', 'SENSO', 'ACRO', 'RedGreen', 'SENSO+ACRO', 'SENSO+RG', 'ACRO+RG', 'All Three']


print(f"  Loading and processing detection data from '{DETECTION_FILE}'...")
plot_data, raw_sample_data = load_detection_data_for_plotting(DETECTION_FILE)
if not plot_data:
    print("  ERROR: No data loaded. Exiting plotting script.")
    sys.exit(1)

flat_data = []
for config_key, values in plot_data.items():
    if config_key not in config_name_mapping: continue
    config_label, delta_label = config_name_mapping[config_key]
    median_z = np.median(values['z_scores']) if values['z_scores'] else np.nan
    median_pplx_inc = np.median(values['pplx_increases']) if values['pplx_increases'] else np.nan
    z_score_threshold = norm.ppf(0.005)
    detection_rate = (sum(z < z_score_threshold for z in values['z_scores']) / len(values['z_scores'])) * 100 if values['z_scores'] else 0
    flat_data.append({
        'Config Key': config_key, 'Configuration': config_label, 'Delta Level': delta_label,
        'Median Z-Score': median_z, 'Median PPLX Increase': median_pplx_inc,
        'Detection Rate (%)': detection_rate, 'Z-Scores': values['z_scores']
    })
df_plot = pd.DataFrame(flat_data)
df_plot['Configuration'] = df_plot['Configuration'].replace('Baseline', 'Human')


# 1. Generate Box Plots
print("  Generating Z-Score Box Plots (per delta level)...")
z_score_threshold = norm.ppf(0.005)
delta_levels = ['Weak Delta', 'Medium Delta', 'Strong Delta']
for delta in delta_levels:
    print(f"    Plotting for: {delta}")
    df_boxplot = df_plot[df_plot['Delta Level'] == delta].copy()
    df_baselines = df_plot[df_plot['Delta Level'] == 'Baseline']
    df_boxplot = pd.concat([df_baselines, df_boxplot]).reset_index()
    df_boxplot = df_boxplot.set_index('Configuration').reindex(config_order_labels).reset_index()
    df_boxplot = df_boxplot.dropna(subset=['Configuration', 'Z-Scores'])
    if df_boxplot.empty:
        print(f"    Skipping box plot for {delta}: No data found.")
        continue
    plt.figure(figsize=(12, 7)); sns.set_theme(style="ticks")
    box = plt.boxplot([data for data in df_boxplot['Z-Scores'] if data],
                      labels=df_boxplot['Configuration'], patch_artist=True, medianprops=dict(color='black'))
    colors = ['#D3D3D3', '#D3D3D3'] + ['#ADD8E6'] * (len(df_boxplot) - 3) + ['#FF6347']
    for patch, color in zip(box['boxes'], colors[:len(box['boxes'])]): patch.set_facecolor(color)
    plt.ylabel('Z-Scores (Lower is Better)'); plt.title(f'Boxplot of Z-Scores ({delta} Setting)')
    plt.axhline(y=z_score_threshold, color='red', linestyle='--', label=f'Detection Threshold (p = 0.005)')
    ylim = plt.gca().get_ylim()
    for i, row in df_boxplot.iterrows():
        y_pos = ylim[0] - (ylim[1] - ylim[0]) * 0.15
        plt.text(i + 1, y_pos, f"PPLX +{row['Median PPLX Increase']:.2f}\nDet: {row['Detection Rate (%)']:.1f}%",
                 ha='center', va='top', fontsize=9)
    plt.legend(loc='upper right'); plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    output_filename = f"{PLOT_BOXPLOT_PATH_PREFIX}_{delta.lower().replace(' ', '_')}.png"
    plt.savefig(output_filename, dpi=300); plt.close()
    print(f"    Boxplot saved to '{output_filename}'")


# 2. Generate Config Comparison Heatmap (Medium Delta)
print("  Generating Configuration Comparison Heatmap (Medium Delta)...")
medium_z_data = {
    key: data['z_scores']
    for key, data in plot_data.items()
    if config_name_mapping.get(key, ('', ''))[1] == 'Medium Delta' and data['z_scores']
}
baseline_z_data = {
    key: data['z_scores']
    for key, data in plot_data.items()
    if config_name_mapping.get(key, ('', ''))[1] == 'Baseline' and data['z_scores']
}
medium_z_data.update(baseline_z_data)
medium_z_data_labeled = {config_name_mapping[k][0]: v for k, v in medium_z_data.items()}
heatmap_labels = [label for label in config_order_labels if label in medium_z_data_labeled]
pairwise_results = []
config_pairs = list(combinations(heatmap_labels, 2))
for (label1, label2) in config_pairs:
    data1 = medium_z_data_labeled.get(label1, []); data2 = medium_z_data_labeled.get(label2, [])
    if len(data1) > 1 and len(data2) > 1:
        try: stat, p = mannwhitneyu(data1, data2, alternative='two-sided');
        except ValueError: p = 1.0
    else: p = np.nan
    pairwise_results.append({'config1': label1, 'config2': label2, 'p_value': p})
if pairwise_results:
    p_values_only = [res['p_value'] for res in pairwise_results if not np.isnan(res['p_value'])]
    if p_values_only:
        reject, pvals_corrected, _, _ = multipletests(p_values_only, method='bonferroni', alpha=ALPHA)
        corrected_idx = 0
        for res in pairwise_results:
            if not np.isnan(res['p_value']): res['adjusted_p'] = pvals_corrected[corrected_idx]; corrected_idx += 1
            else: res['adjusted_p'] = np.nan
    else:
        for res in pairwise_results: res['adjusted_p'] = res['p_value']
    p_value_matrix = pd.DataFrame(index=heatmap_labels, columns=heatmap_labels, dtype=float); p_value_matrix[:] = np.nan
    for res in pairwise_results:
        p_value_matrix.loc[res['config1'], res['config2']] = res['adjusted_p']
        p_value_matrix.loc[res['config2'], res['config1']] = res['adjusted_p']
    np.fill_diagonal(p_value_matrix.values, np.nan)
    plt.figure(figsize=(10, 8)); sns.set_theme(style="ticks")
    mask = np.isnan(p_value_matrix.values)
    sns.heatmap(p_value_matrix, mask=mask, annot=True, fmt=".3f", cmap="viridis_r",
                linewidths=0.5, cbar_kws={'label': 'Adjusted p-value (Bonferroni)'}, vmin=0, vmax=1.0)
    plt.title('Pairwise Comparison of Watermark Configurations (Medium Delta Z-Scores)')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOT_HEATMAP_PATH, dpi=300); plt.close()
    print(f"  Configuration heatmap saved to '{PLOT_HEATMAP_PATH}'")
else:
    print("  Skipping configuration heatmap: No pairwise results calculated.")


# 3. Generate Delta Trade-off Scatter Plot (All Deltas)
print("  Generating Delta Trade-off Scatter Plot...")
df_scatter = df_plot[df_plot['Configuration'] != 'Human']  # Remove Human rows
delta_levels_present = [lvl for lvl in df_scatter['Delta Level'].unique() if lvl != 'Baseline']
if df_scatter.empty or len(delta_levels_present) < 1:
    print("  Skipping delta trade-off scatter plot (no watermark data).")
else:
    custom_colors = {'Strong Delta': '#4682B4', 'Medium Delta': '#FFD700', 'Weak Delta': '#FF6347'}
    custom_markers = {'ACRO': 'o', 'ACRO+RG': 's', 'All Three': 'D', 'LLM Baseline': '*',
                      'RedGreen': 'v', 'SENSO': 'P', 'SENSO+ACRO': 'X', 'SENSO+RG': '^'}
    hue_order = ['Weak Delta', 'Medium Delta', 'Strong Delta']
    style_order = ['LLM Baseline', 'SENSO', 'ACRO', 'RedGreen', 'SENSO+ACRO',
                   'SENSO+RG', 'ACRO+RG', 'All Three']
    active_hue_labels = df_scatter['Delta Level'].unique()
    active_style_labels = df_scatter['Configuration'].unique()
    active_palette = {k: v for k, v in custom_colors.items() if k in active_hue_labels}
    active_markers = {k: v for k, v in custom_markers.items() if k in active_style_labels}
    active_hue_order = [h for h in hue_order if h in active_palette]
    active_style_order = [s for s in style_order if s in active_markers]
    
    plt.figure(figsize=(16, 8)); plt.rcParams.update({'font.size': 14})
    sns.set_context("talk", font_scale=1.2); sns.set_style("ticks")

    scatter = sns.scatterplot(
        data=df_scatter, x='Median PPLX Increase', y='Median Z-Score', # <-- Fixed
        hue='Delta Level', style='Configuration',
        hue_order=active_hue_order, style_order=active_style_order,
        palette=active_palette, markers=active_markers,
        s=300, edgecolor='gray', alpha=0.9
    )
    plt.title('Median Z-Score vs. Median Perplexity Increase by Delta Strength', fontsize=20)
    plt.xlabel('Median PPLX Increase (vs. LLM Baseline)', fontsize=16) # <-- Fixed
    plt.ylabel('Median Z-Score (Lower is Better Detection)', fontsize=16)
    plt.axhline(y=norm.ppf(0.005), color='red', linestyle='--', label='Detection Threshold (p=0.005)')
    
    handles, labels = scatter.get_legend_handles_labels()
    hue_map = {label: handle for handle, label in zip(handles, labels) if label in active_palette.keys()}
    style_map = {label: handle for handle, label in zip(handles, labels) if label in active_markers.keys()}
    hue_handles_ordered = [hue_map[lbl] for lbl in active_hue_order if lbl in hue_map]
    style_handles_ordered = [style_map[lbl] for lbl in active_style_order if lbl in style_map]
    
    first_legend = plt.legend(hue_handles_ordered, active_hue_order, title='Delta Strength',
                              bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.,
                              fontsize=12, title_fontsize=14)
    plt.gca().add_artist(first_legend)
    plt.legend(style_handles_ordered, active_style_order, title='Configuration',
               bbox_to_anchor=(1.03, 0.65), loc='upper left', borderaxespad=0.,
               fontsize=12, title_fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(PLOT_SCATTER_PATH, dpi=300, bbox_inches='tight')
    print(f"  Delta trade-off scatter plot saved to '{PLOT_SCATTER_PATH}'")
    plt.close()

# 4. Include Baseline Curve
print("  Generating Detection Rate vs. Sentences Step Plot...")
try:
    df_step_raw = pd.DataFrame(raw_sample_data)
    if df_step_raw.empty:
        raise ValueError("No raw sample data found for step plot.")
        
    df_step_raw['Configuration'] = df_step_raw['Config Key'].map(lambda x: config_name_mapping.get(x, ('', ''))[0])
    df_step_raw['Delta Level'] = df_step_raw['Config Key'].map(lambda x: config_name_mapping.get(x, ('', ''))[1])
    
    z_score_threshold = norm.ppf(0.005)
    df_step_raw['Detected'] = df_step_raw['Z-Score'] < z_score_threshold

    df_step_plot_data = df_step_raw[
        (df_step_raw['Configuration'] != 'Human') &
        (df_step_raw['Config Key'] != 'llm_baseline')
    ]
    if df_step_plot_data.empty:
        raise ValueError("No watermark data found for step plot.")

    df_agg = (
        df_step_plot_data.groupby(['Configuration', 'Delta Level', 'Num Sentences'])['Detected']
        .mean()
        .reset_index()
        .rename(columns={'Detected': 'Detection Rate'})
        .sort_values(by='Num Sentences')
    )

    baseline_df = df_step_raw[df_step_raw['Config Key'] == 'baseline']
    if baseline_df.empty:
        print("  WARNING: No baseline data found — skipping baseline curve.")
        baseline_curve = None
    else:
        baseline_curve = (
            baseline_df.groupby('Num Sentences')['Detected']
            .mean()
            .reset_index()
            .rename(columns={'Detected': 'Detection Rate'})
        )

    df_agg = df_agg[df_agg['Delta Level'].isin(['Weak Delta', 'Medium Delta', 'Strong Delta'])]
    plot_configs = [lbl for lbl in config_order_labels if lbl in df_agg['Configuration'].unique()]
    delta_colors = {
        'Weak Delta': '#6fd0ff',
        'Medium Delta': '#5a9bd5',
        'Strong Delta': '#002060'
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True, sharex=True)
    axes = axes.flatten()
    all_handles, all_labels = [], []

    for idx, config in enumerate(plot_configs):
        ax = axes[idx]
        df_config = df_agg[df_agg['Configuration'] == config]

        for delta_label, color in delta_colors.items():
            df_line = df_config[df_config['Delta Level'] == delta_label]
            if not df_line.empty:
                (line,) = ax.step(
                    df_line['Num Sentences'],
                    df_line['Detection Rate'],
                    where='post',
                    color=color,
                    linewidth=2,
                    label=delta_label.replace('Delta', 'Watermark Strength')
                )
                if delta_label not in all_labels:
                    all_handles.append(line)
                    all_labels.append(delta_label)

        if baseline_curve is not None and not baseline_curve.empty:
            (base_line,) = ax.step(
                baseline_curve['Num Sentences'],
                baseline_curve['Detection Rate'],
                where='post',
                color='gray',
                linestyle='--',
                linewidth=2,
                label='Baseline'
            )
            if 'Baseline' not in all_labels:
                all_handles.append(base_line)
                all_labels.append('Baseline')

        ax.set_title(config, fontsize=12, fontweight='bold')
        ax.set_xlim(0, df_agg['Num Sentences'].max() + 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx % 4 == 0:
            ax.set_ylabel("Detection Rate", fontsize=11)
        else:
            ax.set_ylabel("")
        if idx >= 4:
            ax.set_xlabel("Number of Sentences", fontsize=11)
        else:
            ax.set_xlabel("")

    for j in range(len(plot_configs), len(axes)):
        axes[j].axis("off")

    legend_labels = [
        lbl.replace("Delta", "Watermark Strength") if "Delta" in lbl else lbl
        for lbl in all_labels
    ]
    fig.legend(
        all_handles, legend_labels,
        title="LLMs",
        loc='lower right', bbox_to_anchor=(0.97, 0.03),
        fontsize=11, title_fontsize=12, frameon=True
    )

    plt.suptitle("Detection Rate vs. Number of Sentences Across Configurations", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plt.savefig(PLOT_STEP_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"  Step plot saved to '{PLOT_STEP_PLOT_PATH}'")
    plt.close()

except Exception as e:
    print(f"  Skipping step plot due to error: {e}")

print("  All plotting complete.")