import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, mannwhitneyu
import math
import pandas as pd
import seaborn as sns
import matplotlib.font_manager as fm
from itertools import combinations
import os

DETECTION_FILE_PATH = os.environ.get('DETECTION_FILE_PATH', 'outputs/detection_results.json')
PLOT_OUTPUT_PATH = os.environ.get('PLOT_OUTPUT_PATH', 'outputs/results_boxplot.png')

print(f"  Loading detection data from '{DETECTION_FILE_PATH}'...")
try:
    with open(DETECTION_FILE_PATH, 'r') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} detection samples.")
except FileNotFoundError:
    print(f"  ERROR: Cannot find '{DETECTION_FILE_PATH}'. Cannot generate plot.")
    data = []

if data:
    name = 'Boxplot of Z-Scores for Different Configurations (Llama 3.1 8B, Medium Setting)'

    all_configurations = list(data[0]['detection'].keys())
    if 'llm_baseline' not in all_configurations:
        print("  ERROR: 'llm_baseline' not in detection data. Plotting may fail.")
        configurations = all_configurations
    else:
        configurations = [config for config in all_configurations if config != 'llm_baseline']
        if 'baseline' in all_configurations and 'baseline' not in configurations:
            configurations.insert(0, 'baseline')
            
    final_scores = {config: [] for config in configurations}
    perplexity_increments = {config: [] for config in configurations}
    baseline_perplexities = []

    for entry in data:
        detection = entry.get('detection', {})
        if 'llm_baseline' not in detection:
            continue

        baseline_perplexity = detection['llm_baseline'].get('perplexity', np.nan)
        if not np.isnan(baseline_perplexity):
            baseline_perplexities.append(baseline_perplexity)
        
        for config in configurations:
            if config not in detection:
                continue
            
            final_score = detection[config].get('final_score', np.nan)
            perplexity = detection[config].get('perplexity', np.nan)

            if not np.isnan(perplexity) and not np.isnan(final_score):
                perplexity_increment = perplexity - baseline_perplexity
                perplexity_increments[config].append(perplexity_increment)
                final_scores[config].append(final_score)

    if not baseline_perplexities:
        median_baseline_perplexity = 0
        print("  Warning: No valid 'llm_baseline' perplexity values found.")
    else:
        median_baseline_perplexity = np.median(baseline_perplexities)
    print(f"  Median 'llm_baseline' Perplexity: {median_baseline_perplexity:.4f}")

    def prob_to_z_score(prob):
        if prob <= 0: prob = 1e-12
        elif prob >= 1: prob = 1 - 1e-12
        return norm.ppf(prob)

    z_scores = {config: [prob_to_z_score(score) for score in final_scores[config]] for config in configurations}
    median_perplexity_increment = {config: np.median(perps) if perps else 0 for config, perps in perplexity_increments.items()}
    
    config_name_mapping = {
        "baseline": "Human",
        "llm_senso_medium": "Senso",
        "llm_acro_medium": "Acro",
        "llm_redgreen_medium": "Red-Green",
        "llm_both_medium": "Senso +\nAcro",
        "llm_senso_redgreen_medium": "Senso +\nRed-Green",
        "llm_acro_redgreen_medium": "Acro +\nRed-Green",
        "llm_all_three_medium": "All Three"
    }
    
    plot_configs = [c for c in configurations if c in config_name_mapping]
    plot_labels = [config_name_mapping[c] for c in plot_configs]
    plot_z_scores = [z_scores[c] for c in plot_configs]
    
    print("  Generating plot...")
    sns.set_theme()
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    box = ax.boxplot(plot_z_scores, labels=plot_labels, patch_artist=True, medianprops=dict(color='black'))

    colors = ['#D3D3D3'] + ['#ADD8E6'] * (len(plot_labels) - 2) + ['#FF6347']
    if 'All Three' not in plot_labels:
        colors = ['#D3D3D3'] + ['#ADD8E6'] * (len(plot_labels) - 1)
        
    for patch, color in zip(box['boxes'], colors[:len(box['boxes'])]):
        patch.set_facecolor(color)

    ax.set_ylabel('Z-Scores (Lower is Better)')
    ax.set_title(name)
    z_score_threshold = norm.ppf(0.005)
    ax.axhline(y=z_score_threshold, color='red', linestyle='--', label=f'Detection Threshold (p = 0.005)')
    
    ylim = ax.get_ylim()
    for i, config_key in enumerate(plot_configs, start=1):
        med_perp = median_perplexity_increment[config_key]
        scores = z_scores[config_key]
        if not scores: continue
        percentage_below = (sum(z < z_score_threshold for z in scores) / len(scores)) * 100
        
        y_pos = ylim[0] - (ylim[1] - ylim[0]) * 0.1
        ax.text(i, y_pos, f'PPLX +{med_perp:.2f}\nDet: {percentage_below:.1f}%', ha='center', va='top', fontsize=9)
    
    ax.legend()
    plt.grid(True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"  Plot saved to '{PLOT_OUTPUT_PATH}'")
    plt.close(fig)
else:
    print("  Skipping plotting as no detection data was loaded.")