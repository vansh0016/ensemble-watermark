# pipeline_scripts/plot_t5_attack_results.py
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import os

# === Parameters ===
# Assumes run_all.py was run to create the original detection file
DETECTION_ORIGINAL_FILE = os.environ.get('DETECTION_ORIGINAL_FILE', 'outputs/detection_results.json')
# This file is created by the T5 attack pipeline steps
DETECTION_ATTACK_T5_FILE = os.environ.get('DETECTION_ATTACK_T5_FILE', 'outputs/detection_results_attack_t5.json')
# Output plot file
PLOT_T5_ATTACK_OUTPUT_PATH = os.environ.get('PLOT_T5_ATTACK_OUTPUT_PATH', 'outputs/t5_attack_results_plot.png')
# ---

def load_detection_data(filepath):
    """Loads detection data and converts final_scores to z-scores."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"  WARNING: Could not find detection file: {filepath}")
        return None
    except json.JSONDecodeError:
         print(f"  WARNING: Could not decode JSON from: {filepath}")
         return None

    results = {}
    for entry in data:
        detection = entry.get('detection', {})
        for config, scores in detection.items():
            if 'error' in scores: continue
            final_score = scores.get('final_score')
            if final_score is not None:
                z_score = prob_to_z_score(final_score)
                if config not in results: results[config] = []
                # Ensure z_score is a standard float, handle potential numpy types
                if not np.isnan(z_score):
                     results[config].append(float(z_score))
    return results

def prob_to_z_score(prob):
    """Converts p-value (final_score) to z-score."""
    if prob is None: return np.nan
    try:
        prob = float(prob)
        if prob <= 0: prob = 1e-12
        elif prob >= 1: prob = 1 - 1e-12
        z = norm.ppf(prob)
        # Cap z-scores at reasonable bounds if needed, e.g., -10 to 10
        return np.clip(z, -10, 10)
    except (ValueError, TypeError):
        return np.nan # Return NaN if conversion fails

print("  Loading detection results for T5 comparison...")
original_z_scores = load_detection_data(DETECTION_ORIGINAL_FILE)
t5_z_scores = load_detection_data(DETECTION_ATTACK_T5_FILE)

if not original_z_scores or not t5_z_scores:
    print("  ERROR: Cannot generate plot. Missing original or T5 attack detection results.")
else:
    # --- Data Preparation for Plotting ---
    plot_data = []
    config_name_mapping = {
        "baseline": "Human",
        "llm_baseline": "LLM Baseline",
        "llm_senso_medium": "Senso",
        "llm_acro_medium": "Acro",
        "llm_redgreen_medium": "Red-Green",
        "llm_both_medium": "Senso+Acro",
        "llm_senso_redgreen_medium": "Senso+RG",
        "llm_acro_redgreen_medium": "Acro+RG",
        "llm_all_three_medium": "All Three"
    }
    config_order = [
        "baseline", "llm_baseline", "llm_senso_medium", "llm_acro_medium",
        "llm_redgreen_medium", "llm_both_medium", "llm_senso_redgreen_medium",
        "llm_acro_redgreen_medium", "llm_all_three_medium"
    ]

    for config in config_order:
        if config not in original_z_scores and config not in t5_z_scores: continue

        label = config_name_mapping.get(config, config)

        # Original
        original_scores = original_z_scores.get(config, [])
        for z in original_scores:
            if not np.isnan(z):
                plot_data.append({'Configuration': label, 'Attack': 'Original', 'Z-Score': z})

        # T5 Attack
        t5_scores = t5_z_scores.get(config, [])
        for z in t5_scores:
             if not np.isnan(z):
                plot_data.append({'Configuration': label, 'Attack': 'T5 Paraphrase', 'Z-Score': z})

    if not plot_data:
        print(" ERROR: No valid data found to plot after processing results.")
    else:
        df_plot = pd.DataFrame(plot_data)

        # --- Plotting ---
        print("  Generating T5 attack comparison plot...")
        sns.set_theme(style="ticks")
        plt.figure(figsize=(14, 7))

        # Boxplot comparing Original vs T5 Attack
        sns.boxplot(
            data=df_plot,
            x='Configuration',
            y='Z-Score',
            hue='Attack',
            palette={'Original': '#ADD8E6', 'T5 Paraphrase': '#FFB6C1'}, # Example: Light Blue, Light Pink
            showfliers=False # Hide outliers for clarity
        )

        z_score_threshold = norm.ppf(0.005) # p = 0.005
        plt.axhline(y=z_score_threshold, color='red', linestyle='--', label=f'Detection Threshold (p = 0.005)')

        plt.title('Watermark Detection Z-Scores Before and After T5 Paraphrase Attack')
        plt.ylabel('Z-Score (Lower indicates stronger detection)')
        plt.xlabel('Watermark Configuration')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Attack Type')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(PLOT_T5_ATTACK_OUTPUT_PATH, dpi=300)
        print(f"  T5 attack comparison plot saved to '{PLOT_T5_ATTACK_OUTPUT_PATH}'")
        plt.close()