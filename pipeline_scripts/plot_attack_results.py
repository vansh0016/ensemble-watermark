import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import os

# === Parameters ===
DETECTION_ORIGINAL_FILE = os.environ.get('DETECTION_FILE_PATH', 'outputs/detection_results.json')
DETECTION_ATTACK_TRANSLATION_FILE = os.environ.get('DETECTION_ATTACK_OUTPUT_FILE_TRANSLATION', 'outputs/detection_results_attack_translation.json')
DETECTION_ATTACK_T5_FILE = os.environ.get('DETECTION_ATTACK_OUTPUT_FILE_T5', 'outputs/detection_results_attack_t5.json')
PLOT_ATTACK_OUTPUT_PATH = os.environ.get('PLOT_ATTACK_OUTPUT_PATH', 'outputs/attack_results_comparison.png')
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
                results[config].append(z_score)
    return results

def prob_to_z_score(prob):
    """Converts p-value (final_score) to z-score."""
    if prob is None: return np.nan
    prob = float(prob)
    if prob <= 0: prob = 1e-12
    elif prob >= 1: prob = 1 - 1e-12
    z = norm.ppf(prob)
    # Cap z-scores at reasonable bounds if needed, e.g., -10 to 10
    return np.clip(z, -10, 10)

print("  Loading detection results for comparison...")
original_z_scores = load_detection_data(DETECTION_ORIGINAL_FILE)
translation_z_scores = load_detection_data(DETECTION_ATTACK_TRANSLATION_FILE)
t5_z_scores = load_detection_data(DETECTION_ATTACK_T5_FILE)

if not original_z_scores:
    print("  ERROR: Cannot plot without original detection results. Exiting plot script.")
else:
    # --- Data Preparation for Plotting ---
    plot_data = []
    # Config renaming from your notebook
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
    
    # Define order for plotting
    config_order = [
        "baseline", "llm_baseline", "llm_senso_medium", "llm_acro_medium", 
        "llm_redgreen_medium", "llm_both_medium", "llm_senso_redgreen_medium",
        "llm_acro_redgreen_medium", "llm_all_three_medium"
    ]

    for config in config_order:
        if config not in original_z_scores: continue # Skip if config not found
        
        label = config_name_mapping.get(config, config)
        
        # Original
        for z in original_z_scores.get(config, []):
            plot_data.append({'Configuration': label, 'Attack': 'Original', 'Z-Score': z})
            
        # Translation Attack
        if translation_z_scores and config in translation_z_scores:
            for z in translation_z_scores.get(config, []):
                plot_data.append({'Configuration': label, 'Attack': 'Translation', 'Z-Score': z})

        # T5 Attack
        if t5_z_scores and config in t5_z_scores:
             for z in t5_z_scores.get(config, []):
                plot_data.append({'Configuration': label, 'Attack': 'T5 Paraphrase', 'Z-Score': z})

    df_plot = pd.DataFrame(plot_data)
    
    # --- Plotting ---
    print("  Generating attack comparison plot...")
    sns.set_theme(style="ticks")
    plt.figure(figsize=(15, 7))

    # Boxplot comparing Original vs Attacks for each configuration
    sns.boxplot(
        data=df_plot, 
        x='Configuration', 
        y='Z-Score', 
        hue='Attack',
        palette='viridis', # Choose a suitable palette
        showfliers=False # Hide outliers for clarity
    )
    
    # Add detection threshold line
    z_score_threshold = norm.ppf(0.005) # p = 0.005
    plt.axhline(y=z_score_threshold, color='red', linestyle='--', label=f'Detection Threshold (p = 0.005)')

    plt.title('Watermark Detection Z-Scores Before and After Attacks')
    plt.ylabel('Z-Score (Lower indicates stronger detection)')
    plt.xlabel('Watermark Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Attack Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(PLOT_ATTACK_OUTPUT_PATH, dpi=300)
    print(f"  Attack comparison plot saved to '{PLOT_ATTACK_OUTPUT_PATH}'")
    plt.close()