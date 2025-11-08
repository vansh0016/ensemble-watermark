import json
import numpy as np
import os
from scipy.stats import norm

# p < 0.05 threshold
DETECTION_THRESHOLD_Z = norm.ppf(0.05)

def load_and_process_data(filepath, config_mapping):
    """Loads a detection JSON and calculates median Z-Score, PPLX, and Detection Rate."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found {filepath}\n")
        return None

    # Store lists of scores for each config
    results_raw = {key: {'zscores': [], 'pplx_increases': []} for key in config_mapping.keys()}
    
    for sample in data:
        detections = sample.get('detection', {})
        
        # Get the baseline perplexity for this sample
        try:
            baseline_pplx = float(detections.get('llm_baseline', {}).get('perplexity', np.nan))
        except TypeError:
            baseline_pplx = np.nan
        
        for config_key in config_mapping.keys():
            if config_key not in detections:
                continue
                
            scores = detections[config_key]
            
            # 1. Get Z-Score
            if 'zscore_at_token' in scores:
                z_list = scores['zscore_at_token']
                if z_list:
                    final_z = z_list[-1]
                    results_raw[config_key]['zscores'].append(final_z)
            
            # 2. Get Perplexity Increase
            try:
                current_pplx = float(scores.get('perplexity', np.nan))
                if not np.isnan(baseline_pplx) and not np.isnan(current_pplx):
                    pplx_increase = current_pplx - baseline_pplx
                    results_raw[config_key]['pplx_increases'].append(pplx_increase)
            except TypeError:
                continue

    # Calculate final metrics
    final_metrics = {}
    for key, label in config_mapping.items():
        zscores = results_raw[key]['zscores']
        pplx_increases = results_raw[key]['pplx_increases']
        
        if not zscores:
            final_metrics[label] = {'zscore': 'N/A', 'pplx_inc': 'N/A', 'det_rate': 'N/A'}
            continue
            
        median_z = np.median(zscores)
        median_pplx_inc = np.median(pplx_increases) if pplx_increases else 0.0
        
        # Detection Rate = % of samples that passed the p < 0.05 threshold
        detected_count = sum(1 for z in zscores if z < DETECTION_THRESHOLD_Z)
        det_rate = (detected_count / len(zscores)) * 100
        
        final_metrics[label] = {
            'zscore': f"{median_z:.2f}",
            'pplx_inc': f"{median_pplx_inc:+.2f}",
            'det_rate': f"{det_rate:.1f}%"
        }
        
    return final_metrics

def print_table(title, metrics, labels):
    """Prints a formatted Markdown table."""
    print(f"\n### {title}")
    print("| Configuration | Median Z-Score (↓) | Median PPLX ↑ (↓) | Detection Rate (↑) |")
    print("| :--- | :---: | :---: | :---: |")
    
    for label in labels:
        if label in metrics:
            m = metrics[label]
            print(f"| **{label}** | {m['zscore']} | {m['pplx_inc']} | {m['det_rate']} |")

print("="*60)
print("Ablation Study Results")
print("="*60)

baseline_file = 'outputs/detection_results.json'
baseline_configs = {
    "llm_baseline": "Baseline (LLM)",
    # Weak
    "llm_senso_weak": "Senso (S) - Weak",
    "llm_acro_weak": "Acro (A) - Weak",
    "llm_redgreen_weak": "RedGreen (RG) - Weak",
    "llm_all_three_weak": "Full Ensemble (S+A+RG) - Weak",
    # Medium
    "llm_senso_medium": "Senso (S) - Medium",
    "llm_acro_medium": "Acro (A) - Medium",
    "llm_redgreen_medium": "RedGreen (RG) - Medium",
    "llm_all_three_medium": "Full Ensemble (S+A+RG) - Medium",
    # Strong
    "llm_senso_strong": "Senso (S) - Strong",
    "llm_acro_strong": "Acro (A) - Strong",
    "llm_redgreen_strong": "RedGreen (RG) - Strong",
    "llm_all_three_strong": "Full Ensemble (S+A+RG) - Strong",
}
baseline_labels_order = [
    "Baseline (LLM)", 
    "Senso (S) - Weak", "Acro (A) - Weak", "RedGreen (RG) - Weak", "Full Ensemble (S+A+RG) - Weak",
    "Senso (S) - Medium", "Acro (A) - Medium", "RedGreen (RG) - Medium", "Full Ensemble (S+A+RG) - Medium",
    "Senso (S) - Strong", "Acro (A) - Strong", "RedGreen (RG) - Strong", "Full Ensemble (S+A+RG) - Strong",
]
baseline_metrics = load_and_process_data(baseline_file, baseline_configs)
if baseline_metrics:
    print_table("Baseline Ensemble Pipeline (All Deltas)", baseline_metrics, baseline_labels_order)

adaptive_file = 'self_sync_outputs/detection_results.json'

adaptive_configs_weak = {
    "llm_baseline": "Baseline (LLM)",
    "llm_senso_only_weak": "Senso-Sync (S)",
    "llm_redgreen_only_weak": "RedGreen (RG)",
    "llm_full_adaptive_weak": "Full Adaptive (S+RG)"
}
adaptive_labels_weak = [
    "Baseline (LLM)", "Senso-Sync (S)", "RedGreen (RG)", "Full Adaptive (S+RG)"
]
adaptive_metrics_weak = load_and_process_data(adaptive_file, adaptive_configs_weak)
if adaptive_metrics_weak:
    print_table("Self-Synchronizing Pipeline (WEAK Delta)", adaptive_metrics_weak, adaptive_labels_weak)

adaptive_configs_medium = {
    "llm_baseline": "Baseline (LLM)",
    "llm_senso_only_medium": "Senso-Sync (S)",
    "llm_redgreen_only_medium": "RedGreen (RG)",
    "llm_full_adaptive_medium": "Full Adaptive (S+RG)"
}
adaptive_labels_medium = [
    "Baseline (LLM)", "Senso-Sync (S)", "RedGreen (RG)", "Full Adaptive (S+RG)"
]
adaptive_metrics_medium = load_and_process_data(adaptive_file, adaptive_configs_medium)
if adaptive_metrics_medium:
    print_table("Self-Synchronizing Pipeline (MEDIUM Delta)", adaptive_metrics_medium, adaptive_labels_medium)

adaptive_configs_strong = {
    "llm_baseline": "Baseline (LLM)",
    "llm_senso_only_strong": "Senso-Sync (S)",
    "llm_redgreen_only_strong": "RedGreen (RG)",
    "llm_full_adaptive_strong": "Full Adaptive (S+RG)"
}
adaptive_labels_strong = [
    "Baseline (LLM)", "Senso-Sync (S)", "RedGreen (RG)", "Full Adaptive (S+RG)"
]
adaptive_metrics_strong = load_and_process_data(adaptive_file, adaptive_configs_strong)
if adaptive_metrics_strong:
    print_table("Self-Synchronizing Pipeline (STRONG Delta)", adaptive_metrics_strong, adaptive_labels_strong)