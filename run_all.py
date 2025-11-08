import os
import sys
import time
import torch
import gc

# --- Configuration ---
# This is the central configuration for your entire pipeline
CONFIG = {
    "SCRIPT_DIR": "pipeline_scripts",
    "OUTPUT_DIR": "outputs",
    "NUM_SAMPLES": 5, # <-- Set to 400 for full run, 5-10 for testing
    
    # --- T5 Attack Parameters ---
    "T5_REPLACE_PERCENTAGE": 0.15,
    "T5_BATCH_SIZE": 4,
    
    # --- File Names ---
    # These names are now centralized
    "PROMPT_FILE": "c4_prompts.json",
    
    # Phase 2 Output
    "GENERATION_FILE": "generation_results.json",
    # Phase 3 Output
    "DETECTION_ORIGINAL_FILE": "detection_results.json",
    
    # Phase 4 Outputs
    "ATTACK_TRANSLATION_FILE": "attacked_translation.json",
    "ATTACK_T5_FILE": "attacked_t5.json",
    
    # Phase 5 Outputs
    "DETECTION_ATTACK_TRANSLATION_FILE": "detection_results_attack_translation.json",
    "DETECTION_ATTACK_T5_FILE": "detection_results_attack_t5.json",
    
    # Phase 6 Plot Outputs (Main Results)
    "PLOT_BOXPLOT_PREFIX": "results_boxplot", # Will add _weak.png, _medium.png, etc.
    "PLOT_HEATMAP_FILE": "config_comparison_heatmap.png",
    "PLOT_DELTA_SCATTER_FILE": "delta_tradeoff_scatter.png",
    "PLOT_STEP_PLOT_FILE": "detection_rate_vs_sentences.png",
    
    # Phase 7 Plot Outputs (Attack Analysis)
    "PLOT_ATTACK_SIGNIFICANCE_FILE": "attack_significance_heatmap.png",
    "PLOT_ATTACK_TRADEOFF_FILE": "attack_tradeoff_scatter.png"
}

def set_environment_variables(config):
    """Set all environment variables for all sub-scripts."""
    print("  Setting environment variables...")
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)
    
    # --- File Paths ---
    # Note: os.path.join is safer for constructing paths
    gen_path = os.path.join(config["OUTPUT_DIR"], config["GENERATION_FILE"])
    det_orig_path = os.path.join(config["OUTPUT_DIR"], config["DETECTION_ORIGINAL_FILE"])
    attack_trans_path = os.path.join(config["OUTPUT_DIR"], config["ATTACK_TRANSLATION_FILE"])
    attack_t5_path = os.path.join(config["OUTPUT_DIR"], config["ATTACK_T5_FILE"])
    det_trans_path = os.path.join(config["OUTPUT_DIR"], config["DETECTION_ATTACK_TRANSLATION_FILE"])
    det_t5_path = os.path.join(config["OUTPUT_DIR"], config["DETECTION_ATTACK_T5_FILE"])

    # --- Set Variables ---
    os.environ['NUM_SAMPLES'] = str(config["NUM_SAMPLES"])
    os.environ['PROMPT_FILE'] = config["PROMPT_FILE"] # Used by generate_text
    
    # Generation/Detection
    os.environ['GENERATION_FILE_PATH'] = gen_path
    os.environ['DETECTION_FILE_PATH'] = det_orig_path # Used by detect_watermarks & generate_all_plots
    
    # Attacks
    os.environ['ATTACK_TRANSLATION_OUTPUT_FILE'] = attack_trans_path
    os.environ['ATTACK_T5_OUTPUT_FILE'] = attack_t5_path
    os.environ['T5_REPLACE_PERCENTAGE'] = str(config["T5_REPLACE_PERCENTAGE"])
    os.environ['T5_BATCH_SIZE'] = str(config["T5_BATCH_SIZE"])
    
    # Post-Attack Detection
    # Note: detect_attacked_text.py reads ATTACKED_TEXT_FILE_PATH and DETECTION_ATTACK_OUTPUT_FILE
    # We will set these specifically in Phase 5
    
    # Plotting (Main)
    os.environ['PLOT_BOXPLOT_PATH_PREFIX'] = os.path.join(config["OUTPUT_DIR"], config["PLOT_BOXPLOT_PREFIX"])
    os.environ['PLOT_HEATMAP_PATH'] = os.path.join(config["OUTPUT_DIR"], config["PLOT_HEATMAP_FILE"])
    os.environ['PLOT_SCATTER_PATH'] = os.path.join(config["OUTPUT_DIR"], config["PLOT_DELTA_SCATTER_FILE"])
    
    # Plotting (Attack Analysis)
    # analyze_attack_effects.py needs all 3 detection files
    os.environ['DETECTION_ORIGINAL_FILE'] = det_orig_path # Renamed for clarity in script
    os.environ['DETECTION_ATTACK_TRANSLATION_FILE'] = det_trans_path
    os.environ['DETECTION_ATTACK_T5_FILE'] = det_t5_path
    os.environ['PLOT_SIGNIFICANCE_OUTPUT_PATH'] = os.path.join(config["OUTPUT_DIR"], config["PLOT_ATTACK_SIGNIFICANCE_FILE"])
    os.environ['PLOT_TRADEOFF_SCATTER_PATH'] = os.path.join(config["OUTPUT_DIR"], config["PLOT_ATTACK_TRADEOFF_FILE"])
    os.environ['PLOT_STEP_PLOT_PATH'] = os.path.join(config["OUTPUT_DIR"], config["PLOT_STEP_PLOT_FILE"])
    
    print("  Environment variables set.")
    # Return key paths for checking
    return gen_path, det_orig_path, attack_trans_path, attack_t5_path, det_trans_path, det_t5_path

def run_script(script_name, script_dir):
    """Executes a script from the script directory."""
    script_path = os.path.join(script_dir, script_name)
    try:
        with open(script_path, 'r') as f: script_content = f.read()
        compiled_code = compile(script_content, script_path, 'exec')
        exec(compiled_code, globals(), globals())
    except FileNotFoundError: print(f"  ERROR: Script not found: {script_path}"); raise
    except Exception as e: print(f"  ERROR executing {script_name}: {e}"); raise

def main():
    print("="*60)
    print("== Starting FULL Pipeline: Generate, Detect, Attack, Analyze ==")
    print("="*60)
    
    start_total_time = time.time()
    
    # --- 0. Setup Environment ---
    print("\n[PHASE 0: CONFIGURATION]")
    paths = set_environment_variables(CONFIG)
    gen_path, det_orig_path, attack_trans_path, attack_t5_path, det_trans_path, det_t5_path = paths
    print(f"  Running with {CONFIG['NUM_SAMPLES']} samples.")
    print(f"  Input Prompts: '{CONFIG['PROMPT_FILE']}'")

    # --- 1. Load Model and Data ---
    print("\n[PHASE 1: SETUP & LOAD MODEL]")
    t_start = time.time()
    run_script("setup_and_load.py", CONFIG["SCRIPT_DIR"])
    print(f"--- Phase 1 complete in {time.time() - t_start:.2f}s ---")

    # --- 2. Generate Text (All Delta Levels) ---
    print("\n[PHASE 2: GENERATE TEXT]")
    t_start = time.time()
    run_script("generate_text.py", CONFIG["SCRIPT_DIR"])
    print(f"--- Phase 2 complete in {time.time() - t_start:.2f}s ---")

    # --- 3. Detect Watermarks (Original Text) ---
    print("\n[PHASE 3: DETECT WATERMARKS (ORIGINAL)]")
    t_start = time.time()
    run_script("detect_watermarks.py", CONFIG["SCRIPT_DIR"])
    print(f"--- Phase 3 complete in {time.time() - t_start:.2f}s ---")

    # --- 4. Run Attacks ---
    print("\n[PHASE 4: RUN ATTACKS]")
    # Check if generation file exists before attacking
    if not os.path.exists(gen_path):
        print(f"  ERROR: Generation file '{gen_path}' not found. Skipping all attacks.")
    else:
        print("  Running Translation Attack...")
        t_start_trans = time.time()
        run_script("attack_translation.py", CONFIG["SCRIPT_DIR"])
        print(f"  Translation attack complete in {time.time() - t_start_trans:.2f}s")
        
        print("  Running T5 Paraphrase Attack...")
        t_start_t5 = time.time()
        run_script("attack_t5.py", CONFIG["SCRIPT_DIR"])
        print(f"  T5 attack complete in {time.time() - t_start_t5:.2f}s")
    print(f"--- Phase 4 complete ---")
    
    # --- 5. Detect Watermarks (Attacked Text) ---
    print("\n[PHASE 5: DETECT WATERMARKS (ATTACKED)]")
    # Detect Translation Attack
    if not os.path.exists(attack_trans_path):
        print("  Skipping detection (Translation): File not found.")
    else:
        print("  Running detection on Translated text...")
        t_start = time.time()
        os.environ['ATTACKED_TEXT_FILE_PATH'] = attack_trans_path
        os.environ['DETECTION_ATTACK_OUTPUT_FILE'] = det_trans_path
        run_script("detect_attacked_text.py", CONFIG["SCRIPT_DIR"])
        print(f"  Translation detection complete in {time.time() - t_start:.2f}s")
        
    # Detect T5 Attack
    if not os.path.exists(attack_t5_path):
        print("  Skipping detection (T5): File not found.")
    else:
        print("  Running detection on T5 Paraphrased text...")
        t_start = time.time()
        os.environ['ATTACKED_TEXT_FILE_PATH'] = attack_t5_path
        os.environ['DETECTION_ATTACK_OUTPUT_FILE'] = det_t5_path
        run_script("detect_attacked_text.py", CONFIG["SCRIPT_DIR"])
        print(f"  T5 detection complete in {time.time() - t_start:.2f}s")
    print(f"--- Phase 5 complete ---")

    # --- 6. Generate Main Plots ---
    print("\n[PHASE 6: GENERATE MAIN PLOTS]")
    t_start = time.time()
    if not os.path.exists(det_orig_path):
        print(f"  Skipping main plots: File '{det_orig_path}' not found.")
    else:
        run_script("generate_all_plots.py", CONFIG["SCRIPT_DIR"])
        print(f"--- Phase 6 complete in {time.time() - t_start:.2f}s ---")
        
    # --- 7. Generate Attack Analysis Plots ---
    print("\n[PHASE 7: GENERATE ATTACK ANALYSIS PLOTS]")
    t_start = time.time()
    if not (os.path.exists(det_orig_path) and (os.path.exists(det_trans_path) or os.path.exists(det_t5_path))):
        print(f"  Skipping attack plots: Missing one or more required detection files.")
    else:
        run_script("analyze_attack_effects.py", CONFIG["SCRIPT_DIR"])
        print(f"--- Phase 7 complete in {time.time() - t_start:.2f}s ---")

    # --- 8. Cleanup ---
    print("\n[PHASE 8: CLEANUP]")
    try:
        import modules.shared as shared
        if hasattr(shared, 'model'): del shared.model
        if hasattr(shared, 'tokenizer'): del shared.tokenizer
        if hasattr(shared, 'nlp'): del shared.nlp
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("  Model, tokenizer, and CUDA cache cleared.")
    except Exception as e: print(f"  Error during cleanup: {e}")

    print("\n" + "="*60)
    print("== FULL Pipeline Finished Successfully ==")
    print(f"  Total execution time: {time.time() - start_total_time:.2f}s")
    print(f"  All outputs saved in '{CONFIG['OUTPUT_DIR']}'")
    print("="*60)

if __name__ == "__main__":
    # Ensure the script's directory is the CWD
    # This makes all relative paths (like 'modules/') work
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    main()