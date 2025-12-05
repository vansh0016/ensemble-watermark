import os
import sys
import time
import torch
import gc

CONFIG = {
    "SCRIPT_DIR": "adaptive_pipeline",
    "OUTPUT_DIR": "self_sync_outputs_Llama-3B",
    # "OUTPUT_DIR": "self_sync_outputs",
    # "OUTPUT_DIR": "self_sync_outputs_Mistral7B",
    "NUM_SAMPLES": 50, 
    "SECRET_KEY_BITS": 40,
    "MODEL_NAME": "Llama-3.2-3B-Instruct",
    # "MODEL_NAME": "Llama-3.2-8B",
    # "MODEL_NAME": "Mistral-7B",
    
    "T5_REPLACE_PERCENTAGE": 0.15,
    "T5_BATCH_SIZE": 4,
    
    "PROMPT_FILE": "c4_prompts.json",
    "GENERATION_FILE": "generation_results.json",
    "DETECTION_ORIGINAL_FILE": "detection_results.json",
    "ATTACK_TRANSLATION_FILE": "attacked_translation.json",
    "ATTACK_T5_FILE": "attacked_t5.json",
    "DETECTION_ATTACK_TRANSLATION_FILE": "detection_results_attack_translation.json",
    "DETECTION_ATTACK_T5_FILE": "detection_results_attack_t5.json",
    
    "PLOT_TRADEOFF_SCATTER_FILE": "zscore_vs_perplexity_scatter.png",
    "PLOT_ROBUSTNESS_BAR_FILE": "robustness_zscore_barplot.png",
    "PLOT_STEP_PLOT_FILE": "zscore_vs_tokens_stepplot.png"
}

def set_environment_variables(config):
    """Set all environment variables for all sub-scripts."""
    print("  Setting environment variables for SELF-SYNCHRONIZING pipeline...")
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)
    
    gen_path = os.path.join(config["OUTPUT_DIR"], config["GENERATION_FILE"])
    det_orig_path = os.path.join(config["OUTPUT_DIR"], config["DETECTION_ORIGINAL_FILE"])
    attack_trans_path = os.path.join(config["OUTPUT_DIR"], config["ATTACK_TRANSLATION_FILE"])
    attack_t5_path = os.path.join(config["OUTPUT_DIR"], config["ATTACK_T5_FILE"])
    det_trans_path = os.path.join(config["OUTPUT_DIR"], config["DETECTION_ATTACK_TRANSLATION_FILE"])
    det_t5_path = os.path.join(config["OUTPUT_DIR"], config["DETECTION_ATTACK_T5_FILE"])

    os.environ['NUM_SAMPLES'] = str(config["NUM_SAMPLES"])
    os.environ['SECRET_KEY_BITS'] = str(config["SECRET_KEY_BITS"])
    os.environ['MODEL_NAME'] = config["MODEL_NAME"]
    os.environ['PROMPT_FILE'] = config["PROMPT_FILE"]
    
    os.environ['GENERATION_FILE_PATH'] = gen_path
    os.environ['DETECTION_FILE_PATH'] = det_orig_path
    
    os.environ['ATTACK_TRANSLATION_OUTPUT_FILE'] = attack_trans_path
    os.environ['ATTACK_T5_OUTPUT_FILE'] = attack_t5_path
    os.environ['T5_REPLACE_PERCENTAGE'] = str(config["T5_REPLACE_PERCENTAGE"])
    os.environ['T5_BATCH_SIZE'] = str(config["T5_BATCH_SIZE"])
    
    os.environ['DETECTION_ATTACK_TRANSLATION_FILE'] = det_trans_path
    os.environ['DETECTION_ATTACK_T5_FILE'] = det_t5_path
    
    os.environ['DETECTION_ORIGINAL_FILE'] = det_orig_path 
    os.environ['PLOT_TRADEOFF_SCATTER_PATH'] = os.path.join(config["OUTPUT_DIR"], config["PLOT_TRADEOFF_SCATTER_FILE"])
    os.environ['PLOT_ROBUSTNESS_BAR_PATH'] = os.path.join(config["OUTPUT_DIR"], config["PLOT_ROBUSTNESS_BAR_FILE"])
    os.environ['PLOT_STEP_PLOT_PATH'] = os.path.join(config["OUTPUT_DIR"], config["PLOT_STEP_PLOT_FILE"])
    
    print("  Environment variables set.")
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
    print("== Starting SELF-SYNCHRONIZING Pipeline ==")
    print("="*60)
    
    start_total_time = time.time()
    
    print("\n[PHASE 0: CONFIGURATION]")
    paths = set_environment_variables(CONFIG)
    gen_path, det_orig_path, attack_trans_path, attack_t5_path, det_trans_path, det_t5_path = paths
    
    print("\n[PHASE 1: SETUP & LOAD MODEL]")
    t_start = time.time()
    run_script("setup_and_load.py", CONFIG["SCRIPT_DIR"])
    print(f"--- Phase 1 complete in {time.time() - t_start:.2f}s ---")

    print("\n[PHASE 2: GENERATE TEXT]")
    t_start = time.time()
    run_script("generate_text.py", CONFIG["SCRIPT_DIR"])
    print(f"--- Phase 2 complete in {time.time() - t_start:.2f}s ---")

    print("\n[PHASE 3: DETECT WATERMARKS (ORIGINAL)]")
    t_start = time.time()
    run_script("detect_watermarks.py", CONFIG["SCRIPT_DIR"])
    print(f"--- Phase 3 complete in {time.time() - t_start:.2f}s ---")

    print("\n[PHASE 4: RUN ATTACKS]")
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
    
    print("\n[PHASE 5: DETECT WATERMARKS (ATTACKED)]")
    if not os.path.exists(attack_trans_path):
        print("  Skipping detection (Translation): File not found.")
    else:
        print("  Running detection on Translated text...")
        t_start = time.time()
        os.environ['ATTACKED_TEXT_FILE_PATH'] = attack_trans_path
        os.environ['DETECTION_ATTACK_OUTPUT_FILE'] = det_trans_path
        run_script("detect_attacked_text.py", CONFIG["SCRIPT_DIR"])
        print(f"  Translation detection complete in {time.time() - t_start:.2f}s")
        
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

    print("\n[PHASE 6: GENERATE PLOTS]")
    t_start = time.time()
    if not os.path.exists(det_orig_path):
        print(f"  Skipping plots: File '{det_orig_path}' not found.")
    else:
        run_script("generate_plots.py", CONFIG["SCRIPT_DIR"])
        print(f"--- Phase 6 complete in {time.time() - t_start:.2f}s ---")
        
    print("\n[PHASE 7: CLEANUP]")
    try:
        import adaptive_modules.shared as shared
        if hasattr(shared, 'model'): del shared.model
        if hasattr(shared, 'tokenizer'): del shared.tokenizer
        if hasattr(shared, 'nlp'): del shared.nlp
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("  Model, tokenizer, and CUDA cache cleared.")
    except Exception as e: print(f"  Error during cleanup: {e}")

    print("\n" + "="*60)
    print("== SELF-SYNCHRONIZING Pipeline Finished Successfully ==")
    print(f"  Total execution time: {time.time() - start_total_time:.2f}s")
    print(f"  All outputs saved in '{CONFIG['OUTPUT_DIR']}'")
    print("="*60)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    main()