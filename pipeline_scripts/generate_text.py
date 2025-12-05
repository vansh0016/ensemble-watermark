import random
import json
import itertools
import modules.shared as shared
from modules.text_generation import generate_reply
import os

GENERATION_FILE_PATH = os.environ.get('GENERATION_FILE_PATH', 'outputs/generation_results.json')
PROMPT_FILE = "c4_prompts.json" 

print(f"  Loading fixed prompt dataset from '{PROMPT_FILE}'...")
try:
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        c4_random_samples = json.load(f)
    print(f"  Successfully loaded {len(c4_random_samples)} prompts.")
    
    NUM_SAMPLES = int(os.environ.get('NUM_SAMPLES', len(c4_random_samples)))
    c4_random_samples = c4_random_samples[:NUM_SAMPLES]
    print(f"  Using {len(c4_random_samples)} samples for this run.")

except FileNotFoundError:
    print(f"  ERROR: Could not find {PROMPT_FILE}. Please run 'create_prompt_dataset.py' first.")
    raise
except Exception as e:
    print(f"  ERROR loading {PROMPT_FILE}: {e}")
    raise

feature_levels = {
    'weak':   {'delta_senso': 1.0, 'delta_acro': 10.0, 'delta_redgreen': 1.0},
    'medium': {'delta_senso': 2.5, 'delta_acro': 20.0, 'delta_redgreen': 2.0},
    'strong': {'delta_senso': 5.0, 'delta_acro': 40.0, 'delta_redgreen': 4.0}
}
print(f"  Generating for {list(feature_levels.keys())} delta levels.")

configurations = {
    'baseline': {'type': 'baseline'},
    'llm_baseline': {'type': 'llm', 'delta_senso': 0.0, 'delta_acro': 0.0, 'delta_redgreen': 0.0}
}

for level_name, levels in feature_levels.items():
    configurations.update({
        f'llm_senso_{level_name}': {'type': 'llm', 'delta_senso': levels['delta_senso'], 'delta_acro': 0.0, 'delta_redgreen': 0.0},
        f'llm_acro_{level_name}': {'type': 'llm', 'delta_senso': 0.0, 'delta_acro': levels['delta_acro'], 'delta_redgreen': 0.0},
        f'llm_redgreen_{level_name}': {'type': 'llm', 'delta_senso': 0.0, 'delta_acro': 0.0, 'delta_redgreen': levels['delta_redgreen']},
        f'llm_both_{level_name}': {'type': 'llm', 'delta_senso': levels['delta_senso'], 'delta_acro': levels['delta_acro'], 'delta_redgreen': 0.0},
        f'llm_senso_redgreen_{level_name}': {'type': 'llm', 'delta_senso': levels['delta_senso'], 'delta_acro': 0.0, 'delta_redgreen': levels['delta_redgreen']},
        f'llm_acro_redgreen_{level_name}': {'type': 'llm', 'delta_senso': 0.0, 'delta_acro': levels['delta_acro'], 'delta_redgreen': levels['delta_redgreen']},
        f'llm_all_three_{level_name}': {'type': 'llm', 'delta_senso': levels['delta_senso'], 'delta_acro': levels['delta_acro'], 'delta_redgreen': levels['delta_redgreen']}
    })

print(f"  Total configurations to generate per sample: {len(configurations)}")

def generate_replies(sample, config, generate_params):
    if config['type'] == 'baseline':
        return sample['baseline']
    elif config['type'] == 'llm':
        shared.delta_senso = config.get('delta_senso', 0.0)
        shared.delta_acro = config.get('delta_acro', 0.0)
        shared.delta_redgreen = config.get('delta_redgreen', 0.0)
        shared.secret_key = [0, 0]
        shared.new_sentence = False
        
        question = f'''<|begin_of_text|> {sample["prompt"]}'''
        reply, done = generate_reply(question, generate_params, eos_token='<|end_of_text|>')
        return reply

generate_params = {
    'max_new_tokens': 200, 'add_bos_token': False, 'truncation_length': 4096,
    'custom_stopping_strings': ["### Human:", "Human:", "user:", "USER:", "Q:", "<|im_end|>", "<|im_start|>system"],
    'ban_eos_token': False, 'skip_special_tokens': True, 'do_sample': False,
    'typical_p': 1, 'repetition_penalty': 1.0, 'encoder_repetition_penalty': 1,
    'num_beams': 1, 'penalty_alpha': 0, 'min_length': 0, 'length_penalty': 1, 
    'no_repeat_ngram_size': 0, 'early_stopping': False, 'seed': 0,
    'pad_token_id': shared.tokenizer.eos_token_id
}

results = []
print("  Starting text generation loop (all delta levels)...")
for i, sample in enumerate(c4_random_samples, start=1):
    print(f"    Processing sample {i}/{len(c4_random_samples)}...")
    sample_result = {'prompt': sample['prompt'], 'baseline': sample['baseline']}
    
    for config_name, config in configurations.items():
        try:
            reply = generate_replies(sample, config, generate_params)
            sample_result[config_name] = reply
        except Exception as e:
            print(f"    ERROR generating for {config_name}: {e}")
            sample_result[config_name] = None
    
    results.append(sample_result)

print(f"  Generation complete. Saving {len(results)} results to '{GENERATION_FILE_PATH}'...")
try:
    with open(GENERATION_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("  Save complete.")
except Exception as e:
    print(f"  ERROR saving results: {e}")