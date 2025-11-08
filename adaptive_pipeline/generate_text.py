import random
import json
import itertools
import os
import numpy as np
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
parent_root = os.path.dirname(project_root)
if parent_root not in sys.path:
    sys.path.append(parent_root)

import adaptive_modules.shared as shared
from adaptive_modules.adaptive_text_generation import generate_reply

GENERATION_FILE_PATH = os.environ.get('GENERATION_FILE_PATH', 'self_sync_outputs/generation_results.json')
PROMPT_FILE = "c4_prompts.json" 
SECRET_KEY_BITS = int(os.environ.get('SECRET_KEY_BITS', 40))

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

print(f"  Loading fixed {SECRET_KEY_BITS}-bit secret key...")
if not hasattr(shared, 'SECRET_KEY_BITS'):
    FIXED_KEY = [
        1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1,
        0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1
    ]
    if len(FIXED_KEY) < SECRET_KEY_BITS:
        raise ValueError(f"Fixed key is too short! Need {SECRET_KEY_BITS} bits.")
        
    shared.SECRET_KEY_BITS = FIXED_KEY[:SECRET_KEY_BITS]

adaptive_configs = {
    'weak':   {'delta_senso': 1.0, 'delta_redgreen': 1.0},
    'medium': {'delta_senso': 2.5, 'delta_redgreen': 2.0},
    'strong': {'delta_senso': 5.0, 'delta_redgreen': 4.0}
}
print(f"  Generating full ablation study for {list(adaptive_configs.keys())} delta levels.")

configurations = {
    'baseline': {'type': 'baseline'},
    'llm_baseline': {'type': 'llm', 'delta_senso': 0.0, 'delta_redgreen': 0.0}
}

for level_name, levels in adaptive_configs.items():
    configurations.update({
        # 1. Sensorimotor-Sync Only
        f'llm_senso_only_{level_name}': { 
            'type': 'llm_watermarked', 
            'delta_senso': levels['delta_senso'], 
            'delta_redgreen': 0.0  # RedGreen OFF
        },
        # 2. RedGreen Only
        f'llm_redgreen_only_{level_name}': { 
            'type': 'llm_watermarked', 
            'delta_senso': 0.0, # Sensorimotor OFF
            'delta_redgreen': levels['delta_redgreen']
        },
        # 3. Both
        f'llm_full_adaptive_{level_name}': { 
            'type': 'llm_watermarked', 
            'delta_senso': levels['delta_senso'], 
            'delta_redgreen': levels['delta_redgreen']
        }
    })

print(f"  Total configurations to generate per sample: {len(configurations)}")

def generate_replies(sample, config, generate_params, message_bits):
    if config['type'] == 'baseline':
        return sample['baseline']
    
    shared.delta_senso = config.get('delta_senso', 0.0)
    shared.delta_redgreen = config.get('delta_redgreen', 0.0)
    
    if config['type'] == 'llm':
        reply, done = generate_reply(sample["prompt"], generate_params, message_bits=None)
        return reply

    elif config['type'] == 'llm_watermarked':
        reply, done = generate_reply(sample["prompt"], generate_params, message_bits=message_bits)
        return reply

# Standard generation parameters
generate_params = {
    'max_new_tokens': 200, 'add_bos_token': True, 'truncation_length': 4096,
    'custom_stopping_strings': ["### Human:", "Human:", "user:", "USER:", "Q:", "<|im_end|>", "<|im_start|>system"],
    'ban_eos_token': False, 'skip_special_tokens': True, 
    'do_sample': True, 
    'typical_p': 1, 'repetition_penalty': 1.0, 'encoder_repetition_penalty': 1,
    'num_beams': 1, 'penalty_alpha': 0, 'min_length': 0, 'length_penalty': 1, 
    'no_repeat_ngram_size': 0, 'early_stopping': False, 'seed': 0,
    'pad_token_id': shared.tokenizer.eos_token_id,
    'temperature': 0.7, 'top_p': 0.9, 'top_k': 50 
}

results = []
print("  Starting text generation loop (Self-Synchronizing)...")
for i, sample in enumerate(c4_random_samples, start=1):
    print(f"    Processing sample {i}/{len(c4_random_samples)}...")
    
    sample_result = {
        'prompt': sample['prompt'], 
        'baseline': sample['baseline'],
        'message_bits': shared.SECRET_KEY_BITS
    }
    
    for config_name, config in configurations.items():
        try:
            reply = generate_replies(sample, config, generate_params, shared.SECRET_KEY_BITS)
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