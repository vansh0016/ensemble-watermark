import os
import json
import hashlib
import math
import pandas as pd
import torch
import sys
import numpy as np
from scipy.stats import binom, norm

project_root = os.path.dirname(os.path.abspath(__file__))
parent_root = os.path.dirname(project_root)
if parent_root not in sys.path:
    sys.path.append(parent_root)

import adaptive_modules.shared as shared

GENERATION_FILE_PATH = os.environ.get('GENERATION_FILE_PATH', 'self_sync_outputs/generation_results.json')
DETECTION_FILE_PATH = os.environ.get('DETECTION_FILE_PATH', 'self_sync_outputs/detection_results.json')
ZSCORE_INTERVALS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Self-Sync Detection script using device: {device}")

df_sensorimotor = shared.sensorimotor
classes_list = shared.classes 
tokenizer = shared.tokenizer
model = shared.model

try:
    GROUP_0_MASK = shared.GROUP_0_MASK
    GROUP_1_MASK = shared.GROUP_1_MASK
except AttributeError:
    print("  FATAL: Vocab masks not found in shared module. Run setup_and_load.py.")
    sys.exit(1)


def calculate_perplexity(text, model, tokenizer):
    """Calculates the perplexity of a given text."""
    if not text or not model or not tokenizer:
        return 0.0
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
        perplexity = torch.exp(loss).item()
        if math.isinf(perplexity) or math.isnan(perplexity): return 0.0 
        return perplexity
    except Exception:
        return 0.0

def token_hash(token_id):
    """Simple token hash function for bit selection."""
    return int(hashlib.sha256(str(token_id).encode('utf-8')).hexdigest(), 16)

def calculate_z_score(reply, original_message_bits):
    """
    Calculates Z-score (not BER) using token-hash synchronization.
    Returns a list of Z-scores at intervals and a final perplexity.
    """
    if not reply or not original_message_bits:
        return [0.0], 0.0

    perplexity = calculate_perplexity(reply, model, tokenizer)
    tokens = tokenizer.encode(reply, add_special_tokens=False)
    message_len = len(original_message_bits)
    
    z_score_at_token = []
    hits = []
    num_opportunities = 0

    for i in range(1, len(tokens)):
        last_token = tokens[i-1]
        current_token = tokens[i]
        
        is_group_0 = GROUP_0_MASK[current_token].item()
        is_group_1 = GROUP_1_MASK[current_token].item()
        
        if not is_group_0 and not is_group_1:
            continue
            
        num_opportunities += 1
        bit_index = token_hash(last_token) % message_len
        expected_bit = original_message_bits[bit_index]
        is_hit = (expected_bit == 0 and is_group_0) or (expected_bit == 1 and is_group_1)
        hits.append(is_hit)
        
        if num_opportunities > 0 and num_opportunities % ZSCORE_INTERVALS == 0:
            num_hits = sum(hits)
            p_value = binom.sf(num_hits - 1, num_opportunities, 0.5)
            z_score = norm.ppf(p_value)
            z_score_at_token.append(float(z_score) if not math.isnan(z_score) else 0.0)

    if num_opportunities > 0:
        num_hits = sum(hits)
        p_value = binom.sf(num_hits - 1, num_opportunities, 0.5)
        z_score = norm.ppf(p_value)
        z_score_at_token.append(float(z_score) if not math.isnan(z_score) else 0.0)
    
    if not z_score_at_token:
        return [0.0], perplexity

    return z_score_at_token, perplexity

print(f"  Loading generated text from '{GENERATION_FILE_PATH}'...")
try:
    with open(GENERATION_FILE_PATH, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)
    print(f"  Successfully loaded {len(generated_data)} samples.")
except Exception as e:
    print(f"  ERROR: Cannot find '{GENERATION_FILE_PATH}'. {e}")
    sys.exit(1)

detection_results = []
print("  Starting Z-Score & perplexity loop...")

for idx, sample in enumerate(generated_data, start=1):
    prompt = sample.get('prompt', '')
    original_message_bits = sample.get('message_bits', [])
    print(f"    Detecting sample {idx}/{len(generated_data)}...")
    
    sample_detection = {'prompt': prompt, 'message_bits': original_message_bits, 'detection': {}}
    
    for key, text in sample.items():
        if key in ['prompt', 'message_bits']:
            continue
        
        try:
            z_score_list, perplexity = calculate_z_score(text, original_message_bits)
            
            sample_detection['detection'][key] = {
                'zscore_at_token': z_score_list,
                'perplexity': perplexity,
            }
        except Exception as e:
            print(f"      ERROR detecting {key}: {e}")
            sample_detection['detection'][key] = {'error': str(e)}

    detection_results.append(sample_detection)

print(f"  Detection complete. Saving {len(detection_results)} results to '{DETECTION_FILE_PATH}'...")
try:
    with open(DETECTION_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(detection_results, f, ensure_ascii=False, indent=4)
    print("  Save complete.")
except Exception as e:
    print(f"  ERROR saving detection results: {e}")