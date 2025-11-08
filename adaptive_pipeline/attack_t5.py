import json
import re
import random
import math
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import os
import adaptive_modules.shared as shared

# Input file
GENERATION_FILE_PATH = os.environ.get('GENERATION_FILE_PATH', 'adaptive_outputs/generation_results.json')
# Output file
ATTACK_T5_OUTPUT_FILE = os.environ.get('ATTACK_T5_OUTPUT_FILE', 'adaptive_outputs/attacked_t5.json')
REPLACE_PERCENTAGE = float(os.environ.get('T5_REPLACE_PERCENTAGE', 0.10))
BATCH_SIZE = int(os.environ.get('T5_BATCH_SIZE', 8))
MAX_ITER_MULTIPLIER = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  T5 Attack using device: {device}")

print("  Loading T5-base model and tokenizer...")
try:
    tokenizer_t5 = T5Tokenizer.from_pretrained('t5-base')
    model_t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
    model_t5.to(device)
    model_t5.eval()
    print("  T5 model loaded.")
except Exception as e:
    print(f"  ERROR loading T5 model: {e}. Stopping T5 attack.")
    with open(ATTACK_T5_OUTPUT_FILE, 'w') as f: json.dump([], f)
    exit()

# Function to reconstruct text from words & handle punctuation
def reconstruct_text(words):
    text = ''
    for i, word in enumerate(words):
        if i > 0:
            prev_word = words[i-1]
            if word and prev_word and \
               word[0].isalnum() and prev_word[-1].isalnum():
                text += ' '
            elif word and word[0] in '([{"\'' and prev_word[-1].isalnum():
                 text += ' '
        text += word
    return text.strip()

print(f"  Loading generated text from '{GENERATION_FILE_PATH}' for T5 attack...")
try:
    with open(GENERATION_FILE_PATH, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)
    print(f"  Loaded {len(generated_data)} samples.")
except FileNotFoundError:
    print(f"  ERROR: Cannot find '{GENERATION_FILE_PATH}'. Stopping T5 attack.")
    generated_data = []
except json.JSONDecodeError:
    print(f"  ERROR: Cannot decode JSON from '{GENERATION_FILE_PATH}'.")
    generated_data = []

processed_data = []
print(f"  Starting T5 paraphrase attack (replace approx {REPLACE_PERCENTAGE*100:.0f}%)...")

keys_to_process = []
if generated_data:
    keys_to_process = [k for k in generated_data[0].keys() if k not in ['prompt', 'message_bits']]

for sample_idx, sample in enumerate(tqdm(generated_data, desc="  T5 Paraphrasing Samples")):
    new_sample = sample.copy()
    
    for field in keys_to_process:
        if field in sample and sample[field]:
            text = sample[field]
            words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
            T = len(words)
            if T == 0: continue

            epsilon_T = max(1, math.ceil(REPLACE_PERCENTAGE * T))
            successful_replacements = 0
            max_iterations = max(MAX_ITER_MULTIPLIER * epsilon_T, T*2)
            
            processed_indices = set()
            available_indices = list(range(T))
            random.shuffle(available_indices)
            
            current_iter = 0
            
            # Process in batches
            for batch_start in range(0, len(available_indices), BATCH_SIZE):
                if successful_replacements >= epsilon_T: break
                
                batch_indices = available_indices[batch_start : batch_start + BATCH_SIZE]
                batch_input_texts = []
                batch_word_indices_map = {}

                for i, word_idx in enumerate(batch_indices):
                    if word_idx in processed_indices or current_iter >= max_iterations: continue
                    current_iter+=1

                    original_word = words[word_idx]
                    # Skip punctuation or very short words
                    if not original_word or not original_word[0].isalnum() or len(original_word) < 2:
                        processed_indices.add(word_idx)
                        continue
                        
                    words_masked = words[:word_idx] + ['<extra_id_0>'] + words[word_idx+1:]
                    masked_text = reconstruct_text(words_masked)
                    
                    input_text = f"fill mask: {masked_text}" 
                    batch_input_texts.append(input_text)
                    batch_word_indices_map[len(batch_input_texts)-1] = word_idx
                    processed_indices.add(word_idx)

                if not batch_input_texts: continue

                # Generate replacements
                inputs = tokenizer_t5(batch_input_texts, return_tensors='pt', padding=True, truncation=True).to(device)
                
                with torch.no_grad():
                    outputs = model_t5.generate(
                        **inputs,
                        max_length=10,
                        num_beams=5,
                        num_return_sequences=3,
                        early_stopping=True
                    )

                decoded_outputs = tokenizer_t5.batch_decode(outputs, skip_special_tokens=True)
                
                output_idx = 0
                for i in range(len(batch_input_texts)):
                    if i not in batch_word_indices_map: continue

                    word_idx = batch_word_indices_map[i]
                    original_word = words[word_idx]
                    
                    found_replacement = False
                    for k in range(3):
                        replacement = decoded_outputs[output_idx + k].strip()
                        if replacement and replacement.lower() != original_word.lower() and ' ' not in replacement:
                            words[word_idx] = replacement
                            successful_replacements += 1
                            found_replacement = True
                            break
                    output_idx += 3
                    if successful_replacements >= epsilon_T: break

            final_text = reconstruct_text(words)
            new_sample[field] = final_text

    processed_data.append(new_sample)

print(f"\nT5 attack complete. Saving {len(processed_data)} results to '{ATTACK_T5_OUTPUT_FILE}'...")
try:
    with open(ATTACK_T5_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print("  Save complete.")
except Exception as e:
    print(f"  ERROR saving T5 attack results: {e}")

del model_t5
del tokenizer_t5
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()