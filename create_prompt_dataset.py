import json
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import islice

TARGET_SAMPLES = 400
NUM_TOKENS_TRIMMED = 200
TOKENIZER_NAME = "gpt2"
BUFFER_SIZE = 4000
OUTPUT_FILE = "c4_prompts.json"

print(f"Loading tokenizer '{TOKENIZER_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def process_text(text, num_tokens_trimmed):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= num_tokens_trimmed:
        return None, None
    prompt_tokens = tokens[:-num_tokens_trimmed]
    baseline_tokens = tokens[-num_tokens_trimmed:]
    prompt = tokenizer.convert_tokens_to_string(prompt_tokens)
    baseline_completion = tokenizer.convert_tokens_to_string(baseline_tokens)
    return prompt, baseline_completion

print("Loading C4 dataset (streaming)...")
dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)

print(f"Streaming {BUFFER_SIZE} samples from dataset to find {TARGET_SAMPLES} valid prompts...")
samples_list = list(islice(dataset, BUFFER_SIZE))
random.shuffle(samples_list)

c4_random_samples = []
processed_count = 0

for sample in samples_list:
    processed_count += 1
    if len(c4_random_samples) >= TARGET_SAMPLES:
        print(f"Collected {len(c4_random_samples)} valid prompts. Stopping.")
        break
    
    try:
        prompt, baseline_completion = process_text(sample["text"], NUM_TOKENS_TRIMMED)
        if prompt and baseline_completion:
            c4_random_samples.append({
                "prompt": prompt,
                "baseline": baseline_completion
            })
    except Exception:
        pass

print(f"Processed {processed_count} total samples from the buffer.")
if len(c4_random_samples) < TARGET_SAMPLES:
    print(f"WARNING: Only found {len(c4_random_samples)} valid prompts.")
else:
    print(f"Successfully collected {len(c4_random_samples)} prompts.")

print(f"Saving prompts to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(c4_random_samples, f, ensure_ascii=False, indent=4)

print("Done.")