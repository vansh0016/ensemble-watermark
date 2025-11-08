import json
import time
import os
# Use the working deep-translator library
from deep_translator import GoogleTranslator
import modules.shared as shared # To potentially access shared config if needed

# === Parameters ===
GENERATION_FILE_PATH = os.environ.get('GENERATION_FILE_PATH', 'outputs/generation_results.json')
ATTACK_TRANSLATION_OUTPUT_FILE = os.environ.get('ATTACK_TRANSLATION_OUTPUT_FILE', 'outputs/attacked_translation.json')
# ---

# Function to translate text using deep-translator (Google backend)
def translate_text_deep_translator(text, src_lang='en', intermediate_lang='es', target_lang='en', retries=3, delay=2):
    # Handle None or non-string input immediately
    if not isinstance(text, str) or not text.strip():
        # print("    DEBUG: Input text is None, empty, or whitespace. Returning empty string.")
        return ""

    current_text = text # Keep track of original text for retries

    for attempt in range(retries):
        try:
            # --- Stage 1: EN -> ES ---
            # print(f"    DEBUG: Attempt {attempt+1}, Stage 1 Input: '{current_text[:50]}...'")
            intermediate_text = GoogleTranslator(source=src_lang, target=intermediate_lang).translate(current_text)

            # Check intermediate result (deep-translator usually returns string or raises error)
            if not isinstance(intermediate_text, str) or not intermediate_text.strip():
                 print(f"    WARNING: Intermediate translation (to {intermediate_lang}) resulted in None or empty. Attempt {attempt + 1}/{retries}")
                 time.sleep(delay * (attempt + 1))
                 current_text = text # Reset for retry
                 continue
            # print(f"    DEBUG: Attempt {attempt+1}, Stage 1 Output ({intermediate_lang}): '{intermediate_text[:50]}...'")
            time.sleep(delay / 2) # Delay between API calls

            # --- Stage 2: ES -> EN ---
            # print(f"    DEBUG: Attempt {attempt+1}, Stage 2 Input: '{intermediate_text[:50]}...'")
            final_text = GoogleTranslator(source=intermediate_lang, target=target_lang).translate(intermediate_text)

            # Check final result
            if not isinstance(final_text, str): # Allow "" but not None
                 print(f"    WARNING: Final translation (to {target_lang}) resulted in None. Attempt {attempt + 1}/{retries}")
                 time.sleep(delay * (attempt + 1))
                 current_text = text # Reset for retry
                 continue
            # print(f"    DEBUG: Attempt {attempt+1}, Stage 2 Output ({target_lang}): '{final_text[:50]}...'")

            # If both stages succeeded:
            return final_text

        except Exception as e:
            # Catch errors from deep_translator (e.g., connection issues, unexpected responses)
            print(f"    WARNING: deep-translator error: {e}. Attempt {attempt + 1}/{retries}")
            time.sleep(delay * (attempt + 1))
            current_text = text # Reset to original text on any exception during retry

    print(f"    WARNING: Translation failed for text chunk after {retries} retries using deep-translator. Returning original.")
    return text # Return original if all retries fail

# --- Main script logic (loading, looping, saving) remains the same ---

print(f"  Loading generated text from '{GENERATION_FILE_PATH}' for translation attack...")
try:
    with open(GENERATION_FILE_PATH, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)
    print(f"  Loaded {len(generated_data)} samples.")
except FileNotFoundError:
    print(f"  ERROR: Cannot find '{GENERATION_FILE_PATH}'. Stopping translation attack.")
    generated_data = []
except json.JSONDecodeError:
    print(f"  ERROR: Cannot decode JSON from '{GENERATION_FILE_PATH}'.")
    generated_data = []

translated_results = []
print("  Starting back-and-forth translation (EN->ES->EN) using deep-translator...")

keys_to_translate = []
if generated_data:
    keys_to_translate = [k for k in generated_data[0].keys() if k != 'prompt']

for idx, sample in enumerate(generated_data, start=1):
    print(f"    Translating sample {idx}/{len(generated_data)}...")
    translated_sample = {'prompt': sample.get('prompt', '')}

    for key in keys_to_translate:
        original_text = sample.get(key)
        # Use the deep-translator function
        translated_text = translate_text_deep_translator(original_text)
        translated_sample[key] = translated_text
        time.sleep(0.5) # Keep delay

    translated_results.append(translated_sample)

print(f"  Translation complete. Saving {len(translated_results)} results to '{ATTACK_TRANSLATION_OUTPUT_FILE}'...")
try:
    with open(ATTACK_TRANSLATION_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(translated_results, f, ensure_ascii=False, indent=4)
    print("  Save complete.")
except Exception as e:
    print(f"  ERROR saving translation results: {e}")