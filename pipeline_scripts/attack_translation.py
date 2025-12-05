import json
import time
import os
from deep_translator import GoogleTranslator
import modules.shared as shared

GENERATION_FILE_PATH = os.environ.get('GENERATION_FILE_PATH', 'outputs/generation_results.json')
ATTACK_TRANSLATION_OUTPUT_FILE = os.environ.get('ATTACK_TRANSLATION_OUTPUT_FILE', 'outputs/attacked_translation.json')

def translate_text_deep_translator(text, src_lang='en', intermediate_lang='es', target_lang='en', retries=3, delay=2):
    """ Translates text from src_lang to target_lang via deep-translator"""
    if not isinstance(text, str) or not text.strip():
        return ""

    current_text = text

    for attempt in range(retries):
        try:
            # EN -> ES
            intermediate_text = GoogleTranslator(source=src_lang, target=intermediate_lang).translate(current_text)

            if not isinstance(intermediate_text, str) or not intermediate_text.strip():
                 print(f"    WARNING: Intermediate translation (to {intermediate_lang}) resulted in None or empty. Attempt {attempt + 1}/{retries}")
                 time.sleep(delay * (attempt + 1))
                 current_text = text
                 continue
            time.sleep(delay / 2)

            # ES -> EN
            final_text = GoogleTranslator(source=intermediate_lang, target=target_lang).translate(intermediate_text)

            if not isinstance(final_text, str):
                 print(f"    WARNING: Final translation (to {target_lang}) resulted in None. Attempt {attempt + 1}/{retries}")
                 time.sleep(delay * (attempt + 1))
                 current_text = text
                 continue

            return final_text

        except Exception as e:
            print(f"    WARNING: deep-translator error: {e}. Attempt {attempt + 1}/{retries}")
            time.sleep(delay * (attempt + 1))
            current_text = text

    print(f"    WARNING: Translation failed for text chunk after {retries} retries using deep-translator. Returning original.")
    return text

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
        translated_text = translate_text_deep_translator(original_text)
        translated_sample[key] = translated_text
        time.sleep(0.5)

    translated_results.append(translated_sample)

print(f"  Translation complete. Saving {len(translated_results)} results to '{ATTACK_TRANSLATION_OUTPUT_FILE}'...")
try:
    with open(ATTACK_TRANSLATION_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(translated_results, f, ensure_ascii=False, indent=4)
    print("  Save complete.")
except Exception as e:
    print(f"  ERROR saving translation results: {e}")