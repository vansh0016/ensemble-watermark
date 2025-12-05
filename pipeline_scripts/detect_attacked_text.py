import os
import json
import spacy
import hashlib
from scipy.stats import norm, binom
import math
import pandas as pd
import modules.shared as shared
import torch
import torch.nn.functional as F
import sys

ATTACKED_TEXT_FILE_PATH = os.environ.get('ATTACKED_TEXT_FILE_PATH', 'outputs/attacked_translation.json') 
DETECTION_ATTACK_OUTPUT_FILE = os.environ.get('DETECTION_ATTACK_OUTPUT_FILE', 'outputs/detection_results_attack_translation.json')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Attack Detection script using device: {device}")

# Load dependencies from shared module
try:
    nlp = shared.nlp
    df_sensorimotor = shared.sensorimotor # From Lancaster CSV
    classes_list = shared.classes
    tokenizer_opt = shared.tokenizer
    model_opt = shared.model
except AttributeError as e:
    print(f"  ERROR: Shared module not initialized properly (perhaps setup didn't run?). Missing attribute: {e}")
    sys.exit(1)

print("  Loading sensorimotor frequency data 'updated_word_frequencies_with_percent.csv' for p_class...")
try:
    df_freq = pd.read_csv('updated_word_frequencies_with_percent.csv', header=0)
    shared.sensorimotor_freq = df_freq.set_index('Word').T.to_dict('dict')
except FileNotFoundError:
    print("  ERROR: 'updated_word_frequencies_with_percent.csv' not found. Detection may be inaccurate.")
    df_freq = pd.DataFrame(columns=['Word', 'Dominant.sensorimotor', 'Word_Percent'])
    shared.sensorimotor_freq = {}

# Sensorimotor Classes and Statistics from notebook
classes_mean_names = [
    'Auditory.mean', 'Gustatory.mean', 'Haptic.mean', 'Interoceptive.mean',
    'Olfactory.mean', 'Visual.mean', 'Foot_leg.mean', 'Hand_arm.mean',
    'Head.mean', 'Mouth.mean', 'Torso.mean'
]
mean_value = [1.51, 0.32, 1.07, 1.03, 0.39, 2.90, 0.81, 1.45, 2.28, 1.26, 0.82]
std_deviation = [0.99, 0.70, 0.93, 0.88, 0.62, 0.90, 0.75, 0.91, 0.72, 0.90, 0.67]
shared.secret_key = [0, 0]

def compute_p_class(df, classes):
    p_class_dict = {}
    total_word_percent = df['Word_Percent'].sum()
    for cls in classes:
        class_word_percent = df[df['Dominant.sensorimotor'] == cls]['Word_Percent'].sum()
        p = class_word_percent / total_word_percent if total_word_percent > 0 else 0.0
        p_class_dict[cls] = p
    return p_class_dict

p_class_dict = compute_p_class(df_freq, classes_list)

def split_into_sentences(text):
    if not text: return []
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def get_words_in_sentence(sentence):
    if not sentence: return []
    doc = nlp(sentence)
    return [token.text for token in doc if not token.is_punct and not token.is_space]

def secure_hash_for_word(word, range_min, range_max):
    hashed_word_bytes = hashlib.sha256(str(word).encode('utf-8')).digest()
    hashed_word_int = int.from_bytes(hashed_word_bytes[:4], byteorder='big')
    return (hashed_word_int % (range_max - range_min + 1)) + range_min

def secure_hash_for_sentence(sentence, range_min, range_max):
    if not sentence: return range_min
    doc = nlp(sentence)
    core_sentence = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    core_sentence_str = " ".join(core_sentence)
    hashed_sentence_bytes = hashlib.sha256(core_sentence_str.encode()).digest()
    hashed_sentence_int = int.from_bytes(hashed_sentence_bytes[:4], byteorder='big')
    return (hashed_sentence_int % (range_max - range_min + 1)) + range_min

def calculate_perplexity(text, model, tokenizer):
    if not text or not model or not tokenizer: return 0.0
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
    except Exception as e:
        return 0.0

def calculate_probs(reply):
    shared.secret_key = [0, 0]
    z_scores = []
    probabilities = []
    correct_acrosticons = 0
    sensorimotor_matches = 0
    sensorimotor_matches_per_class = {cls: 0 for cls in classes_list}
    sensorimotor_total_per_class = {cls: 0 for cls in classes_list}
    total_words = 0
    correct_words = 0
    
    if not reply:
        return 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, sensorimotor_matches_per_class

    try:
        tokenizer = shared.tokenizer
        tokens = tokenizer.encode(reply, add_special_tokens=False)
        vocab_size = len(tokenizer)
    except Exception as e:
        print(f"    WARNING: Tokenization failed for reply: {e}")
        return 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, sensorimotor_matches_per_class
        
    gamma = 0.5
    in_green_list = 0
    T = len(tokens) - 1

    sentences = split_into_sentences(reply)
    sensorimotor_class_idx = shared.secret_key[0]
    token_idx = 0

    for idx, sentence in enumerate(sentences, start=1):
        words = get_words_in_sentence(sentence)
        if not words: continue

        for word_idx, word in enumerate(words):
            total_words += 1

            is_acrostic_check = (word_idx == 0 and idx > 1)
            
            if is_acrostic_check:
                expected_start = chr(ord('A') + (shared.secret_key[1] % 26))
                actual_start = word[0].strip().upper() if word else ''
                if actual_start == expected_start:
                    correct_acrosticons += 1
            
            if 0 <= sensorimotor_class_idx < len(classes_list):
                selected_class_mean_name = classes_mean_names[sensorimotor_class_idx]
                selected_class = classes_list[sensorimotor_class_idx]
                word_upper = word.strip().upper()

                if word_upper in shared.sensorimotor_freq:
                    if shared.sensorimotor_freq[word_upper].get("Dominant.sensorimotor") == selected_class:
                        sensorimotor_matches_per_class[selected_class] += 1
                        sensorimotor_matches += 1
                    
                    if word_upper in df_sensorimotor and selected_class_mean_name in df_sensorimotor[word_upper]:
                         word_mean = df_sensorimotor[word_upper].get(selected_class_mean_name, mean_value[sensorimotor_class_idx]) # Default to mean if key missing
                         std_dev = std_deviation[sensorimotor_class_idx]
                         word_z = (word_mean - mean_value[sensorimotor_class_idx]) / std_dev if std_dev > 0 else 0.0
                         word_prob = 1 - norm.cdf(word_z)
                         z_scores.append(word_z)
                         probabilities.append(word_prob)

                    correct_words += 1
                    sensorimotor_total_per_class[selected_class] += 1

            sensorimotor_hash = secure_hash_for_word(word, 0, 10)
            sensorimotor_class_idx = sensorimotor_hash

            if token_idx < T and T > 0 and vocab_size > 0:
                try:
                    last_token = tokens[token_idx]
                    current_token = tokens[token_idx + 1]

                    def default_hash_fn(token_id):
                        return int(hashlib.sha256(str(token_id).encode('utf-8')).hexdigest(), 16) % (10 ** 8)

                    seed = default_hash_fn(last_token)
                    generator = torch.Generator(device='cuda:0' if device.type == 'cuda' else 'cpu').manual_seed(seed)
                    gli = torch.randperm(vocab_size, generator=generator, device=generator.device)
                    gls = int(gamma * vocab_size)

                    token_position = (gli == current_token).nonzero(as_tuple=True)[0]
                    if token_position.numel() > 0 and token_position.item() < gls:
                        in_green_list += 1
                except IndexError:
                    pass
                except Exception as e:
                    print(f"    WARNING: RedGreen check failed: {e}")
            
            token_idx += 1

        sentence_hash = secure_hash_for_sentence(sentence, 0, 25)
        shared.secret_key[1] = sentence_hash

    num_acrostic_checks = len(sentences) - 1 if len(sentences) > 1 else 0
    acrostic_prob = binom.sf(correct_acrosticons - 1, num_acrostic_checks, 1/26) if num_acrostic_checks > 0 else 1.0

    sensorimotor_prob = binom.sf(sensorimotor_matches - 1, correct_words, 1/len(classes_list)) if correct_words > 0 else 1.0
    
    if T > 0 and gamma * (1 - gamma) > 0:
        expected_in_green = gamma * T
        variance = T * gamma * (1 - gamma)
        redgreen_z_score = (in_green_list - expected_in_green) / math.sqrt(variance) if variance > 1e-9 else 0.0
        redgreen_prob = norm.sf(redgreen_z_score)
    else:
        redgreen_prob = 1.0
        
    final_score = acrostic_prob * sensorimotor_prob * redgreen_prob
    
    if math.isnan(final_score): final_score = 1.0
    if math.isnan(sensorimotor_prob): sensorimotor_prob = 1.0
    if math.isnan(acrostic_prob): acrostic_prob = 1.0
    if math.isnan(redgreen_prob): redgreen_prob = 1.0
    
    return (final_score, sensorimotor_prob, acrostic_prob, redgreen_prob, len(sentences),
            correct_acrosticons, total_words, correct_words, sensorimotor_matches,
            sensorimotor_matches_per_class)


print(f"  Loading ATTACKED text from '{ATTACKED_TEXT_FILE_PATH}'...")
try:
    with open(ATTACKED_TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
        attacked_data = json.load(f)
    print(f"  Successfully loaded {len(attacked_data)} attacked samples.")
except FileNotFoundError:
    print(f"  ERROR: Cannot find '{ATTACKED_TEXT_FILE_PATH}'. Stopping detection.")
    attacked_data = [] 
except json.JSONDecodeError:
    print(f"  ERROR: Could not decode JSON from '{ATTACKED_TEXT_FILE_PATH}'.")
    attacked_data = []

detection_results = []
print("  Starting detection and perplexity loop for attacked text...")

for idx, sample in enumerate(attacked_data, start=1):
    prompt = sample.get('prompt', '')
    print(f"    Detecting attacked sample {idx}/{len(attacked_data)}...")
    
    sample_detection = {'prompt': prompt, 'outputs': {}, 'detection': {}}
    keys_to_detect = [k for k in sample.keys() if k != 'prompt']
    
    for key in keys_to_detect:
        text = sample.get(key)
        sample_detection['outputs'][key] = text
        
        try:
            (final_score, sensorimotor_prob, acrostic_prob, redgreen_prob, num_sentences,
             correct_acrosticons, total_words, correct_words, sensorimotor_matches,
             sensorimotor_matches_per_class) = calculate_probs(text)
            
            perplexity = calculate_perplexity(text, model_opt, tokenizer_opt)
            
            sample_detection['detection'][key] = {
                'final_score': final_score,
                'sensorimotor_prob': sensorimotor_prob,
                'acrostic_prob': acrostic_prob,
                "redgreen_prob": redgreen_prob,
                'num_sentences': num_sentences,
                'correct_acrosticons': correct_acrosticons,
                'total_words': total_words,
                'correct_words': correct_words,
                'sensorimotor_matches': sensorimotor_matches,
                'sensorimotor_matches_per_class': sensorimotor_matches_per_class,
                'perplexity': perplexity
            }
        except Exception as e:
            print(f"      ERROR detecting {key}: {e}")
            sample_detection['detection'][key] = {'error': str(e)}

    detection_results.append(sample_detection)

print(f"  Detection complete. Saving {len(detection_results)} results to '{DETECTION_ATTACK_OUTPUT_FILE}'...")
try:
    with open(DETECTION_ATTACK_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(detection_results, f, ensure_ascii=False, indent=4)
    print("  Save complete.")
except Exception as e:
    print(f"  ERROR saving detection results for attacked text: {e}")