import time
from pathlib import Path
import textwrap
import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

import modules.shared as shared
from modules.model import load_model
import hashlib
import spacy
import pandas as pd
import os

print("  Loading Spacy model 'en_core_web_sm'...")
nlp = spacy.load("en_core_web_sm")
shared.nlp = nlp

shared.model_name = "Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
shared.act_order = True
print(f"  Loading model: {shared.model_name}...")
t0 = time.time()

shared.groupsize = 128
shared.wbits = 4
shared.use_flash_attention_2 = False
shared.model, shared.tokenizer = load_model(shared.model_name, gptq=False, awq=False)

print(f"  Model loaded in {(time.time()-t0):.2f} seconds.")

csv_path = 'Lancaster_sensorimotor_norms_for_39707_words.csv'
if not os.path.exists(csv_path):
    print(f"  ERROR: Cannot find {csv_path}. Make sure it's in the root folder.")
else:
    print(f"  Loading sensorimotor data from {csv_path}...")
    df = pd.read_csv(csv_path, header=0)
    shared.sensorimotor = df.set_index('Word').T.to_dict('dict')
    shared.classes = ['Auditory', 'Gustatory', 'Haptic', 'Interoceptive', 'Olfactory', 'Visual', 'Foot_leg', 'Hand_arm', 'Head', 'Mouth', 'Torso']
    print("  Sensorimotor data loaded into shared.sensorimotor.")

from modules.text_generation import generate_reply

def secure_hash_for_sentence(last_sentence, range_min, range_max):
    doc = nlp(last_sentence)
    core_sentence = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    core_sentence_str = " ".join(core_sentence)
    
    hashed_sentence_bytes = hashlib.sha256(core_sentence_str.encode()).digest()
    hashed_sentence_integers = int.from_bytes(hashed_sentence_bytes[:4], byteorder='big')
    
    mapped_sentence_number = (hashed_sentence_integers % (range_max - range_min + 1)) + range_min
    return mapped_sentence_number

def secure_hash_for_token(last_token, range_min, range_max):
    hashed_token_bytes = hashlib.sha256(str(last_token.item()).encode()).digest()
    hashed_token_integers = int.from_bytes(hashed_token_bytes[:4], byteorder='big')
    
    mapped_token_number = (hashed_token_integers % (range_max - range_min + 1)) + range_min
    return mapped_token_number

def secure_hash_to_numbers(last_sentence, last_token, range_list):
    result_numbers = []
    if last_token is not None:
        token_hash = secure_hash_for_token(last_token, range_list[0][0], range_list[0][1])
    else:
        token_hash = shared.secret_key[0]
    result_numbers.append(token_hash)
    
    if last_sentence is not None:
        sentence_hash = secure_hash_for_sentence(last_sentence, range_list[1][0], range_list[1][1])
    else:
        sentence_hash = shared.secret_key[1]
    result_numbers.append(sentence_hash)
    
    return result_numbers

def get_last_sentence(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences[-1] if sentences else None

shared.secure_hash_for_sentence = secure_hash_for_sentence
shared.secure_hash_for_token = secure_hash_for_token
shared.secure_hash_to_numbers = secure_hash_to_numbers
shared.get_last_sentence = get_last_sentence