import time
from pathlib import Path
import textwrap
import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
parent_root = os.path.dirname(project_root)
if parent_root not in sys.path:
    sys.path.append(parent_root)

import adaptive_modules.shared as shared
from adaptive_modules.model import load_model
import hashlib
import spacy
import pandas as pd

print("  Loading Spacy model 'en_core_web_sm'...")
nlp = spacy.load("en_core_web_sm")
shared.nlp = nlp

shared.model_name = os.environ.get('MODEL_NAME', "Mistral-7B-Instruct-v0.3")
shared.act_order = True
shared.groupsize = 128
shared.wbits = 4
shared.use_flash_attention_2 = False

print(f"  Loading model: {shared.model_name}...")
t0 = time.time()
shared.model, shared.tokenizer = load_model(shared.model_name, gptq=False, awq=False)

# Enforce tokenizer-safe parameters globally
if hasattr(shared.tokenizer, "clean_up_tokenization_spaces"):
    shared.tokenizer.clean_up_tokenization_spaces = False
if hasattr(shared.tokenizer, "use_fast"):
    try:
        shared.tokenizer = AutoTokenizer.from_pretrained(
            shared.model_name,
            use_fast=False,
            clean_up_tokenization_spaces=False,
            trust_remote_code=True
        )
        print("  Reloaded tokenizer with safe decoding parameters.")
    except Exception as e:
        print(f"  WARNING: Could not reload tokenizer safely ({e}), using existing tokenizer.")

shared.model_name = shared.model_name
print(f"  Model loaded in {(time.time()-t0):.2f} seconds.")

csv_path = 'Lancaster_sensorimotor_norms_for_39707_words.csv'
if not os.path.exists(csv_path):
    print(f"  ERROR: Cannot find {csv_path}. Make sure it's in the root folder.")
else:
    print(f"  Loading sensorimotor data from {csv_path}...")
    df = pd.read_csv(csv_path, header=0)
    shared.sensorimotor = df.set_index('Word').T.to_dict('dict')
    all_classes = [
        'Auditory', 'Gustatory', 'Haptic', 'Interoceptive',
        'Olfactory', 'Visual', 'Foot_leg', 'Hand_arm', 'Head', 'Mouth', 'Torso'
    ]
    shared.classes = all_classes[:8]
    print(f"  Sensorimotor data loaded for {len(shared.classes)} classes.")

print("  Pre-computing sensorimotor vocabulary masks...")
try:
    GROUP_0_CLASSES = set(shared.classes[:4])  # Auditory, Gustatory, Haptic, Interoceptive
    GROUP_1_CLASSES = set(shared.classes[4:8])  # Olfactory, Visual, Foot_leg, Hand_arm

    VOCAB_SIZE = len(shared.tokenizer.get_vocab())
    VOCAB_DECODE = [shared.tokenizer.decode(i, skip_special_tokens=False) for i in range(VOCAB_SIZE)]
    TOKEN_STRIPPED_UPPER = [token.strip().upper() for token in VOCAB_DECODE]

    shared.GROUP_0_MASK = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device='cuda:0')
    shared.GROUP_1_MASK = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device='cuda:0')

    for i, token_str in enumerate(TOKEN_STRIPPED_UPPER):
        if token_str in shared.sensorimotor:
            dominant_class = shared.sensorimotor[token_str]["Dominant.sensorimotor"]
            if dominant_class in GROUP_0_CLASSES:
                shared.GROUP_0_MASK[i] = True
            elif dominant_class in GROUP_1_CLASSES:
                shared.GROUP_1_MASK[i] = True
    print("  Sensorimotor masks computed and attached to shared module.")
except Exception as e:
    print(f"  FATAL ERROR: Could not compute sensorimotor masks. {e}")
    sys.exit(1)

print("  Setup for adaptive pipeline complete.")