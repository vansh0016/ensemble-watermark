import gc
import json
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaTokenizer)

import adaptive_modules.shared as shared



def find_model_type(model_name):
    path_to_model = Path(f'{shared.model_dir}/{model_name}')
    if not path_to_model.exists():
        return 'None'

    model_name_lower = model_name.lower()
    
    config = AutoConfig.from_pretrained(path_to_model, trust_remote_code=shared.trust_remote_code)
    if config.to_dict().get("is_encoder_decoder", False):
        return 'HF_seq2seq'
    else:
        return 'HF_generic'
    

def get_model_metadata(model):
    model_settings = None
    return model_settings

def load_model(model_name, gptq = False, awq = False):
    print(f"Loading {model_name}...")
    t0 = time.time()

    
    shared.model_type = find_model_type(model_name)
    if shared.model_type == 'None':
        print('The path to the model does not exist. Exiting.')
        return None, None
    
    shared.model_name = model_name

    if gptq == True:
        load_func = AutoGPTQ_loader
    elif awq == True:
        load_func = AutoAWQ_loader
    else:
        load_func = huggingface_loader

    output = load_func(model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        model = output
        if model is None:
            return None, None
        else:
            tokenizer = load_tokenizer(model_name, model)



    print(f"Loaded the model in {(time.time()-t0):.2f} seconds.\n")
    return model, tokenizer

def load_tokenizer(model_name, model):
    tokenizer = None
    path_to_model = Path(f"{shared.model_dir}/{model_name}/")
    tokenizer = AutoTokenizer.from_pretrained(
            path_to_model,
            trust_remote_code=shared.trust_remote_code,
            use_fast=shared.no_use_fast
        )

    return tokenizer

def huggingface_loader(model_name):
    path_to_model = Path(f'{shared.model_dir}/{model_name}')

    params = {
        'low_cpu_mem_usage': True,
        'torch_dtype': torch.float16,
    }
    if shared.trust_remote_code:
        params['trust_remote_code'] = True
    if shared.use_flash_attention_2:
        params['use_flash_attention_2'] = True
    if shared.use_eager_attention:
        params['attn_implementation'] = 'eager'

    config = AutoConfig.from_pretrained(path_to_model, trust_remote_code=shared.trust_remote_code)
    if 'chatglm' in model_name.lower():
        LoaderClass = AutoModel
    else:
        if config.to_dict().get('is_encoder_decoder', False):
            LoaderClass = AutoModelForSeq2SeqLM
            shared.is_seq2seq = True
        else:
            LoaderClass = AutoModelForCausalLM

    model = LoaderClass.from_pretrained(path_to_model, **params)
    model = model.cuda()
  
    return model

def GPTQ_loader(model_name):
    import modules.GPTQ_loader

    model = modules.GPTQ_loader.load_quantized(model_name)

    return model

def AutoGPTQ_loader(model_name):
    import modules.AutoGPTQ_loader

    return modules.AutoGPTQ_loader.load_quantized(model_name,)

def AutoAWQ_loader(model_name):
    from awq import AutoAWQForCausalLM

    model_dir = Path(f'{shared.model_dir}/{model_name}')

    model = AutoAWQForCausalLM.from_quantized(
                quant_path=model_dir,
                max_new_tokens=512,
                trust_remote_code=shared.trust_remote_code,
                fuse_layers=True,
                max_memory=get_max_memory_dict(),
                batch_size=1,
                safetensors=any(model_dir.glob('*.safetensors')),
            )

    return model


def get_max_memory_dict():
    max_memory = {}

    total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
    suggestion = round((total_mem - 1000) / 1000) * 1000
    if total_mem - suggestion < 800:
        suggestion -= 1000

    suggestion = int(round(suggestion / 1000))
    print(f"Auto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors. You can manually set other values.")
    max_memory = {0: f'{suggestion}GiB', 'cpu': f'{64}GiB'}

    return max_memory if len(max_memory) > 0 else None