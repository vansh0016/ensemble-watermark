import gc
import random
import re
import time
import traceback
import ast
import hashlib

import numpy as np
import torch
import transformers

import adaptive_modules.shared as shared
from adaptive_modules.callbacks import (Iteratorize, Stream, _StopEverythingStoppingCriteria)

try:
    GROUP_0_CLASSES = set(shared.classes[:4]) 
    GROUP_1_CLASSES = set(shared.classes[4:8]) 
    
    GROUP_0_MASK = shared.GROUP_0_MASK
    GROUP_1_MASK = shared.GROUP_1_MASK
    print("  Successfully loaded sensorimotor masks from shared module.")
except AttributeError:
    print("  FATAL: Vocab masks not found in shared module. Run setup_and_load.py first.")
    import sys
    sys.exit(1)
except Exception as e:
    print(f"  FATAL ERROR loading masks: {e}")
    import sys
    sys.exit(1)

def get_max_prompt_length(state):
    return state['truncation_length'] - state['max_new_tokens']

def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    """Encodes a string prompt into tensor IDs and an attention mask."""
    
    inputs = shared.tokenizer(
        str(prompt), 
        return_tensors='pt', 
        add_special_tokens=add_special_tokens, 
        truncation=truncation_length is not None, 
        max_length=truncation_length
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    if hasattr(shared.tokenizer, 'bos_token_id') and shared.tokenizer.bos_token_id is not None:
        if add_bos_token:
            if (len(input_ids[0]) > 0 and input_ids[0][0] != shared.tokenizer.bos_token_id) or len(input_ids[0]) == 0:
                bos_tensor = torch.tensor([[shared.tokenizer.bos_token_id]])
                mask_tensor = torch.tensor([[1]])
                input_ids = torch.cat((bos_tensor, input_ids), 1)
                attention_mask = torch.cat((mask_tensor, attention_mask), 1)

            while len(input_ids[0]) > 1 and input_ids[0][0] == shared.tokenizer.bos_token_id and input_ids[0][1] == shared.tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]
                attention_mask = attention_mask[:, 1:]
        else:
            while len(input_ids[0]) > 0 and input_ids[0][0] == shared.tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]
                attention_mask = attention_mask[:, 1:]

    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]
        attention_mask = attention_mask[:, -truncation_length:]

    return input_ids.cuda(), attention_mask.cuda()

def decode(output_ids, skip_special_tokens=False):
    return shared.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)

def clear_torch_cache():
    gc.collect()
    torch.cuda.empty_cache()

def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

def apply_stopping_strings(reply, all_stop_strings):
    stop_found = False
    for string in all_stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]; stop_found = True; break
    if not stop_found:
        for string in all_stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]; break
            else: continue
            break
    return reply, stop_found

def calculate_adaptive_deltas(scores, base_deltas):
    """Calculates dynamic deltas based on the entropy of the logits."""
    probs = torch.nn.functional.softmax(scores, dim=-1)
    entropy = torch.distributions.Categorical(probs=probs).entropy()
    
    vocab_size = scores.shape[-1]
    max_entropy = torch.log(torch.tensor(vocab_size, device=scores.device))
    normalized_entropy = (entropy / max_entropy).clamp(0.0, 1.0)
    adaptive_strength = (1.0 - normalized_entropy) ** 2
    
    return {
        'senso': base_deltas['senso'] * adaptive_strength.unsqueeze(-1),
        'redgreen': base_deltas['redgreen'] * adaptive_strength.unsqueeze(-1)
    }

def get_redgreen_bias(last_token, delta, device, vocab_size):
    """Gets the bias tensor for the Red-Green watermark."""
    def default_hash_fn(token_id):
        return int(hashlib.sha256(str(token_id).encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    
    seed = default_hash_fn(last_token)
    generator = torch.Generator(device=device).manual_seed(seed)
    gli = torch.randperm(vocab_size, generator=generator, device=device)
    
    gamma = 0.5
    gls = int(gamma * vocab_size)
    green_list_indices = gli[:gls]
    
    bias_tensor = torch.zeros(vocab_size, device=device)
    bias_tensor[green_list_indices] += delta
    return bias_tensor

def get_sensorimotor_bias(current_bit, delta, device, vocab_size):
    """Gets the "push/pull" bias tensor for the Sensorimotor watermark."""
    bias_tensor = torch.zeros(vocab_size, device=device)
    
    if current_bit == 0:
        bias_tensor[GROUP_0_MASK] += delta
        bias_tensor[GROUP_1_MASK] -= delta
    elif current_bit == 1:
        bias_tensor[GROUP_0_MASK] -= delta
        bias_tensor[GROUP_1_MASK] += delta
        
    return bias_tensor

def token_hash(token_id):
    """Simple token hash function for bit selection."""
    return int(hashlib.sha256(str(token_id).encode('utf-8')).hexdigest(), 16)

def boost_tokens_with_adaptive_delta(input_ids, scores, **kwargs):
    """
    The new logits processor that applies adaptive deltas AND
    uses hash-based framing to be robust to desynchronization.
    """
    base_deltas = {
        'senso': shared.delta_senso, 
        'redgreen': shared.delta_redgreen
    }
    dynamic_deltas = calculate_adaptive_deltas(scores, base_deltas)
    
    last_token = input_ids[0][-1].item()
    vocab_size = scores.shape[-1]
    
    # Get Red-Green bias
    bias_tensor = get_redgreen_bias(last_token, dynamic_deltas['redgreen'][0], scores.device, vocab_size)
    
    # Get Sensorimotor bias
    message_bits = shared.message_state['message_bits']
    if message_bits:
        message_len = len(message_bits)
        
        # Use the hash of the last token to pick a bit from the secret key
        bit_index = token_hash(last_token) % message_len
        current_bit = message_bits[bit_index]
        
        # Get the sensorimotor "push/pull" bias for THIS bit
        bias_tensor += get_sensorimotor_bias(current_bit, dynamic_deltas['senso'][0], scores.device, vocab_size)

    scores = scores + bias_tensor
    return scores

def generate_reply(question, state, message_bits=None, eos_token=None, stopping_strings=None):
    """
    Main generation function.
    """    
    clear_torch_cache()
    done = False
    seed = set_manual_seed(state['seed'])
    generate_params = {}
    
    shared.message_state = {
        'message_bits': message_bits if message_bits is not None else []
    }

    for k in ['max_new_tokens', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'do_sample']:
        if k in state and state[k] is not None:
            generate_params[k] = state[k]

    if state.get('ban_eos_token', False):
        generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    input_ids, attention_mask = encode(
        question, 
        add_bos_token=state['add_bos_token'], 
        truncation_length=get_max_prompt_length(state)
    )

    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    if eos_token:
        eos_token_ids.append(shared.tokenizer.encode(eos_token, add_special_tokens=False)[0])
        
    generate_params['eos_token_id'] = eos_token_ids
    
    generate_params.update({
        'inputs': input_ids,
        'attention_mask': attention_mask 
    })
    
    generate_params['stopping_criteria'] = transformers.StoppingCriteriaList()
    generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria())
    generate_params.update({"logits_processor": transformers.LogitsProcessorList([boost_tokens_with_adaptive_delta])})

    full_generated_text = ""
    try:
        def generate_with_callback(callback=None, *args, **kwargs):
            kwargs['stopping_criteria'].append(Stream(callback_func=callback))
            clear_torch_cache()
            with torch.no_grad():
                shared.model.generate(**kwargs, pad_token_id=shared.tokenizer.eos_token_id)

        def generate_with_streaming(**kwargs):
            return Iteratorize(generate_with_callback, [], kwargs, callback=None)

        with generate_with_streaming(**generate_params) as generator:
            is_seq_seq = False 
            starting_from = 0 if is_seq_seq else len(input_ids[0])

            for output in generator:
                if output[-1] in eos_token_ids:
                    done = True
                    break
                
                full_generated_text = decode(
                    output[starting_from:], 
                    state['skip_special_tokens'] if state else True
                )

                if chr(0xfffd) in full_generated_text[-1:]:
                    continue

    except Exception:
        done = True
        traceback.print_exc()

    finally:
        all_stop_strings = []
        for st in (stopping_strings, state.get('custom_stopping_strings', [])):
            if type(st) is str:
                try: st = ast.literal_eval(f"[{st}]")
                except: st = [str(st)]
            if type(st) is list and len(st) > 0:
                all_stop_strings += st
        
        reply, stop_found = apply_stopping_strings(full_generated_text, all_stop_strings)
        return reply, done