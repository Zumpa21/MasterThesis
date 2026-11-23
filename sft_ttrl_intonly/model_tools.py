import os
import json
import torch
import peft
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Union
from tokenizers import Tokenizer
from huggingface_hub import snapshot_download
from transformers import DataCollatorForLanguageModeling


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, response_template, instruction_template=None, *args, mlm=False, ignore_index=-100, padding_free=False, **kwargs):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            self.instruction_token_ids = instruction_template
        self.response_template = response_template
        if isinstance(response_template, str):
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            self.response_token_ids = response_template
        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn("The pad_token_id and eos_token_id values of this tokenizer are identical. If you are planning for multi-turn training, it can result in the model continuously generating questions and answers without eos token. To avoid this, set the pad_token_id to a different value.")
        self.ignore_index = ignore_index
        self.padding_free = padding_free

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None
                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    if self.response_token_ids == batch["labels"][i][idx: idx + len(self.response_token_ids)].tolist():
                        response_token_ids_start_idx = idx
                if response_token_ids_start_idx is None:
                    warnings.warn(f"Could not find response key `{self.response_template}` in the following instance: {self.tokenizer.decode(batch['input_ids'][i])} This instance will be ignored in loss calculation.")
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index
        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []
                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    if self.response_token_ids == batch["labels"][i][assistant_idx: assistant_idx + len(self.response_token_ids)].tolist():
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))
                if len(response_token_ids_idxs) == 0:
                    warnings.warn(f"Could not find response key `{self.response_template}` in the following instance: {self.tokenizer.decode(batch['input_ids'][i])} This instance will be ignored in loss calculation.")
                    batch["labels"][i, :] = self.ignore_index
                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    if human_token_ids == batch["labels"][i][human_idx: human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)
                if len(human_token_ids_idxs) == 0:
                    warnings.warn(f"Could not find instruction key `{self.instruction_template}` in the following instance: {self.tokenizer.decode(batch['input_ids'][i])} This instance will be ignored in loss calculation.")
                    batch["labels"][i, :] = self.ignore_index
                if len(human_token_ids_idxs) > 0 and len(response_token_ids_idxs) > 0 and human_token_ids_idxs[0] > response_token_ids_idxs[0]:
                    human_token_ids_idxs = [0] + human_token_ids_idxs
                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index
                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1]:] = self.ignore_index
        if self.padding_free:
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index
        return batch


class InputMaskingDataCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, mask_first_n_examples=0, **kwargs):
        super().__init__(**kwargs)
        self.mask_first_n_examples = mask_first_n_examples

    def torch_call(self, examples):
        batch = super().torch_call(examples)  # call super, masking all inputs
        for i in range(len(batch['labels'])):
            for _ in range(self.mask_first_n_examples):
                # mask first still unmasked output block
                beg_pos = ((batch['labels'][i] != -100).nonzero().min()).item()
                mid_pos = ((batch['labels'][i][beg_pos:] == -100).nonzero().min()).item() + beg_pos
                end_pos = ((batch['labels'][i] != -100).nonzero().max()).item() + 1
                if mid_pos < end_pos:
                    batch['labels'][i][beg_pos:mid_pos] = -100
        return batch


def load_unsloth_4bit(model_path):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True, #ADDED TO HANDLE NETWORK ERROR
    )
    if model.max_seq_length == 2048 < model.generation_config.max_length:
        print(f'CHANGING MAX_SEQ_LENGTH {model.max_seq_length} -> {model.generation_config.max_length} (unsloth bug?)')
        to_fix = model
        while to_fix is not None:
            to_fix.max_seq_length = model.generation_config.max_length
            to_fix = getattr(to_fix, 'model', None)
    return model, tokenizer


def save_model_and_tokenizer(store_path, model, tokenizer):
    model.save_pretrained(store_path)
    tokenizer.save_pretrained(store_path)
    to_delete = os.path.join(store_path, 'tokenizer.model')  # delete file, as it interferes with token removal
    if os.path.isfile(to_delete):
        os.remove(to_delete)


def fix_dtypes(model, fix_weights=True, fix_quant_states=True):
    for module in model.modules():
        weight = getattr(module, 'weight', None)
        if weight is not None:
            if torch.is_floating_point(weight):
                if fix_weights and weight.dtype != model.dtype:
                    module.to(model.dtype)
            else:
                qs = getattr(weight, 'quant_state', None)
                if qs is not None:
                    if fix_quant_states and qs.dtype != model.dtype:
                        qs.dtype = model.dtype
    return model


def is_peft_model(model):
    return hasattr(model, 'peft_type')


def merge_peft_into_base(model):
    assert is_peft_model(model)
    return fix_dtypes(model.merge_and_unload())


def get_and_fix_peft_weights(store):
    state_dict = peft.load_peft_weights(store)
    for k in list(state_dict.keys()):
        if 'modules_to_save' in k:
            del state_dict[k]
            original_module_key = k.replace('.modules_to_save.', '.original_module.')
            if original_module_key in state_dict: del state_dict[original_module_key]
            assert k.replace('.modules_to_save.', '.') in state_dict
    return state_dict


def set_peft_weights(model, state_dict):
    res = peft.set_peft_model_state_dict(model, state_dict)
    assert not res.unexpected_keys, 'error loading weights - some keys not available in model'


def load_peft_state(model, store):
    set_peft_weights(model, get_and_fix_peft_weights(store))


def get_or_map_special_tokens(data, mapping=None):
    tokens = set()
    if isinstance(data, dict):
        special = data.get('special_tokens')
        if special is not None:
            for v in special.values():
                tokens.update(v['ids'])
                if mapping is not None:
                    v['ids'] = [mapping.get(i) for i in v['ids'] if i in mapping]
        for v in data.values():
            tokens.update(get_or_map_special_tokens(v, mapping))
    if isinstance(data, list):
        for v in data:
            tokens.update(get_or_map_special_tokens(v, mapping))
    return tokens


def remove_tokenizer_normalizer(tokenizer):
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    if tokenizer_json.get('normalizer') is not None:
        tokenizer_json['normalizer'] = None
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))


def shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special=True, remove_unk=False):
    assert tokenizer.is_fast
    tok_json = json.loads(tokenizer._tokenizer.to_str())
    assert tok_json['model']['type'] == "BPE"
    if keep_special:
        keep_indices.update(tokenizer.all_special_ids)
        keep_indices.update(get_or_map_special_tokens(tok_json.get('post_processor')))
    if remove_unk:
        keep_indices -= {tokenizer.unk_token_id}
    mapping = {old: new for new, old in enumerate(sorted(keep_indices))}
    tok_json['model']['vocab'] = {k: mapping[v] for k, v in tok_json['model']['vocab'].items() if v in mapping}
    tok_json['model']['merges'] = []
    tok_json['added_tokens'] = [{**t, 'id': mapping[t['id']]} for t in tok_json['added_tokens'] if t['id'] in mapping]
    tok_json['added_tokens'] = sorted(tok_json['added_tokens'], key=lambda t: t['id'])
    get_or_map_special_tokens(tok_json.get('post_processor'), mapping)
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tok_json))
    if remove_unk:
        tokenizer.unk_token = None
    return mapping


def shrink_model_embeddings(model, mapping):
    with torch.no_grad():
        row_select = torch.tensor([x[0] for x in sorted(mapping.items(), key=lambda x: x[1])])
        row_select = row_select.to(model.get_input_embeddings().weight.data.device)
        new_embed_t = torch.index_select(model.get_input_embeddings().weight.data, 0, row_select)
        row_select = row_select.to(model.get_output_embeddings().weight.data.device)
        new_lm_head = torch.index_select(model.get_output_embeddings().weight.data, 0, row_select)
        model.resize_token_embeddings(len(row_select))
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))


def keep_single_char_tokens(model, tokenizer, keep=None, keep_norm=False, keep_model_tok=True, **kwargs):
    if not keep_norm:
        remove_tokenizer_normalizer(tokenizer)
    if keep is None:
        keep_indices = set(v for k, v in tokenizer.vocab.items() if len(k) == 1)
    else:
        keep_indices = set(tokenizer.vocab[t] for t in keep)
    if keep_model_tok:
        for config in [model.config, model.generation_config]:
            for k, v in config.to_dict().items():
                if k.endswith('token_id'):
                    keep_indices.update(v if isinstance(v, list) else [v])
    keep_indices -= {None}
    mapping = shrink_tokenizer_vocab(tokenizer, keep_indices, **kwargs)
    shrink_model_embeddings(model, mapping)
    return mapping
