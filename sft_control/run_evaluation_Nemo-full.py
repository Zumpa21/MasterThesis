# Copyright 2024 Daniel Franzen and Jan Disselhoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["UNSLOTH_DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_CACHE_ROOT"] = "/path/to/cache/vllm"
os.environ["VLLM_NO_USAGE_STATS"] = "1"

import json
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments
from datasets import Dataset

from transformers import TrainerCallback
import numpy as np

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator, DataCollatorForCompletionOnlyLM
from model_tools import (load_unsloth_4bit, save_model_and_tokenizer, load_peft_state,
                              is_peft_model, keep_single_char_tokens, merge_peft_into_base)
from resilient_inference import inference_run_resilient
from selection import EvalTool
from arc_downloader import download_arc_data

#def build_loo_dataset(arc_set):
"""Rotate each train pair into the 'test' slot, using the rest as context.
Mirrors move_test_to_train's pattern but for one held-out pair at a time."""
#    loo_challenge = {}
#    loo_solutions = {}
#    for base_key, task in arc_set.challenge.items():
#        train_ex = task['train']
#        for i in range(len(train_ex)):
#            loo_key = f'{base_key}-loo{i}'
#            loo_challenge[loo_key] = {
#                'train': train_ex[:i] + train_ex[i+1:],
#                'test': [{'input': train_ex[i]['input']}],
#            }
#            loo_solutions[loo_key] = [train_ex[i]['output']]
#    return ArcDataset(challenge=loo_challenge, solutions=loo_solutions, is_orig=True)

def build_loo_dataset(arc_set, min_remaining=3):
    """Rotate each train pair into the 'test' slot, using the rest as context.
    Mirrors move_test_to_train's pattern but for one held-out pair at a time.

    Skips tasks whose original train set is too small to leave at least
    `min_remaining` context pairs after holding one out -- per the
    loo_context_diagnostic.py finding that the 1-2-remaining-pair buckets
    perform far worse (11.6-15.4%) than 3+ (16.7-37.1%) and, worse, the
    2-remaining bucket alone was ~43% of the entire training distribution."""
    loo_challenge = {}
    loo_solutions = {}
    skipped_tasks = 0
    for base_key, task in arc_set.challenge.items():
        train_ex = task['train']
        if len(train_ex) - 1 < min_remaining:
            skipped_tasks += 1
            continue
        for i in range(len(train_ex)):
            loo_key = f'{base_key}-loo{i}'
            loo_challenge[loo_key] = {
                'train': train_ex[:i] + train_ex[i+1:],
                'test': [{'input': train_ex[i]['input']}],
            }
            loo_solutions[loo_key] = [train_ex[i]['output']]
    print(f"build_loo_dataset: skipped {skipped_tasks} tasks with < {min_remaining + 1} "
          f"train examples (would leave < {min_remaining} remaining pairs)")
    return ArcDataset(challenge=loo_challenge, solutions=loo_solutions, is_orig=True)

class LastOutputOnlyCollator(DataCollatorForCompletionOnlyLM):
    """Loss ONLY on the final output grid (the held-out LOO answer), exactly the part GRPO rewards.
    The parent masks every input; here we also mask every output block except the last one."""
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for i in range(len(batch['labels'])):
            keep = (batch['labels'][i] != -100).nonzero().flatten()
            if len(keep) == 0:
                continue
            gaps = (keep[1:] - keep[:-1] > 1).nonzero().flatten()
            start = keep[gaps[-1] + 1].item() if len(gaps) else keep[0].item()  # start of the last unmasked block
            batch['labels'][i, :start] = -100
        return batch


def main():
    class DiagnosticCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            print(f"[diagnostic] step {state.global_step}, epoch {state.epoch:.2f}")

    # ---- SFT CONTROL: same LOO data as GRPO, supervised loss instead of reward ----
    batch_size = 1          # 1 prompt per optimizer step, like GRPO (1 prompt x 4 generations per step)
    lr = 1e-5 #like GRPO #1e-4               # SFT learning rate (Stage-1 / TTFT recipe)
    embedding_lr = 1e-5     # as in Stage-1
    ms = 700                # same number of prompt visits as the 700-step GRPO runs
    min_remaining = 1       # SAME filter as GRPO (skips ~62 of 100 tasks!)
    only_last_output = True #False # True: loss only on the held-out answer (= what GRPO rewards)

    print(f"[parameters] method: SFT control on LOO data")
    print(f"[parameters] per_device_train_batch_size: {batch_size}")
    print(f"[parameters] learning_rate: {lr}, embedding_learning_rate: {embedding_lr}")
    print(f"[parameters] max_steps: {ms}")
    print(f"[parameters] min_remaining: {min_remaining}, only_last_output: {only_last_output}")

    # --- sharding: identical convention/seed to the base-model script ---
    shard_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_shards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    SPLIT_SEED = 42  # same seed as base-model script -> identical shard membership across both experiments

    raw_base_model = 'nvidia/Mistral-NeMo-Minitron-8B-Base'
    merged_stage1_path = '/path/to/MasterThesis/sft_control/pretrained_models/Mistral-NeMo-Minitron-Full-merged'
    lora_checkpoint = '/path/to/MasterThesis/sft_control/pretrained_models/Mistral-NeMo-Minitron-Full-lora'
    arc_data_path = os.path.join('input', 'arc-prize-2024')
    download_arc_data(arc_data_path)

    # distinct output naming -- avoids collision with the base-model script's own shard dirs
    output_path = os.path.join(f'output_evaluation_Nemo_shard{shard_id}of{n_shards}')
    #inference_cache = os.path.join(output_path, 'inference_cache')
    #inference_cache = os.path.join('/tmp', f'inference_cache_shard{shard_id}')
    import getpass

    user = getpass.getuser()
    job_id = os.environ.get('SLURM_JOB_ID', 'nojob')
    inference_cache = os.path.join('/tmp', f'{user}_inference_cache_sft_control_{job_id}_shard{shard_id}')

    submission_file = os.path.join(output_path, 'submission.json')
    failed_tasks_log = os.path.join(output_path, 'failed_tasks.txt')

    from dataset_utils import load_fixed_split
    _, held_out_set = load_fixed_split(arc_data_path)
    arc_eval_set = held_out_set.split(n_shards, SPLIT_SEED)[shard_id] if n_shards > 1 else held_out_set

    # FIX 2: we save ONLY the LoRA adapter (no merge). Merging into the 4-bit model rounds the update away.
    adapter_path = os.path.join(output_path, 'sft_adapter')
    retrain = not os.path.exists(adapter_path)
    print(f"[setup] retrain={retrain} (adapter_path={adapter_path})")
    if retrain:
        model, tokenizer = load_unsloth_4bit(merged_stage1_path)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(merged_stage1_path, local_files_only=True)  # same (shrunk) tokenizer as Stage-1

    fmt_opts = dict(
        preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        query_beg='I', reply_beg='\n+/-=O', reply_end='\n' + tokenizer.eos_token,
        lines_sep='\n', max_tokens=8192,
    )

    if retrain:
        model = FastLanguageModel.get_peft_model(
            model=model,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head'],
            r=64, lora_alpha=24, lora_dropout=0, bias="none",           # matches the Stage-1 checkpoint's own LoRA config
            use_gradient_checkpointing=True, random_state=42, use_rslora=True, loftq_config=None,
        )
        assert is_peft_model(model)

        #train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        train_aug_opts = dict(tp=True, rt=True, perm=False, shfl_ex=False, seed=0)
        loo_set = build_loo_dataset(arc_eval_set.remove_test_data(), min_remaining=min_remaining).repeat(n=3, seed=0).augment(**train_aug_opts)

        computed_max_len = loo_set.max_new_tokens(**fmt_opts)
        print(f"Data-driven max completion length needed: {computed_max_len}")

        rows = []
        for key in loo_set.keys:
            key, fmt = loo_set.get_task(key, reply=None, len_name='input', **fmt_opts)
            rows.append(dict(
                prompt=fmt['input'],
                task_key=key,   # FIX 1: augmented key (e.g. 'xxxx-loo0_0.tp.rt'); decode(key=...) undoes the augmentation
            ))

        # SFT rows: same LOO keys, but formatted WITH the correct answer (reply) appended
        sft_rows = []
        for r in rows:
            _, fmt = loo_set.get_task(r['task_key'], len_name='text', **fmt_opts)
            sft_rows.append(dict(text=fmt['text']))
        print(f"[data] {len(sft_rows)} LOO training examples from "
              f"{len(set(ArcDataset.get_base_key(r['task_key']).split('-loo')[0] for r in rows))} tasks "
              f"(of {len(arc_eval_set.challenge)} in this shard)")

        checkpoint_dir = os.path.join(output_path, 'checkpoints')
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(checkpoint_dir) if os.path.isdir(checkpoint_dir) else None

        collator_cls = LastOutputOnlyCollator if only_last_output else InputMaskingDataCollator
        collator_kwargs = {} if only_last_output else dict(mask_first_n_examples=1)

        FastLanguageModel.for_training(model)
        tokenizer.padding_side = 'right'
        sft_trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=Dataset.from_list(sft_rows),
            dataset_text_field="text",
            max_seq_length=fmt_opts['max_tokens'],
            packing=False,
            data_collator=collator_cls(
                instruction_template=fmt_opts['query_beg'],
                response_template=fmt_opts['reply_beg'],
                mlm=False,
                tokenizer=tokenizer,
                **collator_kwargs,
            ),
            callbacks=[DiagnosticCallback()],
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=1,
                warmup_ratio=0.25,
                max_steps=ms,
                learning_rate=lr,
                embedding_learning_rate=embedding_lr,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.00,
                lr_scheduler_type='cosine',
                seed=42,
                output_dir=checkpoint_dir,
                save_strategy='steps',
                save_steps=150,
                save_total_limit=2,
                report_to='none',
            ),
        )
        trainer_stats = unsloth_train(sft_trainer, resume_from_checkpoint=last_checkpoint)
        model = sft_trainer.model

        # sanity check: the adapter must change the model's output (diff must NOT be 0)
        import torch
        from inference_tools import calc_score
        _, f0 = loo_set.get_task(rows[0]['task_key'], len_name='input', **fmt_opts)
        with torch.no_grad():
            s_on = calc_score(input=f0['input'], reply=f0['reply'], model_tok=(model, tokenizer))
            with model.disable_adapter():
                s_off = calc_score(input=f0['input'], reply=f0['reply'], model_tok=(model, tokenizer))
        print(f"[adapter check] -logp with adapter {s_on:.6f}, without {s_off:.6f}, diff {s_on - s_off:+.6f}")

        # FIX 2: save ONLY the adapter, no merge_peft_into_base
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        import gc, torch
        del model
        gc.collect()
        torch.cuda.empty_cache()

    infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000)
    infer_dataset = arc_eval_set.repeat(2).augment(**infer_aug_opts)
    print(f"shard {shard_id}/{n_shards}: {len(infer_dataset.grouped_keys())} grouped tasks")

    eval_tool = EvalTool(n_guesses=2)
    inference_results = inference_run_resilient(
        model_path=merged_stage1_path,   # FIX 2: 4-bit Stage-1 ...
        adapter_path=adapter_path,       # ... + SFT adapter on top (not merged)
        dataset=infer_dataset,
        fmt_opts=fmt_opts,
        min_prob=0.1,
        aug_score_opts=infer_aug_opts,
        callback=eval_tool.process_result,
        cache_path=inference_cache,
        failed_tasks_log=failed_tasks_log,
    )

    with open(submission_file, 'w') as f:
        json.dump(arc_eval_set.get_submission(inference_results), f)
    with open(submission_file, 'r') as f:
        print(f"Score for '{submission_file}':", arc_eval_set.validate_submission(json.load(f)))

if __name__ == '__main__':
    main()
