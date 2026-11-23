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

def main():
    import json
    from unsloth import FastLanguageModel
    from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
    from unsloth import UnslothTrainingArguments as TrainingArguments
    from datasets import Dataset

    from arc_loader import ArcDataset
    from model_tools import InputMaskingDataCollator
    from model_tools import (load_unsloth_4bit, save_model_and_tokenizer, load_peft_state,
                              is_peft_model, keep_single_char_tokens, merge_peft_into_base)
    from resilient_inference import inference_run_resilient
    from selection import EvalTool
    from arc_downloader import download_arc_data

    # --- sharding: identical convention/seed to the base-model script ---
    shard_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_shards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    SPLIT_SEED = 42  # same seed as base-model script -> identical shard membership across both experiments

    raw_base_model = 'nvidia/Mistral-NeMo-Minitron-8B-Base'
    lora_checkpoint = '/path/to/MasterThesis/orig_0_25_dataset/training/pretrained_models/Mistral-NeMo-Minitron-Full-lora'
    arc_data_path = os.path.join('input', 'arc-prize-2024')
    download_arc_data(arc_data_path)

    # distinct output naming -- avoids collision with the base-model script's own shard dirs
    output_path = os.path.join(f'output_evaluation_Nemo_shard{shard_id}of{n_shards}')
    save_model_path = os.path.join(output_path, 'finetuned_model')
    #inference_cache = os.path.join(output_path, 'inference_cache')
    #inference_cache = os.path.join('/tmp', f'inference_cache_shard{shard_id}')
    import getpass

    user = getpass.getuser()
    job_id = os.environ.get('SLURM_JOB_ID', 'nojob')
    inference_cache = os.path.join('/tmp', f'{user}_inference_cache_finetuned_ttft_{job_id}_shard{shard_id}')

    submission_file = os.path.join(output_path, 'submission.json')
    failed_tasks_log = os.path.join(output_path, 'failed_tasks.txt')

    from dataset_utils import load_fixed_split
    _, held_out_set = load_fixed_split(arc_data_path)
    arc_eval_set = held_out_set.split(n_shards, SPLIT_SEED)[shard_id] if n_shards > 1 else held_out_set

    retrain = not os.path.exists(save_model_path)
    if retrain:
        model, tokenizer = load_unsloth_4bit(raw_base_model)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(save_model_path, local_files_only=True)  # note: from the SAVED (shrunk) checkpoint, not raw_base_model

    fmt_opts = dict(
        preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        query_beg='I', reply_beg='\n+/-=O', reply_end='\n' + tokenizer.eos_token,
        lines_sep='\n', max_tokens=8192,
    )

    if retrain:
        keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=') + tokenizer.tokenize('\n')
        keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

        model = FastLanguageModel.get_peft_model(
            model=model,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head'],
            r=256, lora_alpha=24, lora_dropout=0, bias="none",           # matches the Stage-1 checkpoint's own LoRA config
            use_gradient_checkpointing=True, random_state=42, use_rslora=True, loftq_config=None,
        )
        load_peft_state(model, lora_checkpoint)
        assert is_peft_model(model)

        train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        train_dataset_augment = arc_eval_set.remove_test_data().repeat(n=48, seed=0).augment(**train_aug_opts)
        train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

        checkpoint_dir = os.path.join(output_path, 'checkpoints')
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(checkpoint_dir) if os.path.isdir(checkpoint_dir) else None

        FastLanguageModel.for_training(model)
        trainer = Trainer(
            model=model, tokenizer=tokenizer,
            train_dataset=Dataset.from_list(train_dataset_as_list),
            dataset_text_field="text", max_seq_length=fmt_opts['max_tokens'],
            data_collator=InputMaskingDataCollator(
                instruction_template=fmt_opts['query_beg'], response_template=fmt_opts['reply_beg'],
                mlm=False, tokenizer=tokenizer, mask_first_n_examples=0,
            ),
            args=TrainingArguments(
                per_device_train_batch_size=2, gradient_accumulation_steps=2,
                warmup_ratio=0.25, num_train_epochs=1, learning_rate=1e-4, embedding_learning_rate=1e-5,
                fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=10,
                optim="adamw_8bit", weight_decay=0.00, lr_scheduler_type='cosine', seed=42,
                output_dir=checkpoint_dir, save_strategy="steps", save_steps=200, save_total_limit=2,
                report_to='none',
            ),
        )
        trainer_stats = unsloth_train(trainer, resume_from_checkpoint=last_checkpoint)

        # merge before saving -- makes save_model_path fully self-contained,
        # so the resilient worker's plain load_unsloth_4bit reload can't hit
        # the vocab-shape mismatch that a bare adapter-only save would risk here
        model = merge_peft_into_base(model)
        save_model_and_tokenizer(save_model_path, model, tokenizer)

        import gc, torch
        del model
        gc.collect()
        torch.cuda.empty_cache()

    infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000)
    infer_dataset = arc_eval_set.repeat(2).augment(**infer_aug_opts)
    print(f"shard {shard_id}/{n_shards}: {len(infer_dataset.grouped_keys())} grouped tasks")

    eval_tool = EvalTool(n_guesses=2)
    inference_results = inference_run_resilient(
        model_path=save_model_path,
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
