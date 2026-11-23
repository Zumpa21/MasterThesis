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
import glob
import gc
import torch
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments
from datasets import Dataset

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import load_unsloth_4bit, keep_single_char_tokens, save_model_and_tokenizer
from model_tools import load_peft_state, merge_peft_into_base
from arc_downloader import download_arc_data

# Force all dataset.map() calls (including Unsloth's internal ones) to run
# single-process, in-process. Setting num_proc=1 is NOT sufficient — Unsloth's
# internal _prepare_dataset() spawns a worker pool regardless of that value.
# See: https://github.com/unslothai/unsloth/issues/4490
import datasets
_original_map = datasets.Dataset.map
def _map_no_subprocess(self, *args, **kwargs):
    kwargs['num_proc'] = None
    return _original_map(self, *args, **kwargs)
datasets.Dataset.map = _map_no_subprocess

os.environ["UNSLOTH_DISABLE_TELEMETRY"] = "1" #disabling unsloth statistic calls (DANI)

# input paths
base_model = 'nvidia/Mistral-NeMo-Minitron-8B-Base'  # auto-downloaded from huggingface.co
arc_data_path = os.path.join('input', 'arc-prize-2024')  # as on kaggle arc prize 2024
download_arc_data(arc_data_path)
re_arc_path = os.path.join('input', 're_arc')  # https://github.com/michaelhodel/re-arc
neoneye_path = os.path.join('input', 'arc-dataset-collection')  # https://github.com/neoneye/arc-dataset-collection

# output paths
save_model_path = os.path.join('pretrained_models', "Mistral-NeMo-Minitron-Full")

for action in ['train', 'merge']:
    # continue if task already accomplished
    if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
        continue
    if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
        continue

    # load base model & reduce embedding size
    model = tokenizer = None  # free memory
    model, tokenizer = load_unsloth_4bit(base_model)
    keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
    keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

    # set formatting options
    fmt_opts = dict(
        preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        query_beg='I',
        reply_beg='\n+/-=O',
        reply_end='\n' + tokenizer.eos_token,
        lines_sep='\n',
        max_tokens=8192,
    )

    # create lora model
    lora_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head']
    model = FastLanguageModel.get_peft_model(
        model=model,
        target_modules=lora_layers,
        r=256,
        lora_alpha=24,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )
    if action == 'train':
        ckpts = sorted(glob.glob('tmp_output/checkpoint-*'), key=lambda p: int(p.split('-')[-1])) #ADDED
        resume_from = ckpts[-1] if ckpts else None #ADDED

        # choose a representative portion of the entire dataset
        dataset_fraction = 0.25
        n_rearc = max(1, round(644 * dataset_fraction))
        repeat_mix = max(1, round(128 * dataset_fraction))
        
        # load training data
        from dataset_utils import load_fixed_split
        train_set, held_out_set = load_fixed_split(arc_data_path)
        concept_arc = ArcDataset.load_from_neoneye(os.path.join(neoneye_path, 'dataset', 'ConceptARC'))
        mix_datasets = {
            'arceval': train_set.move_test_to_train().repeat(repeat_mix),
            'concept': concept_arc.move_test_to_train().repeat(repeat_mix),
        }
        train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=n_rearc, sizes=[6], seed=42, mix_datasets=mix_datasets)

        # augment data set and transform to list (eventually removing examples to stay below the max. token count)
        train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        train_dataset_augment = train_dataset.augment(**train_aug_opts)
        train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

        # run training
        FastLanguageModel.for_training(model)
        tokenizer.padding_side = 'right'
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=Dataset.from_list(train_dataset_as_list),
            dataset_text_field="text",
            max_seq_length=fmt_opts['max_tokens'],
            packing=False,
            data_collator=InputMaskingDataCollator(
                instruction_template=fmt_opts['query_beg'],
                response_template=fmt_opts['reply_beg'],
                mlm=False,
                tokenizer=tokenizer,
                mask_first_n_examples=1,
            ),
            args=TrainingArguments(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
                group_by_length=True,
                dataloader_num_workers=2,
                dataloader_pin_memory=True,
                warmup_ratio=0.25,
                num_train_epochs=1,
                learning_rate=1e-4,
                embedding_learning_rate=1e-5,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.00,
                lr_scheduler_type='cosine',
                seed=42,
                output_dir='tmp_output',
                save_strategy='steps',   #change from "no"
                save_steps=500,          #ADDED
                save_total_limit=3,      #ADDED
                report_to='none',
            ),
        )
        trainer_stats = unsloth_train(trainer, resume_from_checkpoint=resume_from) #ADDED resume_from_checkpoint
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        save_model_and_tokenizer(f'{save_model_path}-lora', model, tokenizer)
        # === FORCE INTER-LOOP PURGE ===
        print("Training complete. Erasing trainer references from Python memory...")
        del trainer
        del trainer_stats

    if action == 'merge':
        # load peft weights and merge
        load_peft_state(model, f'{save_model_path}-lora')
        model = merge_peft_into_base(model)
        save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)

    print(f"Finished {action}. Flushing VRAM completely...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
