# Standalone diagnostic -- NOT a training script.
#
# Question: is the LOO pretext task's difficulty structurally tied to how
# many train pairs remain after holding one out? If accuracy climbs sharply
# with more remaining context, that's evidence the pretext task itself (not
# GRPO's training dynamics) is the bottleneck -- and no amount of extra
# max_steps fixes a task that's missing the example needed to disambiguate
# the rule.
#
# Run this against the Stage-1 MERGED checkpoint (no GRPO at all) first --
# that isolates "is this hard for the base fine-tuned model" from "did GRPO
# training help or hurt". Optionally rerun pointing at a post-GRPO checkpoint
# to see if training changed the per-bucket profile at all.

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["UNSLOTH_DISABLE_TELEMETRY"] = "1"

import sys
from collections import defaultdict
import numpy as np
import torch
from unsloth import FastLanguageModel

from arc_loader import ArcDataset
from model_tools import load_unsloth_4bit
from dataset_utils import load_fixed_split
from arc_downloader import download_arc_data


def build_loo_dataset(arc_set):
    """Same construction as in run_evaluation_Nemo-full.py -- kept identical
    on purpose so bucket membership matches what GRPO actually trained on."""
    loo_challenge = {}
    loo_solutions = {}
    for base_key, task in arc_set.challenge.items():
        train_ex = task['train']
        for i in range(len(train_ex)):
            loo_key = f'{base_key}-loo{i}'
            loo_challenge[loo_key] = {
                'train': train_ex[:i] + train_ex[i + 1:],
                'test': [{'input': train_ex[i]['input']}],
            }
            loo_solutions[loo_key] = [train_ex[i]['output']]
    return ArcDataset(challenge=loo_challenge, solutions=loo_solutions, is_orig=True)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else (
        '/path/to/MasterThesis/orig_0_25_dataset_finetuned_only/'
        'pretrained_models/Mistral-NeMo-Minitron-Full-merged'
    )
    print(f"Loading model from: {model_path}")

    arc_data_path = os.path.join('input', 'arc-prize-2024')
    download_arc_data(arc_data_path)
    _, held_out_set = load_fixed_split(arc_data_path)

    model, tokenizer = load_unsloth_4bit(model_path)
    FastLanguageModel.for_inference(model)

    fmt_opts = dict(
        preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        query_beg='I', reply_beg='\n+/-=O', reply_end='\n' + tokenizer.eos_token,
        lines_sep='\n', max_tokens=8192,
    )

    loo_set = build_loo_dataset(held_out_set.remove_test_data())
    #loo_set = build_loo_dataset(held_out_set.remove_test_data()).repeat(n=3, seed=0).augment(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)

    bucket_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    per_task_log = []

    with torch.no_grad():
        for key in loo_set.keys:
            base_key, reply_num = loo_set.get_base_key_and_reply_num(key)
            n_context = len(loo_set.challenge[base_key]['train'])  # the bucket

            fmt_key, fmt = loo_set.get_task(key, reply=None, len_name='input', **fmt_opts)
            tokenized = tokenizer(fmt['input'], return_tensors='pt').to(model.device)

            gen = model.generate(
                **tokenized,
                max_new_tokens=1000,
                do_sample=False,          # greedy -- this is a capability probe, not a sampling study
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            completion = tokenizer.decode(gen[0, tokenized['input_ids'].shape[1]:])

            pred, _, _ = loo_set.decode(completion, fmt_opts['lines_sep'], key=None)
            true_output = np.array(loo_set.solutions[base_key][reply_num])
            correct = pred is not None and pred.shape == true_output.shape and (pred == true_output).all()

            bucket_results[n_context]['total'] += 1
            bucket_results[n_context]['correct'] += int(correct)
            per_task_log.append((base_key, n_context, correct))

            print(f"[{key}] n_context={n_context} correct={correct}")

    print("\n=== Accuracy by remaining-context size ===")
    for n_context in sorted(bucket_results):
        r = bucket_results[n_context]
        acc = r['correct'] / r['total'] if r['total'] else float('nan')
        print(f"  {n_context} remaining train pairs: {r['correct']:>3}/{r['total']:<3} = {acc:.1%}")


if __name__ == '__main__':
    main()
