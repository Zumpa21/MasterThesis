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

from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
import numpy as np
import hashlib

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import (load_unsloth_4bit, save_model_and_tokenizer, load_peft_state,
                              is_peft_model, keep_single_char_tokens, merge_peft_into_base)
from resilient_inference import inference_run_resilient
from selection import EvalTool
from arc_downloader import download_arc_data

from collections import defaultdict

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

def partial_credit(pred, truth, query_input):
    """Partial credit for a grid with the RIGHT SHAPE but wrong content, in [0, 1].
    = fraction of the gap between 'copy the input' and 'correct answer' that the prediction closes:
        q = (acc(pred) - acc(copy input)) / (1 - acc(copy input)), clipped to [0, 1]
    Copying the input gives 0 (no reward for not solving), a perfect grid gives 1.
    If input and output have different shapes, copying is impossible and q = plain cell accuracy."""
    truth = np.asarray(truth)
    acc = float((pred == truth).mean())
    query_input = np.asarray(query_input)
    base = float((query_input == truth).mean()) if query_input.shape == truth.shape else 0.0
    if base >= 1.0:
        return 0.0
    return float(np.clip((acc - base) / (1.0 - base), 0.0, 1.0))

# EXPERIMENT 5: GRPO on the RAW base model, shaped extrinsic + intrinsic v4b
# (consistency-calibrated confidence + RND novelty on wrong answers). Same settings as the Stage-1 v4b run (exp 7).


def main():
    class DiagnosticCallback(TrainerCallback):
        def on_log(self, args, state, control, **kwargs):
            mean = lambda v: sum(v) / len(v) if v else float('nan')
            if reward_stats['correct']:
                # training-time behaviour, logged in every mode (also when the extrinsic weight is 0)
                print(f"[diagnostic] step {state.global_step}: "
                      f"correct={mean(reward_stats['correct']):.2%} readable={mean(reward_stats['readable']):.2%} "
                      f"right size={mean(reward_stats['right_size']):.2%} copy of input={mean(reward_stats['copy']):.2%} "
                      f"truncated={mean(reward_stats['truncated']):.2%}")
            if reward_stats['conf_correct'] or reward_stats['conf_wrong']:
                print(f"[diagnostic] step {state.global_step}: "
                      f"consistency c correct={mean(reward_stats['conf_correct']):.3f} wrong={mean(reward_stats['conf_wrong']):.3f} | "
                      f"curiosity mean={mean(reward_stats['cur_bonus']):.3f}, first-visit wrong grids={mean(reward_stats['cur_first']):.2%}, "
                      f"rnd error mean={mean(reward_stats['rnd_err']):.2e}, rnd resets={rnd_state['resets']} | "
                      f"distinct (example, grid) states so far={len(visit_counts)}")
            for k in ('conf_correct', 'conf_wrong', 'cur_bonus', 'cur_first', 'rnd_err',
                      'correct', 'readable', 'right_size', 'copy', 'truncated'):
                reward_stats[k].clear()

        def on_save(self, args, state, control, **kwargs):
            print(f"[diagnostic] step {state.global_step}, epoch {state.epoch:.2f}")

    num_generations=4
    temp=0.6
    beta=0.04
    batch_size=1
    lr=1e-5
    ms=700   # 50 for a quick test
    shaping_scale=1.4   # partial credit max = -0.9 + 1.4 = +0.5, always below a correct answer (~+0.85)
    conf_alpha=0.3      # confidence/calibration bonus size (CDE-style, Dai et al.)
    n_views=2           # other augmented views used to measure consistency (cost: n_views forward passes per distinct answer)
    cur_beta=0.2        # RND novelty bonus size (Burda et al.; as in IMAGINE only on wrong answers), bonus in [0, cur_beta]
    rnd_lr=1e-4         # learning rate of the small RND predictor network
    # worst case for a WRONG answer: +0.5 (shaping) + 0.2 (novelty) + 0 (confidence) = +0.7 < correct (~+0.85)
    max_com_length=1000

    print("[parameters] experiment 5 (base_grpo_int)")
    print(f"[parameters] num_generations: {num_generations}")
    print(f"[parameters] temperature: {temp}")
    print(f"[parameters] beta: {beta}")
    print(f"[parameters] per_device_train_batch_size: {batch_size}")
    print(f"[parameters] learning_rate: {lr}")
    print(f"[parameters] max_steps: {ms}")
    print(f"[parameters] reward: SHAPED extrinsic, shaping_scale={shaping_scale}")
    print(f"[parameters] intrinsic v4b: consistency-calibrated confidence (alpha={conf_alpha}, n_views={n_views}) + RND novelty on wrong answers (beta={cur_beta}, rnd_lr={rnd_lr})")
    print(f"[parameters] max_completion_length: {max_com_length}")

    # --- sharding: identical convention/seed to the base-model script ---
    shard_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_shards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    SPLIT_SEED = 42  # same seed as base-model script -> identical shard membership across both experiments

    # STARTING model. Raw base = made by make_raw_base_shrunk.py in this folder (same vocabulary reduction as Stage-1, no training)
    merged_stage1_path = '/path/to/MasterThesis/base_grpo_int/pretrained_models/Mistral-NeMo-Minitron-Raw-shrunk'
    arc_data_path = os.path.join('input', 'arc-prize-2024')
    download_arc_data(arc_data_path)

    # distinct output naming -- avoids collision with the base-model script's own shard dirs
    # each run writes in its own folder runs/<RUN_TAG>/ (set in the sbatch) -> runs can go in parallel
    run_tag = os.environ.get('RUN_TAG', '')
    run_dir = os.path.join('runs', run_tag) if run_tag else '.'
    output_path = os.path.join(run_dir, f'output_evaluation_Nemo_shard{shard_id}of{n_shards}')
    os.makedirs(output_path, exist_ok=True)
    print(f"[setup] run_dir={run_dir}")
    #inference_cache = os.path.join(output_path, 'inference_cache')
    #inference_cache = os.path.join('/tmp', f'inference_cache_shard{shard_id}')
    import getpass

    user = getpass.getuser()
    job_id = os.environ.get('SLURM_JOB_ID', 'nojob')
    inference_cache = os.path.join('/tmp', f'{user}_inference_cache_base_grpo_int_{job_id}_shard{shard_id}')

    submission_file = os.path.join(output_path, 'submission.json')
    failed_tasks_log = os.path.join(output_path, 'failed_tasks.txt')

    from dataset_utils import load_fixed_split
    _, held_out_set = load_fixed_split(arc_data_path)
    arc_eval_set = held_out_set.split(n_shards, SPLIT_SEED)[shard_id] if n_shards > 1 else held_out_set

    # FIX 2: we save ONLY the LoRA adapter (no merge). Merging into the 4-bit model rounds the update away.
    adapter_path = os.path.join(output_path, 'grpo_adapter')
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

        train_aug_opts = dict(tp=True, rt=True, perm=False, shfl_ex=False, seed=0)
        loo_set = build_loo_dataset(arc_eval_set.remove_test_data(), min_remaining=1).repeat(n=3, seed=0).augment(**train_aug_opts)

        computed_max_len = loo_set.max_new_tokens(**fmt_opts)
        print(f"Data-driven max completion length needed: {computed_max_len}")

        rows = []
        for key in loo_set.keys:
            key, fmt = loo_set.get_task(key, reply=None, len_name='input', **fmt_opts)
            rows.append(dict(
                prompt=fmt['input'],
                task_key=key,   # FIX 1: augmented key (e.g. 'xxxx-loo0_0.tp.rt'); decode(key=...) undoes the augmentation
            ))

        computed_max_prompt_len = max(len(tokenizer(r['prompt'])['input_ids']) for r in rows)
        print(f"Data-driven max prompt length needed: {computed_max_prompt_len}")

        def check(completion, key):
            # FIX 1: decode WITH the augmented key -> the prediction is rotated back to the original
            # orientation and compared with the right solution (same as evaluation does)
            pred, correct, info = loo_set.decode(completion, fmt_opts['lines_sep'], key=key)
            return pred, (correct is True), info

        def extrinsic_reward_fn(prompts, completions, task_key, **kwargs):
            rewards = []
            for completion, key in zip(completions, task_key):
                pred, is_correct, info = check(completion, key)
                print(f"completion: {completion}, pred: {pred}, info: {info}, key: {key}")
                len_token = len(tokenizer(completion)['input_ids'])
                if pred is None:
                    print(f"None at length {len_token}")
                # behaviour statistics (logged in every mode)
                _bk, _rn = loo_set.get_base_key_and_reply_num(key)
                _truth = np.asarray(loo_set.solutions[_bk][_rn])
                _qin = np.asarray(loo_set.challenge[_bk]['test'][_rn]['input'])
                reward_stats['correct'].append(bool(is_correct))
                reward_stats['readable'].append(pred is not None)
                reward_stats['right_size'].append(pred is not None and pred.shape == _truth.shape)
                reward_stats['copy'].append(pred is not None and pred.shape == _qin.shape and bool((pred == _qin).all()))
                reward_stats['truncated'].append(len_token >= max_com_length - 10)
                if is_correct:
                    length_penalty = 0.0002 * len_token  # small, scales with length
                    rewards.append(1.0 - length_penalty)
                elif pred is None and len_token >= 990:
                    rewards.append(-1.2)  # slightly worse than a normal wrong answer — specifically discourages truncation
                elif pred is None:
                    rewards.append(-1.0)
                else:
                    # SHAPING: wrong answer that can be read as a grid
                    base_key, reply_num = loo_set.get_base_key_and_reply_num(key)
                    truth = np.asarray(loo_set.solutions[base_key][reply_num])
                    if pred.shape != truth.shape:
                        rewards.append(-0.9)                                   # wrong size: as before
                    else:
                        query_input = loo_set.challenge[base_key]['test'][reply_num]['input']
                        q = partial_credit(pred, truth, query_input)          # 0 = like copying input, 1 = perfect
                        rewards.append(-0.9 + shaping_scale * q)               # in [-0.9, -0.9 + shaping_scale]
                        print(f"[shaping] q={q:.3f} reward={rewards[-1]:.3f} key={key}")
            return rewards

        import torch   # must be imported HERE: torch is used inside mean_token_prob() during training
        from inference_tools import calc_score
        reward_stats = defaultdict(list)

        # ------------------------------------------------------------------------------------------
        # INTRINSIC REWARDS v2
        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------
        # CONSISTENCY ACROSS AUGMENTATIONS ("program consistency")
        # ------------------------------------------------------------------------------------------
        D8_VIEWS = [(tp, rt) for tp in (0, 1) for rt in range(4)]   # the 8 symmetries already used in training

        def view_of(key):
            parts = key.split('.')[1:]
            return (parts.count('tp') % 2, parts.count('rt') % 4)

        def consistency(key, pred):
            """How much the model still believes in its answer when the SAME task is shown from
            n_views OTHER points of view (transposed / rotated). pred is in the original orientation;
            get_task(reply=pred) transforms it into each view, exactly like the ARChitects' augmented scoring.
            Returns the mean per-token probability over those views, in (0, 1]."""
            key_num = key.split('.')[0]
            others = [v for v in D8_VIEWS if v != view_of(key)]
            seed = int(hashlib.md5((key + pred.tobytes().hex()).encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            chosen = [others[i] for i in rng.choice(len(others), size=n_views, replace=False)]
            probs = []
            for tp, rt in chosen:
                view_key = key_num + '.tp' * tp + '.rt' * rt
                _, fmt = loo_set.get_task(view_key, reply=pred, len_name='input', **fmt_opts)
                with torch.no_grad():
                    nll = calc_score(input=fmt['input'], reply=fmt['reply'], model_tok=(model, tokenizer))
                n_tok = max(len(tokenizer(fmt['reply'])['input_ids']), 1)
                probs.append(float(np.exp(-nll / n_tok)))
            return float(np.mean(probs))

        def confidence_reward_fn(prompts, completions, task_key, **kwargs):
            """CONFIDENCE v3 = calibration with augmentation CONSISTENCY c instead of p_bar (v2):
               correct answer -> + alpha * (1 - c) : reward correct answers the model does not yet hold in every view
               wrong answer   -> - alpha * c       : penalise mistakes the model holds in every view
            Only change from v2: the confidence measure (p_bar of one view -> c over other views)."""
            rewards, cache = [], {}
            for completion, key in zip(completions, task_key):
                pred, correct, _ = check(completion, key)
                if pred is None:
                    rewards.append(0.0)
                    continue
                ck = (key, pred.tobytes(), pred.shape)
                if ck not in cache:                      # identical answers in a group are scored once
                    cache[ck] = consistency(key, pred)
                c = cache[ck]
                reward_stats['conf_correct' if correct else 'conf_wrong'].append(c)   # logging only
                if correct:
                    rewards.append(conf_alpha * (1.0 - c))
                else:
                    rewards.append(-conf_alpha * c)
            return rewards

        # ------------------------------------------------------------------------------------------
        # CURIOSITY v4: Random Network Distillation (Burda et al.) on (query input, answer grid)
        # ------------------------------------------------------------------------------------------
        import torch.nn as nn
        N_CH = 11                      # 10 colours + 1 "outside the grid" channel, grids padded to 30x30

        def grid_onehot(g):
            g = np.asarray(g)
            x = np.zeros((N_CH, 30, 30), dtype=np.float32)
            x[10] = 1.0
            h, w = g.shape
            x[10, :h, :w] = 0.0
            x[g, np.arange(h)[:, None], np.arange(w)[None, :]] = 1.0
            return x

        def rnd_net(extra_layer):
            layers = [nn.Conv2d(2 * N_CH, 32, 3, stride=2, padding=1), nn.LeakyReLU(),
                      nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.LeakyReLU(),
                      nn.Flatten(), nn.Linear(64 * 8 * 8, 128)]
            if extra_layer:
                layers += [nn.LeakyReLU(), nn.Linear(128, 128)]
            return nn.Sequential(*layers)

        _gen_state = torch.random.get_rng_state()
        torch.manual_seed(1234)                       # same random target in every run / shard
        rnd_target = rnd_net(extra_layer=False).float().eval()
        rnd_pred = rnd_net(extra_layer=True).float()
        torch.random.set_rng_state(_gen_state)        # do not disturb the trainer's random state
        for p_ in rnd_target.parameters():
            p_.requires_grad_(False)
        rnd_opt = torch.optim.Adam(rnd_pred.parameters(), lr=rnd_lr)
        rnd_buffer = []                               # recent visited states, replayed to train the predictor
        rnd_state = {'ema': None, 'resets': 0}        # running (EMA) mean of the error, number of predictor resets

        def rnd_reset():
            # re-initialise the predictor if it ever produces non-finite values (seen in v4: NaN after ~20 steps)
            nonlocal rnd_pred, rnd_opt
            g = torch.random.get_rng_state()
            torch.manual_seed(4321 + rnd_state['resets'])
            rnd_pred = rnd_net(extra_layer=True).float()
            torch.random.set_rng_state(g)
            rnd_opt = torch.optim.Adam(rnd_pred.parameters(), lr=rnd_lr)
            rnd_state['ema'] = None
            rnd_state['resets'] += 1
            print(f"[rnd] predictor reset #{rnd_state['resets']} (non-finite values)")
        visit_counts = defaultdict(int)               # only for logging (distinct states, first visits)

        def rnd_errors(x):
            # plain float32 on CPU, outside any autocast the trainer may have active
            with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
                x = x.float()
                return ((rnd_pred(x) - rnd_target(x)) ** 2).mean(dim=1)

        def curiosity_reward_fn(prompts, completions, task_key, **kwargs):
            """CURIOSITY as RND novelty: a fixed random network (target) and a small network trained to imitate it
               (predictor) on the states the model has produced. Prediction error = novelty: high for answers
               unlike anything seen before, low for answers equal OR SIMILAR to past ones (unlike exact counts).
               state = (query input, predicted grid), both in the original orientation.
               bonus = beta * min(error / (2 * running mean error), 1); only WRONG answers get it (IMAGINE selection).
               The predictor is trained on every readable answer (correct ones too: they are visited states)."""
            checked = [check(c_, k) for c_, k in zip(completions, task_key)]
            xs, idx = [], []
            for i, ((pred, correct, _), key) in enumerate(zip(checked, task_key)):
                if pred is None:
                    continue
                base_key, reply_num = loo_set.get_base_key_and_reply_num(key)
                query_input = loo_set.challenge[base_key]['test'][reply_num]['input']
                xs.append(np.concatenate([grid_onehot(query_input), grid_onehot(pred)]))
                idx.append(i)
                s_ = (base_key, pred.shape, pred.tobytes())
                if not correct:
                    reward_stats['cur_first'].append(visit_counts[s_] == 0)
                visit_counts[s_] += 1
            rewards = [0.0] * len(completions)
            if not xs:
                return rewards
            x = torch.from_numpy(np.stack(xs))
            err = rnd_errors(x).numpy()
            if not np.isfinite(err).all():
                rnd_reset()
                err = rnd_errors(x).numpy()
            # bonus = beta * min(error / (2 * running mean error), 1), in [0, beta].
            # The running mean follows the error as it shrinks during training, so the bonus does not collapse to 0
            # (v4 used a percentile rank against older, larger errors -> almost always 0).
            ema = rnd_state['ema'] if rnd_state['ema'] is not None else float(err.mean())
            for e, i in zip(err, idx):
                reward_stats['rnd_err'].append(float(e))
                if not checked[i][1]:                                  # wrong answer -> novelty bonus
                    rewards[i] = float(cur_beta * min(e / (2.0 * ema + 1e-12), 1.0))
                    reward_stats['cur_bonus'].append(rewards[i])
            rnd_state['ema'] = 0.9 * ema + 0.1 * float(err.mean())   # fast EMA: follows the shrinking error
            # train the predictor on these states + a replay of recent ones (after computing the bonus)
            rnd_buffer.extend(xs)
            del rnd_buffer[:-512]
            replay = [rnd_buffer[j] for j in np.random.choice(len(rnd_buffer), size=min(28, len(rnd_buffer)), replace=False)]
            # the trainer may call reward functions under no_grad / inference_mode: re-enable gradients here
            with torch.inference_mode(False), torch.enable_grad(), torch.autocast(device_type='cpu', enabled=False):
                xb = torch.from_numpy(np.stack(xs + replay)).float()
                for _ in range(2):
                    loss = ((rnd_pred(xb) - rnd_target(xb)) ** 2).mean()
                    if not torch.isfinite(loss):
                        rnd_reset()
                        break
                    rnd_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(rnd_pred.parameters(), 1.0)
                    rnd_opt.step()
            return rewards

        checkpoint_dir = os.path.join(output_path, 'checkpoints')
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(checkpoint_dir) if os.path.isdir(checkpoint_dir) else None

        reward_funcs = [extrinsic_reward_fn, confidence_reward_fn, curiosity_reward_fn]
        reward_weights = [1.0, 1.0, 1.0]   # extrinsic, confidence, curiosity

        FastLanguageModel.for_training(model)
        grpo_trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_funcs,
            train_dataset=Dataset.from_list(rows),
            callbacks=[DiagnosticCallback()],
            args=GRPOConfig(
                num_generations=num_generations,
                temperature=temp,
                beta=beta,
                reward_weights=reward_weights,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=num_generations,
                learning_rate=lr,
                max_steps=ms, #1500,
                max_prompt_length=computed_max_prompt_len + 16,
                max_completion_length=max_com_length, #fmt_opts['max_tokens'] // 4,
                remove_unused_columns=False,          # required — keeps task_key visible to the reward fn
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                output_dir=checkpoint_dir,
                save_strategy="steps",
                save_steps=150,
                save_total_limit=2,
                report_to='none',
                seed=42,
                #use_vllm=True,
                #vllm_mode="colocate",
                #vllm_enable_sleep_mode=False,
            ),
        )
        trainer_stats = grpo_trainer.train(resume_from_checkpoint=last_checkpoint)
        model = grpo_trainer.model

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
        adapter_path=adapter_path,       # ... + GRPO adapter on top (not merged)
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
