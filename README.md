# Can Intrinsic Motivation Help LLMs Adapt? An Empirical Study on ARC-AGI-1

Code and results of the Master's thesis of Daniele Sindaco (University of Bologna, Master's Degree in Computer Engineering, a.y. 2025-2026).
Supervisor: Prof. Mirco Musolesi. Co-supervisor: Dr. Giorgio Franceschelli.

The thesis studies whether intrinsic rewards (confidence and curiosity) help a language model adapt to new ARC-AGI-1 tasks when it is trained with reinforcement learning at test time. Starting from a replication of the pipeline of [the ARChitects](https://github.com/da-fr/arc-prize-2024) (winners of the ARC Prize 2024, Mistral-NeMo-Minitron-8B-Base), the supervised test-time fine-tuning is replaced with **test-time GRPO**, trained on a leave-one-out (LOO) dataset built from the demonstration pairs of each test task.

**Main findings.** The intrinsic rewards add a learning signal and make the answers more varied, but do not increase the number of solved tasks beyond the variability between seeds. Used alone, they make the model more confident in its mistakes, and without a preliminary supervised stage no reward teaches the tasks. Supervised fine-tuning on the same data remains clearly more effective than test-time GRPO, with more than ten times less training time.

## Results

All results on the same 100 ARC-AGI-1 public evaluation tasks (103 test inputs), split into 4 shards of 25 tasks. Score and correct found are out of 100 (correct found: the solution is among the generated candidates, before the top-2 selection). Zero-grad: share of GRPO steps in which all the answers of a group got the same reward. Reward configurations R1-R12 are described below.

| Run | Start | G | Seed | Variation | Score | Correct found | Zero-grad (%) | Folder |
|---|---|---|---|---|---|---|---|---|
| Raw model | raw | - | - | no training | 0 | 0 | - | `base_notrained` |
| Raw + TTFT | raw | - | - | test-time fine-tuning | 24.5 | 31.5 | - | `base_ttft` |
| Stage-1 | Stage-1 | - | - | no test-time training | 37 | 42.5 | - | `orig_0_25_dataset_finetuned_only` |
| Replication | Stage-1 | - | - | ARChitects' test-time fine-tuning | 61 | 64 | - | `orig_0_25_dataset` |
| SFT control | Stage-1 | - | - | held-out output, lr 1e-4 | 49.5 | 57 | - | `sft_control` |
| SFT control | Stage-1 | - | - | held-out output, lr 1e-5 | 50 | 54 | - | `sft_control` |
| SFT control | Stage-1 | - | - | all outputs except the first, lr 1e-4 | 58.5 | 64 | - | `sft_control` |
| R1 | Stage-1 | 4 | 42 | - | 40 | 43 | 91.0 | `sft_ttrl_ext_fix` |
| R1 | Stage-1 | 4 | 43 | - | 36.5 | 43 | 91.0 | `sft_ttrl_ext_fix` |
| R1 | Stage-1 | 4 | 42 | lr 1e-4 | 17.5 | 20.5 | 91.0 | `sft_ttrl_ext_fix` |
| R2 | Stage-1 | 4 | 42 | - | 39 | 46 | 47.3 | `sft_ttrl_ext_fix` |
| R2 | Stage-1 | 4 | 42 | lr 2e-5 | 37.5 | 44 | 49.5 | `sft_ttrl_ext_fix` |
| R3 | Stage-1 | 4 | 42 / 43 | - | 40.5 / 38 | 46.5 / 44.5 | 37.8 / 34.8 | `final_test` |
| R3 | Stage-1 | 8 | 42 / 43 | - | 37.5 / 45 | 46.5 / 49 | 32.0 / 30.2 | `final_test` |
| R4 | Stage-1 | 4 | 42 | - | 37 | 39 | 67.5 | `sft_ttrl_int_fix` |
| R5 | Stage-1 | 4 | 42 | - | 37 | 41.5 | 33.4 | `sft_ttrl_int_fix` |
| R6 | Stage-1 | 4 | 42 | - | 35 | 39 | 23.6 | `sft_ttrl_int_fix` |
| R7 | Stage-1 | 4 | 42 | - | 38 | 42 | 23.6 | `sft_ttrl_int_fix` |
| R8 | Stage-1 | 4 | 42 | - | 38.5 | 42.5 | 23.0 | `sft_ttrl_int_fix` |
| R9 | Stage-1 | 4 | 42 | RND inactive (bug) | 39.5 | 46.5 | 26.8 | `sft_ttrl_int_fix` |
| R10 | Stage-1 | 4 | 42 | - | 41 | 45.5 | 26.1 | `sft_ttrl_int_fix` |
| R11 | Stage-1 | 4 | 42 / 43 | - | 41 / 39 | 47.5 / 43.5 | 30.4 / 29.0 | `final_test` |
| R11 | Stage-1 | 4 | 42 | alpha 0.1 / 0.5 | 37 / 39.5 | 43 / 46 | 33.5 / 27.4 | `final_test` |
| R11 | Stage-1 | 4 | 42 | beta 0.1 / 0.4 | 36 / 37.5 | 42.5 / 45 | 32.0 / 30.6 | `final_test` |
| R11 | Stage-1 | 8 | 42 / 43 | - | 41.5 / 40.5 | 50.5 / 46 | 26.1 / 28.1 | `final_test` |
| R12 | Stage-1 | 4 | 42 | intrinsic only | 30 | 36.5 | 45.5 | `sft_ttrl_intonly` |
| R2 | raw | 4 | 42 | - | 3 | 4 | 62.9 | `base_grpo_ext` |
| R10 | raw | 4 | 42 | - | 0 | 0 | 15.3 | `base_grpo_int` |
| R3 | raw | 4 | 42 | - | 3.5 | 5.5 | 41.4 | `final_test` |

Notes: the raw model in `base_notrained` and `base_ttft` keeps its full vocabulary, and a few tasks exceeded the 2-hour inference limit (listed in `failed_tasks.txt`), so those scores are lower bounds. In `sft_control`, the run with `min_remaining=3` (only 38 of the 100 tasks trained) is an early version and is not used in the thesis.

### Reward configurations

| ID | Extrinsic | Confidence | Curiosity | Version | Name in the log files |
|---|---|---|---|---|---|
| R1 | binary | - | - | - | `eval_ext_fix_...` |
| R2 | shaped (partial credit for right-size grids) | - | - | - | `eval_shaped_ext_fix_...`, raw: `eval_base_grpo_ext_shaped_...` |
| R3 | signed accuracy q | - | - | - | `eval_q_sft_ext_...`, raw: `eval_q_raw_ext_...` |
| R4 | binary | alpha * p_bar on correct answers | within-group novelty, only when all answers are wrong | v1 | `eval_w_int_fix_..._minrem1_...` |
| R5 | shaped | as R4 | as R4 | v1 | `eval_shaped_w_int_fix_...` |
| R6 | shaped | signed mean token probability | within-group novelty on every wrong answer | v1-fixed | `eval_shaped_v1fixed_...` |
| R7 | shaped | calibrated (CDE) | count-based novelty beta/sqrt(N) | v2 | `eval_shaped_v2_...` |
| R8 | shaped | consistency across views | as R7 | v3 | `eval_shaped_v3_...` |
| R9 | shaped | as R8 | RND, percentile normalization (inactive after ~20 steps) | v4 | `eval_shaped_v4_...` |
| R10 | shaped | as R8 | RND, running-mean normalization | v4b | `eval_shaped_v4b_...`, raw: `eval_base_grpo_int_v4b_...` |
| R11 | signed accuracy q | as R8 | as R10, multiplied by (1 - q) / 2 | v4b, gated | `eval_q_sft_ext_int_...`, scales: `eval_q_int_a01/a05/b01/b04_...` |
| R12 | none (logged only) | + alpha * consistency on every readable answer | RND on every readable answer | v4b, label-free | `eval_sft_ttrl_intonly_v4b_labelfree_...` |

Runs with G = 8 have `g8` in the log name, runs with seed 43 have `s43` (or `difseed` for R1).

Default scales: alpha = 0.3, beta = 0.2 (R4: 0.5 / 0.5).

## Repository layout

One folder per group of experiments. Each folder contains the code (`.py`) and the SLURM scripts (`.sbatch`) used for those runs, the run logs (`.log`, with training diagnostics and per-task evaluation), the task split (`train_tasks.json`, `held_out_tasks.json`) and the submissions (`submission.json` per shard, `*_combined_submission.json`). GRPO runs with their own run folder are in `runs/<run name>/`.

| Folder | Content |
|---|---|
| `orig_0_25_dataset` | Stage-1 SFT (a quarter of the ARChitects' data) and replication with test-time fine-tuning |
| `orig_0_25_dataset_finetuned_only` | Stage-1 model without test-time training |
| `base_notrained`, `base_ttft` | raw base model, without training and with test-time fine-tuning |
| `sft_control` | SFT on the same leave-one-out data used by GRPO |
| `sft_ttrl_ext_fix` | test-time GRPO, binary and shaped extrinsic rewards (R1, R2) |
| `sft_ttrl_int_fix` | test-time GRPO, extrinsic + intrinsic rewards (R4-R10) |
| `sft_ttrl_intonly` | test-time GRPO, intrinsic rewards only (R12) |
| `base_grpo_ext`, `base_grpo_int` | test-time GRPO from the raw base model (R2, R10) |
| `final_test` | final script: signed accuracy reward with and without intrinsic terms (R3, R11), G = 8, sensitivity of the intrinsic scales, R3 from the raw model |

Shared helpers (`arc_loader.py`, `model_tools.py`, `inference_tools.py`, `selection.py`) come from the ARChitects' code. Each folder keeps its own copy, as used for its runs.

## Running

1. Environment: Python 3.11, PyTorch 2.10 (CUDA 12.8), Transformers 4.57, TRL 0.24, PEFT 0.18, Unsloth 2026.6 (full list in `requirements.txt`). All runs used one NVIDIA L40 (48 GB) with SLURM.
2. Replace the placeholders in the `.sbatch` and `.py` files with your own paths: `/path/to/MasterThesis` (this repository), `/path/to/venv` (virtual environment), `/path/to/cache` (Hugging Face cache). The SLURM partition (`l40`) and the e-mail address are also specific to the original cluster.
3. Data: the ARC-AGI-1 tasks are downloaded by `arc_downloader.py`. Stage-1 training also uses [RE-ARC](https://github.com/michaelhodel/re-arc) as in the ARChitects' pipeline.
4. Stage-1: `orig_0_25_dataset/training/run_finetuning_Nemo-full.sbatch`, then the evaluation script in the same folder.
5. Test-time GRPO (final version): `final_test/run_grpo_final.sbatch` with `run_evaluation_Nemo-full_grpo_final.py`. The run is chosen with environment variables, described in the comments at the top of the sbatch file: the starting model (`GRPO_BASE=raw|sft`), the reward mode (`GRPO_REWARD=ext|ext_int|int_only`), the extrinsic reward, the intrinsic scales, the number of steps (`GRPO_MAX_STEPS`, default 700) and the shards (`SHARDS`, default all four). For example:

   ```bash
   sbatch --job-name=sft_ext_int --export=ALL,GRPO_BASE=sft,GRPO_REWARD=ext_int final_test/run_grpo_final.sbatch
   ```

   Each run works in its own folder `runs/<run name>/`, with a copy of the code, and merges the four shard submissions at the end.

Models, adapters and checkpoints are not included (several GB each).

## License and credits

Released under the Apache License 2.0 (see `LICENSE`). The pipeline, the data loading, the inference and the candidate selection are based on the code of [The LLM ARChitect](https://github.com/da-fr/arc-prize-2024) by Daniel Franzen and Jan Disselhoff (Apache 2.0).

### Run folders in `final_test/runs`

Folder names follow `final_<start>_<reward>_qsigned_a<alpha>_b<beta>_ms<steps>[_seed43][_g8]`: `sft` or `raw` is the starting model, `ext` is the extrinsic reward only (R3), `ext_int` adds the intrinsic terms (R11), `a` and `b` are the confidence and curiosity scales, `ms` the number of training steps, and `seed43` / `g8` mark the second seed and groups of G = 8 answers. Without these suffixes the run uses seed 42 and G = 4.
