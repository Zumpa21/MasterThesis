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
import json

def main():
    from arc_loader import ArcDataset
    from resilient_inference import inference_run_resilient
    from selection import EvalTool
    from arc_downloader import download_arc_data

    # --- sharding: which slice of the task set this process handles ---
    # usage: python run_evaluation_Nemo-full_sharded_resilient.py <shard_id> <n_shards>
    shard_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_shards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    SPLIT_SEED = 42  # fixed so shard membership is reproducible/reportable

    # input paths
    base_model = 'nvidia/Mistral-NeMo-Minitron-8B-Base'
    arc_data_path = os.path.join('input', 'arc-prize-2024')  # as on kaggle arc prize 2024
    download_arc_data(arc_data_path)

    # distinct output naming -- avoids collision with the base-model script's own shard dirs
    output_path = os.path.join(f'output_evaluation_Nemo_shard{shard_id}of{n_shards}')
    os.makedirs(output_path, exist_ok=True)
    #inference_cache = os.path.join(output_path, 'inference_cache')
    import getpass

    user = getpass.getuser()
    job_id = os.environ.get('SLURM_JOB_ID', 'nojob')
    inference_cache = os.path.join('/tmp', f'{user}_inference_cache_base_{job_id}_shard{shard_id}')
    #inference_cache = os.path.join('/tmp', f'inference_cache_shard{shard_id}')
    submission_file = os.path.join(output_path, 'submission.json')
    failed_tasks_log = os.path.join(output_path, 'failed_tasks.txt')

    # load evaluation dataset, then take this shard's slice
    from dataset_utils import load_fixed_split
    _, held_out_set = load_fixed_split(arc_data_path)
    arc_eval_set = held_out_set.split(n_shards, SPLIT_SEED)[shard_id] if n_shards > 1 else held_out_set

    # NOTE: the model itself is now loaded INSIDE the worker subprocess (see
    # resilient_inference.py), not here -- this avoids holding a second copy of
    # it in the parent process and lets a worker be freely killed/respawned.
    # We only need the tokenizer here (for tokenizer.eos_token in fmt_opts), so
    # load just that, not the full model.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)

    # set formatting options
    fmt_opts = dict(
        preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        query_beg='I',
        reply_beg='\n+/-=O',
        reply_end='\n' + tokenizer.eos_token,
        lines_sep='\n',
        max_tokens=8192,
    )

    # run inference
    infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000)
    infer_dataset = arc_eval_set.repeat(2).augment(**infer_aug_opts)

    #log
    print(f"shard {shard_id}/{n_shards}: {len(infer_dataset.grouped_keys())} grouped tasks")

    eval_tool = EvalTool(n_guesses=2)
    inference_results = inference_run_resilient(
        model_path=base_model,
        fmt_opts=fmt_opts,
        dataset=infer_dataset,
        min_prob=0.1,
        aug_score_opts=infer_aug_opts,
        callback=eval_tool.process_result,
        cache_path=inference_cache,
        failed_tasks_log=failed_tasks_log,
    )

    # write submission
    with open(submission_file, 'w') as f:
        json.dump(arc_eval_set.get_submission(inference_results), f)
    with open(submission_file, 'r') as f:
        print(f"Score for '{submission_file}':", arc_eval_set.validate_submission(json.load(f)))

if __name__ == '__main__':
    main()
