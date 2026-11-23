import os
import json
import sys
from arc_loader import ArcDataset

n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 2
run_dir = sys.argv[2] if len(sys.argv) > 2 else '.'   # runs/<RUN_TAG> (default: current folder, as before)

arc_data_path = os.path.join('input', 'arc-prize-2024')
from dataset_utils import load_fixed_split
_, arc_eval_set_full = load_fixed_split(arc_data_path)  # 100 held-out tasks

combined = {}
for shard_id in range(n_shards):
    shard_file = os.path.join(run_dir, f'output_evaluation_Nemo_shard{shard_id}of{n_shards}', 'submission.json')
    with open(shard_file) as f:
        shard_submission = json.load(f)
    overlap = set(combined) & set(shard_submission)
    assert not overlap, f"shards are not disjoint, overlapping keys: {overlap}"
    combined.update(shard_submission)

missing = set(arc_eval_set_full.challenge) - set(combined)
if missing:
    print(f"WARNING: {len(missing)} tasks missing from combined submission (shard job likely still running or failed): {sorted(missing)[:10]}...")

out_path = os.path.join(run_dir, 'output_evaluation_Nemo_combined_submission.json')
with open(out_path, 'w') as f:
    json.dump(combined, f)

print(f"Combined {len(combined)}/{len(arc_eval_set_full.challenge)} tasks from {n_shards} shards -> {out_path}")
print("Score:", arc_eval_set_full.validate_submission(combined))
