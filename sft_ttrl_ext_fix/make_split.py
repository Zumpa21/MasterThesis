import os
import json
from arc_loader import ArcDataset

# --- fixed protocol for the whole thesis: DO NOT change these values later ---
SPLIT_SEED = 42
N_CHUNKS = 4          # 400 tasks / 4 = 100 tasks per chunk
HELD_OUT_CHUNK = 0    # which chunk is the permanent held-out test set

arc_data_path = os.path.join('input', 'arc-prize-2024')
arc_eval_set = ArcDataset.load_from_json(os.path.join(arc_data_path, 'arc-agi_evaluation_challenges.json'))
arc_eval_set = arc_eval_set.load_solutions(os.path.join(arc_data_path, 'arc-agi_evaluation_solutions.json'))

chunks = arc_eval_set.split(N_CHUNKS, SPLIT_SEED)

held_out_ids = sorted(chunks[HELD_OUT_CHUNK].challenge.keys())
train_ids = sorted(
    k for i, c in enumerate(chunks) if i != HELD_OUT_CHUNK for k in c.challenge.keys()
)

assert set(held_out_ids).isdisjoint(train_ids)
assert len(held_out_ids) + len(train_ids) == len(arc_eval_set.challenge)

with open('held_out_tasks.json', 'w') as f:
    json.dump(held_out_ids, f, indent=2)
with open('train_tasks.json', 'w') as f:
    json.dump(train_ids, f, indent=2)

print(f"held-out: {len(held_out_ids)} tasks -> held_out_tasks.json")
print(f"train mix: {len(train_ids)} tasks -> train_tasks.json")
