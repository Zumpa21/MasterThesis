import os
import json
from arc_loader import ArcDataset


def load_fixed_split(arc_data_path, split_dir='.'):
    """Returns (train_set, held_out_set) using the fixed, pre-computed split
    saved by make_split.py. Same split for every experiment in the thesis."""
    with open(os.path.join(split_dir, 'train_tasks.json')) as f:
        train_ids = set(json.load(f))
    with open(os.path.join(split_dir, 'held_out_tasks.json')) as f:
        held_out_ids = set(json.load(f))

    full = ArcDataset.load_from_json(os.path.join(arc_data_path, 'arc-agi_evaluation_challenges.json'))
    full = full.load_solutions(os.path.join(arc_data_path, 'arc-agi_evaluation_solutions.json'))

    train_challenge = {k: v for k, v in full.challenge.items() if k in train_ids}
    held_out_challenge = {k: v for k, v in full.challenge.items() if k in held_out_ids}

    train_set = ArcDataset(challenge=train_challenge, solutions=full.solutions, is_orig=True)
    held_out_set = ArcDataset(challenge=held_out_challenge, solutions=full.solutions, is_orig=True)

    assert len(train_set.challenge) == len(train_ids)
    assert len(held_out_set.challenge) == len(held_out_ids)
    return train_set, held_out_set
