"""Build behavior-cloning datasets from the 2019 human-human trajectories.

Each timestep of a human-human game yields two (obs, action) samples, one
per player, using the same 96-dim featurization the RL agents see.
"""

import os

import numpy as np
import pandas as pd

from facet import LAYOUTS
from facet.envtools import make_env, parse_joint_action, parse_state

HUMAN_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../vendor/overcooked_ai/src/human_aware_rl/static/human_data/cleaned",
)

from overcooked_ai_py.mdp.actions import Action


def build_split(split: str, out_dir: str):
    """split in {'train','test'} -> per-layout npz of (obs, act)."""
    df = pd.read_pickle(
        os.path.join(HUMAN_DATA_DIR, f"2019_hh_trials_{split}.pickle")
    )
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for layout in LAYOUTS:
        sub = df[df.layout_name == layout]
        env = make_env(layout)  # configures Recipe before parsing states
        obs_list, act_list = [], []
        for _, row in sub.iterrows():
            state = parse_state(env, row.state)
            joint = parse_joint_action(row.joint_action)
            o0, o1 = env.featurize_state_mdp(state)
            obs_list.append(o0)
            act_list.append(Action.ACTION_TO_INDEX[joint[0]])
            obs_list.append(o1)
            act_list.append(Action.ACTION_TO_INDEX[joint[1]])
        obs = np.asarray(obs_list, dtype=np.float32)
        act = np.asarray(act_list, dtype=np.int64)
        np.savez_compressed(
            os.path.join(out_dir, f"bc_{split}_{layout}.npz"), obs=obs, act=act
        )
        stats[layout] = {
            "timesteps": len(sub),
            "samples": len(act),
            "trials": int(sub.trial_id.nunique()),
            "action_freq": np.bincount(act, minlength=6).tolist(),
        }
        print(f"[{split}] {layout}: {len(sub)} steps -> {len(act)} samples, "
              f"{sub.trial_id.nunique()} games")
    return stats


if __name__ == "__main__":
    import json
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/bc"
    all_stats = {}
    for split in ["train", "test"]:
        all_stats[split] = build_split(split, out)
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)
    print("done")
