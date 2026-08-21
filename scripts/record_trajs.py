"""Record demo trajectories (full state dicts) for the GitHub Pages demo.

For each layout, records episodes of the best FACET agent paired with a
held-out human proxy, plus a self-play pair for contrast.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "vendor/overcooked_ai/src"),
)

import numpy as np


def grid_of(layout):
    from facet.envtools import make_env

    env = make_env(layout)
    return ["".join(row) for row in env.mdp.terrain_mtx], env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="+", required=True)
    ap.add_argument("--out", default="docs/data/trajs.json")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--best", default="results/best_seeds.json",
                    help="json: layout -> {method: best_seed}")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(2)
    from facet.eval import rollout
    from facet.nets import load_actor_critic, load_bc

    best = (
        json.load(open(args.best)) if os.path.exists(args.best) else {}
    )
    out = {}
    for layout in args.layouts:
        terrain, env = grid_of(layout)
        b = best.get(layout, {})
        facet = load_actor_critic(
            os.path.join(args.models_root,
                         f"{layout}__facet__s{b.get('facet', 0)}", "final.pt"))
        ppobc = load_actor_critic(
            os.path.join(args.models_root,
                         f"{layout}__ppo_bc__s{b.get('ppo_bc', 0)}",
                         "final.pt"))
        sp = load_actor_critic(
            os.path.join(args.models_root,
                         f"{layout}__sp__s{b.get('sp', 0)}", "final.pt"))
        proxy = load_bc(
            os.path.join(args.models_root, "bc", f"bc_test_{layout}_s0.pt"))

        episodes = []
        for tag, (n0, n1) in {
            "selfplay+selfplay": (sp, sp),
            "selfplay+human_proxy": (sp, proxy),
            "ppo_bc+human_proxy": (ppobc, proxy),
            "facet+human_proxy": (facet, proxy),
        }.items():
            best_ep = None
            for _ in range(args.episodes):
                score, traj = rollout(env, n0, n1, record=True)
                if best_ep is None or score > best_ep["score"]:
                    best_ep = {"pair": tag, "score": score,
                               "states": traj["states"],
                               "actions": traj["actions"],
                               "rewards": traj["rewards"]}
            episodes.append(best_ep)
            print(layout, tag, "best score", best_ep["score"], flush=True)
        out[layout] = {"terrain": terrain, "episodes": episodes}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
