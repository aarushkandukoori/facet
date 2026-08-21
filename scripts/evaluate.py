"""Cross-play evaluation for one layout.

Pairs every trained agent (bc / sp / ppo_bc / facet, 3 seeds each) with:
  - H_proxy: BC models trained on the held-out TEST human split (3 seeds)
  - itself (self-pair sanity check / SP upper bound)

Writes results/eval_<layout>.json with every episode score.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--episodes-per-pair", type=int, default=10)
    ap.add_argument("--self-episodes", type=int, default=10)
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(2)
    from facet.envtools import make_env
    from facet.eval import eval_pair
    from facet.nets import load_actor_critic, load_bc

    layout = args.layout
    env = make_env(layout)

    def agent_net(method, seed):
        if method == "bc":
            return load_bc(
                os.path.join(args.models_root, "bc",
                             f"bc_train_{layout}_s{seed}.pt")
            )
        return load_actor_critic(
            os.path.join(args.models_root,
                         f"{layout}__{method}__s{seed}", "final.pt")
        )

    proxies = {
        s: load_bc(os.path.join(args.models_root, "bc",
                                f"bc_test_{layout}_s{s}.pt"))
        for s in [0, 1, 2]
    }

    results = {"layout": layout, "episodes_per_pair": args.episodes_per_pair,
               "pairs": []}
    methods = ["bc", "sp", "ppo_bc", "facet"]
    for method in methods:
        for seed in [0, 1, 2]:
            try:
                net = agent_net(method, seed)
            except FileNotFoundError:
                print(f"skip {method} s{seed} (no model)", flush=True)
                continue
            # vs human proxies
            for ps, proxy in proxies.items():
                scores = eval_pair(layout, net, proxy,
                                   episodes=args.episodes_per_pair, env=env)
                results["pairs"].append(
                    {"agent": method, "agent_seed": seed,
                     "partner": "h_proxy", "partner_seed": ps,
                     "scores": scores}
                )
            # self-pair
            scores = eval_pair(layout, net, net,
                               episodes=args.self_episodes, env=env)
            results["pairs"].append(
                {"agent": method, "agent_seed": seed,
                 "partner": "self", "partner_seed": seed, "scores": scores}
            )
            mean_h = np.mean(
                [np.mean(p["scores"]) for p in results["pairs"]
                 if p["agent"] == method and p["agent_seed"] == seed
                 and p["partner"] == "h_proxy"]
            )
            print(f"{layout} {method} s{seed}: vs H_proxy {mean_h:.1f}",
                  flush=True)

    out = args.out or f"results/eval_{layout}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
