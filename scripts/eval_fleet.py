"""Fleet cross-play: pair every agent with the three SP `final` agents,
which stand in for an existing deployed fleet of machine partners.

Note: FACET saw these SP runs' checkpoints during training (they form its
partner pool), so for FACET this measures retained in-distribution fleet
compatibility, not zero-shot transfer. For SP itself, same-seed pairings are
excluded (those are the self-pair cells in the main table); cross-seed SP
pairing is the classic convention-clash test.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--episodes-per-pair", type=int, default=10)
    ap.add_argument("--models-root", default="models")
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
            return load_bc(os.path.join(
                args.models_root, "bc", f"bc_train_{layout}_s{seed}.pt"))
        return load_actor_critic(os.path.join(
            args.models_root, f"{layout}__{method}__s{seed}", "final.pt"))

    fleet = {s: agent_net("sp", s) for s in [0, 1, 2]}
    out = {"layout": layout, "pairs": []}
    for method in ["bc", "sp", "ppo_bc", "facet"]:
        for seed in [0, 1, 2]:
            net = agent_net(method, seed)
            for fs, fnet in fleet.items():
                if method == "sp" and fs == seed:
                    continue  # that's the self-pair cell
                scores = eval_pair(layout, net, fnet,
                                   episodes=args.episodes_per_pair, env=env)
                out["pairs"].append({"agent": method, "agent_seed": seed,
                                     "fleet_seed": fs, "scores": scores})
            print(f"{layout} {method} s{seed} done", flush=True)

    path = f"results/fleet_{layout}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
