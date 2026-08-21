"""CLI for PPO runs: self-play, PPO_BC, or FACET.

Examples:
  python scripts/train_ppo.py --layout cramped_room --method sp --seed 0
  python scripts/train_ppo.py --layout cramped_room --method ppo_bc --seed 0
  python scripts/train_ppo.py --layout cramped_room --method facet --seed 0
"""

import argparse
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
    ap.add_argument("--method", required=True,
                    choices=["sp", "ppo_bc", "facet"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-workers", type=int, default=5)
    ap.add_argument("--envs-per-worker", type=int, default=4)
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--bc-dir", default="models/bc")
    args = ap.parse_args()

    from facet.partners import (
        Partner,
        PartnerPool,
        load_bc_partners,
        load_ckpt_partners,
    )
    from facet.ppo import PPOConfig, train

    run_id = f"{args.layout}__{args.method}__s{args.seed}"
    out_dir = os.path.join(args.models_root, run_id)
    cfg = PPOConfig(
        total_steps=args.steps,
        n_workers=args.n_workers,
        envs_per_worker=args.envs_per_worker,
    )

    partner, pool = None, None
    mode = "sp"
    if args.method == "ppo_bc":
        mode = "partner"
        bcs = load_bc_partners(args.bc_dir, args.layout, split="train")
        assert bcs, f"no BC models for {args.layout}"
        # one human model per learner seed, matching Carroll et al.'s PPO_BC
        partner = bcs[args.seed % len(bcs)]
        print(f"[{run_id}] partner = {partner.name}", flush=True)
    elif args.method == "facet":
        mode = "partner"
        import numpy as np

        bcs = load_bc_partners(args.bc_dir, args.layout, split="train")
        ckpts = load_ckpt_partners(args.models_root, args.layout, method="sp")
        assert bcs and ckpts, f"missing pool members for {args.layout}"
        pool = PartnerPool(
            bcs + ckpts, rng=np.random.default_rng(args.seed)
        )
        print(f"[{run_id}] pool size = {len(pool)} "
              f"({len(bcs)} BC + {len(ckpts)} ckpt)", flush=True)

    print(f"[{run_id}] starting: {args.steps} steps, "
          f"{cfg.n_workers}x{cfg.envs_per_worker} envs", flush=True)
    train(args.layout, mode, args.seed, out_dir, cfg,
          partner=partner, pool=pool)
    print(f"[{run_id}] DONE", flush=True)


if __name__ == "__main__":
    main()
