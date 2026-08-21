"""Partner pools for best-response training.

FACET's pool mixes behavior-cloned human models with frozen self-play
checkpoints saved at several skill stages. Sampling is prioritized toward
partners the learner currently scores WORST with (min-max normalized
mean return, softmax at temperature tau), mixed with an eps-uniform floor.
"""

import glob
import os

import numpy as np
import torch

from facet.nets import load_actor_critic, load_bc


class Partner:
    def __init__(self, name, net, kind):
        self.name = name
        self.net = net
        self.kind = kind  # 'bc' | 'ckpt'
        self.ema_return = None
        self.episodes = 0

    @torch.no_grad()
    def act(self, obs):
        """obs: (B, 96) tensor -> (B,) action indices (stochastic)."""
        out = self.net.act(obs)
        return out[0] if isinstance(out, tuple) else out


class PartnerPool:
    def __init__(self, partners, tau=0.35, eps=0.25, ema=0.05, rng=None):
        assert partners, "empty partner pool"
        self.partners = partners
        self.tau = tau
        self.eps = eps
        self.ema = ema
        self.rng = rng or np.random.default_rng(0)

    def __len__(self):
        return len(self.partners)

    def _priorities(self):
        returns = np.array(
            [
                p.ema_return if p.ema_return is not None else -np.inf
                for p in self.partners
            ]
        )
        seen = np.isfinite(returns)
        if seen.any():
            lo, hi = returns[seen].min(), returns[seen].max()
            span = max(hi - lo, 1.0)
            norm = np.where(seen, (returns - lo) / span, 0.0)  # unseen -> hardest
        else:
            norm = np.zeros(len(returns))
        logits = -norm / self.tau
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return self.eps / len(p) + (1 - self.eps) * p

    def sample(self):
        p = self._priorities()
        return int(self.rng.choice(len(self.partners), p=p / p.sum()))

    def update(self, idx, sparse_return):
        pt = self.partners[idx]
        pt.episodes += 1
        if pt.ema_return is None:
            pt.ema_return = float(sparse_return)
        else:
            pt.ema_return = (
                (1 - self.ema) * pt.ema_return + self.ema * float(sparse_return)
            )

    def stats(self):
        return {
            p.name: {
                "ema_return": p.ema_return,
                "episodes": p.episodes,
                "kind": p.kind,
            }
            for p in self.partners
        }


def load_bc_partners(models_dir, layout, split="train"):
    """All BC seeds for a layout, e.g. models/bc/bc_train_cramped_room_s0.pt"""
    paths = sorted(
        glob.glob(os.path.join(models_dir, f"bc_{split}_{layout}_s*.pt"))
    )
    return [
        Partner(os.path.basename(p).replace(".pt", ""), load_bc(p), "bc")
        for p in paths
    ]


def load_ckpt_partners(models_root, layout, method="sp"):
    """Frozen checkpoints from all seeds of `method` runs on this layout."""
    pattern = os.path.join(
        models_root, f"{layout}__{method}__s*", "ckpt_*.pt"
    )
    partners = []
    for p in sorted(glob.glob(pattern)):
        run = os.path.basename(os.path.dirname(p))
        tag = os.path.basename(p).replace(".pt", "")
        partners.append(
            Partner(f"{run}/{tag}", load_actor_critic(p), "ckpt")
        )
    return partners
