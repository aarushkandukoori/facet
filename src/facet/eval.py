"""Cross-play evaluation: roll out (agent, partner) pairs in-process.

The human proxy H_proxy is behavior cloning trained on the held-out TEST
split of the human data, following Carroll et al. (2019): agents never see
H_proxy during training, so pairing with it estimates coordination with an
unseen human-like partner.
"""

import json
import os

import numpy as np
import torch

from facet.envtools import make_env

from overcooked_ai_py.mdp.actions import Action


@torch.no_grad()
def rollout(env, net0, net1, horizon=400, rng=None, record=False):
    """One episode; net_i controls player i. Returns sparse return
    (+ optional trajectory of state dicts / joint actions)."""
    env.reset()
    total = 0.0
    traj = {"states": [], "actions": [], "rewards": []} if record else None
    for _ in range(horizon):
        o0, o1 = env.featurize_state_mdp(env.state)
        obs = torch.as_tensor(
            np.stack([o0, o1]).astype(np.float32)
        )
        a0 = net0.act(obs[0:1])
        a1 = net1.act(obs[1:2])
        a0 = (a0[0] if isinstance(a0, tuple) else a0).item()
        a1 = (a1[0] if isinstance(a1, tuple) else a1).item()
        joint = (Action.INDEX_TO_ACTION[a0], Action.INDEX_TO_ACTION[a1])
        if record:
            traj["states"].append(env.state.to_dict())
            traj["actions"].append([a0, a1])
        _, r, done, info = env.step(joint)
        total += r
        if record:
            traj["rewards"].append(r)
        if done:
            break
    return (total, traj) if record else total


def eval_pair(layout, net_a, net_b, episodes=5, both_seatings=True,
              horizon=400, env=None):
    """Mean sparse return of (a,b); if both_seatings, half the episodes have
    a in seat 0 and half in seat 1."""
    env = env or make_env(layout, horizon)
    scores = []
    for ep in range(episodes):
        if both_seatings and ep % 2 == 1:
            scores.append(rollout(env, net_b, net_a, horizon))
        else:
            scores.append(rollout(env, net_a, net_b, horizon))
    return scores
