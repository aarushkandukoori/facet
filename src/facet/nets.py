"""Policy networks: actor-critic MLP for PPO, classifier MLP for BC."""

import torch
import torch.nn as nn

from facet import N_ACTIONS, OBS_DIM


def _mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.pi = _mlp([obs_dim, hidden, hidden, n_actions])
        self.v = _mlp([obs_dim, hidden, hidden, 1])

    def forward(self, obs):
        return self.pi(obs), self.v(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs, actions):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=64):
        super().__init__()
        self.net = _mlp([obs_dim, hidden, hidden, n_actions], act=nn.ReLU)

    def forward(self, obs):
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        logits = self.forward(obs)
        if deterministic:
            return logits.argmax(-1)
        return torch.distributions.Categorical(logits=logits).sample()


def save_policy(net, path, meta=None):
    torch.save({"state_dict": net.state_dict(), "meta": meta or {}}, path)


def load_actor_critic(path, hidden=128):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(hidden=hidden)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def load_bc(path, hidden=64):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = BCPolicy(hidden=hidden)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net
