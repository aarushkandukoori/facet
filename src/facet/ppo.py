"""PPO for Overcooked with three partner modes.

  sp      -- one parameter-shared policy controls both players; both
             players' streams enter the buffer (2 * n_envs streams).
  partner -- the learner controls seat i%2 in env i; a frozen partner
             (single net or PartnerPool) controls the other seat; only the
             learner's stream enters the buffer (n_envs streams).

Rewards: shared sparse (+20/soup) plus per-player shaped rewards whose
coefficient anneals linearly to zero over `shaped_frac` of training.
Rewards are scaled by `reward_scale` for optimization only; logging is in
raw sparse units.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from facet.nets import ActorCritic, save_policy


class PPOConfig:
    total_steps = 3_000_000     # env steps (per env-step, not per stream)
    n_workers = 5
    envs_per_worker = 4
    rollout = 400
    lr = 3e-4
    gamma = 0.99
    lam = 0.95
    clip = 0.2
    epochs = 8
    minibatch = 2048
    vf_coef = 0.5
    ent_start = 0.05
    ent_end = 0.005
    ent_anneal_frac = 0.8
    grad_clip = 0.5
    reward_scale = 0.1
    shaped_frac = 0.6           # shaped reward anneals to 0 by this fraction
    ckpt_fracs = (0.08, 0.2, 0.4, 1.0)
    hidden = 128

    def __init__(self, **kw):
        for k, v in kw.items():
            assert hasattr(self, k), k
            setattr(self, k, v)


def _gae(rew, val, done, last_val, gamma, lam):
    T, S = rew.shape
    adv = np.zeros((T, S), dtype=np.float32)
    lastgae = np.zeros(S, dtype=np.float32)
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        nextval = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nextval * nonterm - val[t]
        lastgae = delta + gamma * lam * nonterm * lastgae
        adv[t] = lastgae
    return adv


def train(layout, mode, seed, out_dir, cfg: PPOConfig, partner=None,
          pool=None, log_path=None):
    """mode: 'sp' or 'partner'. For 'partner', pass `partner` (a single
    Partner) or `pool` (a PartnerPool)."""
    from facet.vecenv import VecOvercooked

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(log_path or os.path.join(out_dir, "train_log.jsonl"), "a")

    vec = VecOvercooked(layout, cfg.n_workers, cfg.envs_per_worker,
                        horizon=400, seed=seed)
    n_envs = vec.n_envs
    net = ActorCritic(hidden=cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    is_sp = mode == "sp"
    S = n_envs * 2 if is_sp else n_envs  # learner streams
    learner_seat = np.array([i % 2 for i in range(n_envs)])

    use_pool = pool is not None
    if mode == "partner":
        assert (partner is not None) ^ use_pool
        env_partner = (
            np.array([pool.sample() for _ in range(n_envs)])
            if use_pool else np.zeros(n_envs, dtype=int)
        )

    def partner_of(i):
        return pool.partners[env_partner[i]] if use_pool else partner

    obs = vec.reset()  # (n_envs, 2, 96)
    steps_done = 0
    recent_returns = []
    ep_count = 0
    ckpts_saved = set()
    t_start = time.time()

    while steps_done < cfg.total_steps:
        frac = steps_done / cfg.total_steps
        shaped_coef = max(0.0, 1.0 - frac / cfg.shaped_frac)
        ent_coef = cfg.ent_start + (cfg.ent_end - cfg.ent_start) * min(
            1.0, frac / cfg.ent_anneal_frac
        )

        b_obs = np.empty((cfg.rollout, S, 96), dtype=np.float32)
        b_act = np.empty((cfg.rollout, S), dtype=np.int64)
        b_logp = np.empty((cfg.rollout, S), dtype=np.float32)
        b_val = np.empty((cfg.rollout, S), dtype=np.float32)
        b_rew = np.empty((cfg.rollout, S), dtype=np.float32)
        b_done = np.empty((cfg.rollout, S), dtype=np.float32)

        for t in range(cfg.rollout):
            if is_sp:
                flat = torch.as_tensor(obs.reshape(2 * n_envs, 96))
                with torch.no_grad():
                    a, logp, v = net.act(flat)
                joint = a.numpy().reshape(n_envs, 2)
                l_obs = obs.reshape(2 * n_envs, 96)
                l_act, l_logp, l_val = a.numpy(), logp.numpy(), v.numpy()
            else:
                l_ob = obs[np.arange(n_envs), learner_seat]
                p_ob = obs[np.arange(n_envs), 1 - learner_seat]
                with torch.no_grad():
                    a, logp, v = net.act(torch.as_tensor(l_ob))
                p_act = np.empty(n_envs, dtype=np.int64)
                if use_pool:
                    for pid in np.unique(env_partner):
                        m = env_partner == pid
                        p_act[m] = (
                            pool.partners[pid]
                            .act(torch.as_tensor(p_ob[m]))
                            .numpy()
                        )
                else:
                    p_act[:] = partner.act(torch.as_tensor(p_ob)).numpy()
                joint = np.empty((n_envs, 2), dtype=np.int64)
                joint[np.arange(n_envs), learner_seat] = a.numpy()
                joint[np.arange(n_envs), 1 - learner_seat] = p_act
                l_obs = l_ob
                l_act, l_logp, l_val = a.numpy(), logp.numpy(), v.numpy()

            nobs, sparse, shaped, done, finished = vec.step(joint)

            if is_sp:
                rew = (
                    sparse[:, None].repeat(2, 1) + shaped_coef * shaped
                ).reshape(2 * n_envs)
                d = done[:, None].repeat(2, 1).reshape(2 * n_envs)
            else:
                sh_l = shaped[np.arange(n_envs), learner_seat]
                rew = sparse + shaped_coef * sh_l
                d = done

            b_obs[t] = l_obs
            b_act[t] = l_act
            b_logp[t] = l_logp
            b_val[t] = l_val
            b_rew[t] = rew * cfg.reward_scale
            b_done[t] = d.astype(np.float32)

            for ep in finished:
                recent_returns.append(ep["sparse"])
                ep_count += 1
                if mode == "partner" and use_pool:
                    pool.update(env_partner[ep["env"]], ep["sparse"])
                    env_partner[ep["env"]] = pool.sample()

            obs = nobs
            steps_done += n_envs

        # bootstrap value for streams (episodes are horizon-aligned with the
        # rollout, so done=1 at the last step and last_val is multiplied out;
        # still computed for safety if rollout != horizon)
        if is_sp:
            with torch.no_grad():
                _, _, lv = net.act(torch.as_tensor(obs.reshape(2 * n_envs, 96)))
            last_val = lv.numpy()
        else:
            l_ob = obs[np.arange(n_envs), learner_seat]
            with torch.no_grad():
                _, _, lv = net.act(torch.as_tensor(l_ob))
            last_val = lv.numpy()

        adv = _gae(b_rew, b_val, b_done, last_val, cfg.gamma, cfg.lam)
        ret = adv + b_val
        f_obs = torch.as_tensor(b_obs.reshape(-1, 96))
        f_act = torch.as_tensor(b_act.reshape(-1))
        f_logp = torch.as_tensor(b_logp.reshape(-1))
        f_adv = torch.as_tensor(adv.reshape(-1))
        f_ret = torch.as_tensor(ret.reshape(-1))
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        n = f_obs.shape[0]
        idx = np.arange(n)
        pi_losses, v_losses, ents = [], [], []
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch):
                mb = idx[start:start + cfg.minibatch]
                logp, ent, val = net.evaluate_actions(f_obs[mb], f_act[mb])
                ratio = torch.exp(logp - f_logp[mb])
                a_mb = f_adv[mb]
                l_pi = -torch.min(
                    ratio * a_mb,
                    torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a_mb,
                ).mean()
                l_v = nn.functional.mse_loss(val, f_ret[mb])
                loss = l_pi + cfg.vf_coef * l_v - ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                opt.step()
                pi_losses.append(l_pi.item())
                v_losses.append(l_v.item())
                ents.append(ent.mean().item())

        recent = recent_returns[-40:]
        rec = {
            "steps": steps_done,
            "ep": ep_count,
            "sparse_mean_recent": float(np.mean(recent)) if recent else 0.0,
            "shaped_coef": round(shaped_coef, 4),
            "ent_coef": round(ent_coef, 4),
            "pi_loss": float(np.mean(pi_losses)),
            "v_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(ents)),
            "sps": int(steps_done / (time.time() - t_start)),
        }
        if mode == "partner" and use_pool:
            rec["pool"] = pool.stats()
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

        for cf in cfg.ckpt_fracs:
            if cf not in ckpts_saved and steps_done >= cf * cfg.total_steps:
                tag = f"ckpt_{int(cf * 100):03d}"
                save_policy(net, os.path.join(out_dir, f"{tag}.pt"),
                            {"layout": layout, "mode": mode, "seed": seed,
                             "steps": steps_done, "frac": cf})
                ckpts_saved.add(cf)

    save_policy(net, os.path.join(out_dir, "final.pt"),
                {"layout": layout, "mode": mode, "seed": seed,
                 "steps": steps_done})
    vec.close()
    log_f.close()
    return net
