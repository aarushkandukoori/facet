"""Subprocess vector env for Overcooked.

Each worker process owns `envs_per_worker` serial environments and does all
Python-heavy stepping + featurization. The main process owns every neural
network and sends joint actions down the pipe.

Protocol per step exchange (worker -> main):
    obs:      (K, 2, 96) float32   featurized obs for both players
    sparse:   (K,)       float32   shared sparse reward this step
    shaped:   (K, 2)     float32   per-player shaped reward this step
    done:     (K,)       bool
    ep_stats: list of dicts for envs that finished this step
"""

import multiprocessing as mp

import numpy as np


def _worker(remote, layout, horizon, envs_per_worker, seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    from facet.envtools import make_env

    envs = [make_env(layout, horizon) for _ in range(envs_per_worker)]
    ep_sparse = np.zeros(envs_per_worker)
    ep_shaped = np.zeros((envs_per_worker, 2))
    ep_len = np.zeros(envs_per_worker, dtype=int)

    def obs_of(i):
        o0, o1 = envs[i].featurize_state_mdp(envs[i].state)
        return np.stack([o0, o1]).astype(np.float32)

    while True:
        cmd, payload = remote.recv()
        if cmd == "reset":
            for e in envs:
                e.reset()
            ep_sparse[:] = 0
            ep_shaped[:] = 0
            ep_len[:] = 0
            remote.send(np.stack([obs_of(i) for i in range(envs_per_worker)]))
        elif cmd == "step":
            from overcooked_ai_py.mdp.actions import Action

            joint_idx = payload  # (K, 2) ints
            obs = np.empty((envs_per_worker, 2, 96), dtype=np.float32)
            sparse = np.zeros(envs_per_worker, dtype=np.float32)
            shaped = np.zeros((envs_per_worker, 2), dtype=np.float32)
            done = np.zeros(envs_per_worker, dtype=bool)
            finished = []
            for i, env in enumerate(envs):
                a = (
                    Action.INDEX_TO_ACTION[int(joint_idx[i, 0])],
                    Action.INDEX_TO_ACTION[int(joint_idx[i, 1])],
                )
                _, r, d, info = env.step(a)
                sh = info.get("shaped_r_by_agent", [0.0, 0.0])
                sparse[i] = r
                shaped[i] = sh
                ep_sparse[i] += r
                ep_shaped[i] += sh
                ep_len[i] += 1
                done[i] = d
                if d:
                    finished.append(
                        {
                            "env": i,
                            "sparse": float(ep_sparse[i]),
                            "shaped": ep_shaped[i].tolist(),
                            "len": int(ep_len[i]),
                        }
                    )
                    env.reset()
                    ep_sparse[i] = 0
                    ep_shaped[i] = 0
                    ep_len[i] = 0
                obs[i] = obs_of(i)
            remote.send((obs, sparse, shaped, done, finished))
        elif cmd == "close":
            remote.close()
            break


class VecOvercooked:
    """n_workers x envs_per_worker Overcooked envs behind pipes."""

    def __init__(self, layout, n_workers=6, envs_per_worker=4, horizon=400,
                 seed=0):
        self.layout = layout
        self.n_workers = n_workers
        self.envs_per_worker = envs_per_worker
        self.n_envs = n_workers * envs_per_worker
        self.horizon = horizon
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for w in range(n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(child, layout, horizon, envs_per_worker, seed * 997 + w),
                daemon=True,
            )
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        obs = np.concatenate([r.recv() for r in self.remotes])
        return obs  # (n_envs, 2, 96)

    def step(self, joint_idx):
        """joint_idx: (n_envs, 2) int action indices."""
        k = self.envs_per_worker
        for w, r in enumerate(self.remotes):
            r.send(("step", joint_idx[w * k:(w + 1) * k]))
        obs, sparse, shaped, done, finished = [], [], [], [], []
        for w, r in enumerate(self.remotes):
            o, s, sh, d, f = r.recv()
            obs.append(o)
            sparse.append(s)
            shaped.append(sh)
            done.append(d)
            for ep in f:
                ep["env"] += w * k
                finished.append(ep)
        return (
            np.concatenate(obs),
            np.concatenate(sparse),
            np.concatenate(shaped),
            np.concatenate(done),
            finished,
        )

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=2)
