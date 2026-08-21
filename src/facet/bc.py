"""Behavior cloning on the human-human data."""

import json
import os

import numpy as np
import torch
import torch.nn as nn

from facet.nets import BCPolicy, save_policy


def train_bc(npz_path, out_path, seed=0, epochs=60, lr=1e-3, batch=512,
             val_frac=0.1, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    d = np.load(npz_path)
    obs, act = d["obs"], d["act"]
    n = len(act)
    idx = np.random.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    obs_t = torch.as_tensor(obs)
    act_t = torch.as_tensor(act)

    net = BCPolicy()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
    best_val, best_state, bad = np.inf, None, 0
    hist = []
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(tr_idx)
        tot, count = 0.0, 0
        for s in range(0, len(perm), batch):
            mb = perm[s:s + batch]
            logits = net(obs_t[mb])
            loss = nn.functional.cross_entropy(logits, act_t[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(mb)
            count += len(mb)
        net.eval()
        with torch.no_grad():
            vlogits = net(obs_t[val_idx])
            vloss = nn.functional.cross_entropy(vlogits, act_t[val_idx]).item()
            vacc = (vlogits.argmax(-1) == act_t[val_idx]).float().mean().item()
        hist.append({"epoch": ep, "train_loss": tot / count,
                     "val_loss": vloss, "val_acc": vacc})
        if vloss < best_val - 1e-4:
            best_val, bad = vloss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        vlogits = net(obs_t[val_idx])
        final_acc = (vlogits.argmax(-1) == act_t[val_idx]).float().mean().item()
    save_policy(net, out_path, {"npz": os.path.basename(npz_path),
                                "seed": seed, "val_acc": final_acc,
                                "val_loss": best_val, "n_samples": n})
    return {"val_acc": final_acc, "val_loss": best_val, "epochs": len(hist),
            "n_samples": n, "history": hist}


if __name__ == "__main__":
    import sys

    data_dir, out_dir = sys.argv[1], sys.argv[2]
    seeds = [0, 1, 2]
    from facet import LAYOUTS

    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for split in ["train", "test"]:
        for layout in LAYOUTS:
            npz = os.path.join(data_dir, f"bc_{split}_{layout}.npz")
            for seed in seeds:
                out = os.path.join(out_dir, f"bc_{split}_{layout}_s{seed}.pt")
                r = train_bc(npz, out, seed=seed)
                key = f"{split}/{layout}/s{seed}"
                report[key] = {k: r[k] for k in
                               ["val_acc", "val_loss", "epochs", "n_samples"]}
                print(key, report[key], flush=True)
    with open(os.path.join(out_dir, "bc_report.json"), "w") as f:
        json.dump(report, f, indent=2)
