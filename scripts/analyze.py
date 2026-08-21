"""Generate paper tables, figures, and site data from raw results.

Stages:
  data     -> paper/tables/data_table.tex (dataset stats; runnable now)
  results  -> paper/tables/main_table.tex, perseed_table.tex,
              docs/data/results.json, results/best_seeds.json,
              results/summary.json
  figures  -> paper/figs/*.pdf (learning curves, priority dynamics,
              generalization gap)
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "vendor/overcooked_ai/src"),
)

import numpy as np

LAYOUTS = ["cramped_room", "asymmetric_advantages", "coordination_ring",
           "random0", "random3"]
PRETTY = {"cramped_room": "Cramped Room",
          "asymmetric_advantages": "Asymm.\\ Advantages",
          "coordination_ring": "Coordination Ring",
          "random0": "Forced Coordination",
          "random3": "Counter Circuit"}
PRETTY_PLAIN = {"cramped_room": "Cramped Room",
                "asymmetric_advantages": "Asymmetric Advantages",
                "coordination_ring": "Coordination Ring",
                "random0": "Forced Coordination",
                "random3": "Counter Circuit"}
METHODS = ["bc", "sp", "ppo_bc", "facet"]
MLABEL = {"bc": "BC", "sp": "SP", "ppo_bc": "PPO$_{\\mathrm{BC}}$",
          "facet": "\\facet{}"}
# dataviz palette, light mode, fixed series order
COLOR = {"bc": "#2a78d6", "sp": "#eb6834", "ppo_bc": "#1baf7a",
         "facet": "#eda100"}
INK2, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"


def hh_reference():
    """Mean human-human test-game score normalized to 400 steps."""
    import pandas as pd

    df = pd.read_pickle(
        "vendor/overcooked_ai/src/human_aware_rl/static/human_data/cleaned/"
        "2019_hh_trials_test.pickle")
    out = {}
    g = df.groupby(["layout_name", "trial_id"]).agg(
        steps=("cur_gameloop", "max"), final=("score", "max"))
    for layout in LAYOUTS:
        sub = g.loc[layout]
        out[layout] = float((sub["final"] * 400.0 / sub["steps"]).mean())
    return out


def stage_data():
    stats = json.load(open("data/bc/stats.json"))
    rows = []
    tot = {"train_t": 0, "train_g": 0, "test_t": 0, "test_g": 0}
    for layout in LAYOUTS:
        tr, te = stats["train"][layout], stats["test"][layout]
        rows.append(
            f"{PRETTY[layout]} & {tr['trials']} & {tr['timesteps']:,} & "
            f"{te['trials']} & {te['timesteps']:,} \\\\")
        tot["train_t"] += tr["timesteps"]; tot["train_g"] += tr["trials"]
        tot["test_t"] += te["timesteps"]; tot["test_g"] += te["trials"]
    body = "\n".join(rows)
    tex = f"""\\begin{{tabular}}{{lrrrr}}
\\toprule
& \\multicolumn{{2}}{{c}}{{Train split}} & \\multicolumn{{2}}{{c}}{{Test split (\\hproxy{{}})}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
Layout & games & timesteps & games & timesteps \\\\
\\midrule
{body}
\\midrule
Total & {tot['train_g']} & {tot['train_t']:,} & {tot['test_g']} & {tot['test_t']:,} \\\\
\\bottomrule
\\end{{tabular}}"""
    os.makedirs("paper/tables", exist_ok=True)
    open("paper/tables/data_table.tex", "w").write(tex)
    print("wrote paper/tables/data_table.tex")


def load_eval():
    """-> agg[layout][method][partner] = {mean, se, seed_means},
       raw per-seed means as well."""
    agg = {}
    for layout in LAYOUTS:
        path = f"results/eval_{layout}.json"
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        agg[layout] = {}
        for method in METHODS:
            cell = {}
            for partner in ["h_proxy", "self"]:
                seed_means = []
                for seed in [0, 1, 2]:
                    scores = []
                    for p in d["pairs"]:
                        if (p["agent"] == method and p["agent_seed"] == seed
                                and p["partner"] == partner):
                            scores += p["scores"]
                    if scores:
                        seed_means.append(float(np.mean(scores)))
                if seed_means:
                    cell[partner] = {
                        "mean": float(np.mean(seed_means)),
                        "se": float(np.std(seed_means, ddof=1)
                                    / np.sqrt(len(seed_means)))
                        if len(seed_means) > 1 else 0.0,
                        "seed_means": seed_means,
                    }
            if cell:
                agg[layout][method] = cell
    return agg


def stage_results():
    agg = load_eval()
    hh = hh_reference()

    # --- main table ---
    lines = []
    for layout in LAYOUTS:
        if layout not in agg:
            continue
        a = agg[layout]
        # bold the best h_proxy mean, plus any method within one s.e. of it
        hvals = {m: a[m]["h_proxy"]["mean"] for m in METHODS if m in a}
        best_m = max(hvals, key=hvals.get)
        thresh = hvals[best_m] - a[best_m]["h_proxy"]["se"]
        cells_h, cells_s = [], []
        for m in METHODS:
            if m not in a:
                cells_h.append("--"); cells_s.append("--")
                continue
            hm = a[m]["h_proxy"]
            sm = a[m].get("self")
            txt = f"{hm['mean']:.1f} $\\pm$ {hm['se']:.1f}"
            if hm["mean"] + hm["se"] >= thresh:
                txt = f"\\textbf{{{txt}}}"
            cells_h.append(txt)
            cells_s.append(
                f"{sm['mean']:.1f} $\\pm$ {sm['se']:.1f}" if sm else "--")
        lines.append(
            f"{PRETTY[layout]} & " + " & ".join(cells_h) +
            f" & {hh[layout]:.0f} \\\\")
        lines.append(
            "\\quad\\emph{with self} & " + " & ".join(cells_s) + " & \\\\")
    header = (" & ".join([""] + [MLABEL[m] for m in METHODS] + ["HH"]))
    tex = f"""\\begin{{tabular}}{{lccccc}}
\\toprule
{header} \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}"""
    open("paper/tables/main_table.tex", "w").write(tex)

    # --- per-seed appendix table ---
    ps = []
    for layout in LAYOUTS:
        if layout not in agg:
            continue
        for m in METHODS:
            if m not in agg[layout]:
                continue
            sm = agg[layout][m]["h_proxy"]["seed_means"]
            ps.append(f"{PRETTY[layout]} & {MLABEL[m]} & " +
                      " & ".join(f"{v:.1f}" for v in sm) + " \\\\")
    tex = ("\\begin{table}[h]\\centering\\caption{Per-seed mean scores with "
           "\\hproxy{} (30 episodes per seed: 3 proxy seeds $\\times$ 10 "
           "episodes).}\\label{tab:perseed}\\small\n"
           "\\begin{tabular}{llccc}\n\\toprule\nLayout & Method & seed 0 & "
           "seed 1 & seed 2 \\\\\n\\midrule\n" + "\n".join(ps) +
           "\n\\bottomrule\n\\end{tabular}\\end{table}")
    open("paper/tables/perseed_table.tex", "w").write(tex)

    # --- site data + best seeds ---
    site = {"h_proxy": {}, "self": {}, "hh": hh}
    best_seeds = {}
    for layout in LAYOUTS:
        if layout not in agg:
            continue
        site["h_proxy"][layout] = {}
        site["self"][layout] = {}
        best_seeds[layout] = {}
        for m in METHODS:
            if m not in agg[layout]:
                continue
            c = agg[layout][m]
            site["h_proxy"][layout][m] = {
                "mean": c["h_proxy"]["mean"], "se": c["h_proxy"]["se"]}
            if "self" in c:
                site["self"][layout][m] = {
                    "mean": c["self"]["mean"], "se": c["self"]["se"]}
            best_seeds[layout][m] = int(np.argmax(c["h_proxy"]["seed_means"]))
    os.makedirs("docs/data", exist_ok=True)
    json.dump(site, open("docs/data/results.json", "w"))
    os.makedirs("results", exist_ok=True)
    json.dump(best_seeds, open("results/best_seeds.json", "w"), indent=1)
    json.dump({"agg": agg, "hh": hh}, open("results/summary.json", "w"),
              indent=1, default=float)
    print("wrote main_table, perseed_table, results.json, best_seeds.json")

    # console summary
    for layout in LAYOUTS:
        if layout not in agg:
            continue
        row = "  ".join(
            f"{m}:{agg[layout][m]['h_proxy']['mean']:6.1f}±{agg[layout][m]['h_proxy']['se']:4.1f}"
            for m in METHODS if m in agg[layout])
        print(f"{layout:24s} {row}")


def _style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def stage_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.labelcolor": INK2, "text.color": INK2,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42,
    })
    os.makedirs("paper/figs", exist_ok=True)

    # ---------- Fig 1: layouts (terrain thumbnails) + learning curves ----
    from facet.envtools import make_env

    TERRAIN_COLOR = {"X": "#cfccc0", " ": "#f4f2ec", "O": "#eb6834",
                     "D": "#9ec5f4", "P": "#57534c", "S": "#0ca30c"}
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.2),
                             gridspec_kw={"height_ratios": [1, 1.7]})
    for ci, layout in enumerate(LAYOUTS):
        ax = axes[0][ci]
        env = make_env(layout)
        t = env.mdp.terrain_mtx
        img = np.zeros((len(t), len(t[0]), 3))
        for y, row in enumerate(t):
            for x, c in enumerate(row):
                h = TERRAIN_COLOR.get(c, "#ffffff").lstrip("#")
                img[y, x] = [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]
        ax.imshow(img, interpolation="nearest")
        ax.set_title(PRETTY_PLAIN[layout], fontsize=8.5, color="#0b0b0b")
        ax.axis("off")

        ax = axes[1][ci]
        for m in ["sp", "ppo_bc", "facet"]:
            curves = []
            for seed in [0, 1, 2]:
                p = f"models/{layout}__{m}__s{seed}/train_log.jsonl"
                if not os.path.exists(p):
                    continue
                steps, val = [], []
                for line in open(p):
                    d = json.loads(line)
                    steps.append(d["steps"] / 1e6)
                    val.append(d["sparse_mean_recent"])
                if steps:
                    curves.append((np.array(steps), np.array(val)))
            if not curves:
                continue
            grid_x = np.linspace(0, max(c[0][-1] for c in curves), 120)
            ys = np.stack([np.interp(grid_x, c[0], c[1]) for c in curves])
            mean, lo, hi = ys.mean(0), ys.min(0), ys.max(0)
            ax.plot(grid_x, mean, color=COLOR[m], lw=1.6,
                    label={"sp": "SP", "ppo_bc": "PPO_BC",
                           "facet": "FACET"}[m])
            ax.fill_between(grid_x, lo, hi, color=COLOR[m], alpha=0.15,
                            lw=0)
        _style_ax(ax)
        ax.set_xlabel("env steps (M)", fontsize=8)
        if ci == 0:
            ax.set_ylabel("sparse return / episode", fontsize=8)
            leg = ax.legend(fontsize=7.5, loc="upper left", frameon=True,
                            framealpha=0.85, edgecolor="none",
                            facecolor="white")
    fig.tight_layout()
    fig.savefig("paper/figs/layouts_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------- Fig 2: FACET priority dynamics ----------
    fig, axes = plt.subplots(1, len(LAYOUTS), figsize=(11, 2.3),
                             sharey=True)
    TAU, EPS = 0.35, 0.25
    for ci, layout in enumerate(LAYOUTS):
        ax = axes[ci]
        p = f"models/{layout}__facet__s0/train_log.jsonl"
        if os.path.exists(p):
            steps, mass_bc, mass_early, mass_late = [], [], [], []
            for line in open(p):
                d = json.loads(line)
                if "pool" not in d:
                    continue
                names = list(d["pool"].keys())
                emas = np.array([
                    d["pool"][n]["ema_return"]
                    if d["pool"][n]["ema_return"] is not None else -np.inf
                    for n in names])
                seen = np.isfinite(emas)
                if seen.any():
                    lo, hi = emas[seen].min(), emas[seen].max()
                    span = max(hi - lo, 1.0)
                    norm = np.where(seen, (emas - lo) / span, 0.0)
                else:
                    norm = np.zeros(len(emas))
                logits = -norm / TAU
                sm = np.exp(logits - logits.max()); sm /= sm.sum()
                prob = EPS / len(sm) + (1 - EPS) * sm
                prob /= prob.sum()
                is_bc = np.array([d["pool"][n]["kind"] == "bc"
                                  for n in names])
                is_early = np.array([("ckpt_008" in n or "ckpt_020" in n)
                                     for n in names])
                is_late = ~is_bc & ~is_early
                steps.append(d["steps"] / 1e6)
                mass_bc.append(prob[is_bc].sum())
                mass_early.append(prob[is_early].sum())
                mass_late.append(prob[is_late].sum())
            ax.plot(steps, mass_bc, color=COLOR["bc"], lw=1.5,
                    label="BC humans (3)")
            ax.plot(steps, mass_early, color=COLOR["sp"], lw=1.5,
                    label="early ckpts (6)")
            ax.plot(steps, mass_late, color=COLOR["ppo_bc"], lw=1.5,
                    label="late ckpts (6)")
        _style_ax(ax)
        ax.set_title(PRETTY_PLAIN[layout], fontsize=8.5, color="#0b0b0b")
        ax.set_xlabel("env steps (M)", fontsize=8)
        if ci == 0:
            ax.set_ylabel("sampling prob.\nmass", fontsize=8)
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig("paper/figs/priority.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------- Fig 3: generalization gap ----------
    agg = json.load(open("results/summary.json"))["agg"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    x = np.arange(len(LAYOUTS))
    w = 0.2
    for mi, m in enumerate(["sp", "ppo_bc", "facet"]):
        selfv = [agg[l][m]["self"]["mean"] for l in LAYOUTS]
        hv = [agg[l][m]["h_proxy"]["mean"] for l in LAYOUTS]
        pos = x + (mi - 1) * (w + 0.03)
        ax.bar(pos, selfv, width=w, color=COLOR[m], alpha=0.25, lw=0)
        ax.bar(pos, hv, width=w, color=COLOR[m], lw=0,
               label={"sp": "SP", "ppo_bc": "PPO_BC",
                      "facet": "FACET"}[m])
        for xx, sv, hvv in zip(pos, selfv, hv):
            ax.plot([xx, xx], [hvv, sv], color=COLOR[m], lw=1.0,
                    alpha=0.7)
    _style_ax(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_PLAIN[l] for l in LAYOUTS], fontsize=8)
    ax.set_ylabel("mean episode score", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig("paper/figs/gap.pdf", bbox_inches="tight")
    plt.close(fig)

    print("wrote paper/figs/{layouts_curves,priority,gap}.pdf")


def stage_fleet():
    """Aggregate fleet cross-play into a table + site json."""
    agg = {}
    for layout in LAYOUTS:
        path = f"results/fleet_{layout}.json"
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        agg[layout] = {}
        for method in METHODS:
            seed_means = []
            for seed in [0, 1, 2]:
                scores = []
                for p in d["pairs"]:
                    if p["agent"] == method and p["agent_seed"] == seed:
                        scores += p["scores"]
                if scores:
                    seed_means.append(float(np.mean(scores)))
            if seed_means:
                agg[layout][method] = {
                    "mean": float(np.mean(seed_means)),
                    "se": float(np.std(seed_means, ddof=1)
                                / np.sqrt(len(seed_means)))
                    if len(seed_means) > 1 else 0.0,
                    "seed_means": seed_means,
                }
    # tex table: fleet + h_proxy side by side per method
    hagg = json.load(open("results/summary.json"))["agg"]
    lines = []
    for layout in LAYOUTS:
        if layout not in agg:
            continue
        fvals = {m: agg[layout][m]["mean"] for m in METHODS if m in agg[layout]}
        best_f = max(fvals, key=fvals.get)
        thresh = fvals[best_f] - agg[layout][best_f]["se"]
        cells = []
        for m in METHODS:
            if m not in agg[layout]:
                cells.append("--")
                continue
            c = agg[layout][m]
            txt = f"{c['mean']:.1f} $\\pm$ {c['se']:.1f}"
            if c["mean"] + c["se"] >= thresh:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(f"{PRETTY[layout]} & " + " & ".join(cells) + " \\\\")
    header = " & ".join([""] + [MLABEL[m] for m in METHODS])
    tex = f"""\\begin{{tabular}}{{lcccc}}
\\toprule
{header} \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}"""
    open("paper/tables/fleet_table.tex", "w").write(tex)
    json.dump(agg, open("results/fleet_summary.json", "w"), indent=1,
              default=float)
    site = json.load(open("docs/data/results.json"))
    site["fleet"] = {
        l: {m: {"mean": agg[l][m]["mean"], "se": agg[l][m]["se"]}
            for m in agg[l]} for l in agg}
    json.dump(site, open("docs/data/results.json", "w"))
    print("wrote fleet_table.tex, fleet_summary.json, updated results.json")
    for layout in agg:
        row = "  ".join(f"{m}:{agg[layout][m]['mean']:6.1f}±{agg[layout][m]['se']:4.1f}"
                        for m in METHODS if m in agg[layout])
        print(f"{layout:24s} {row}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "results", "figures", "fleet"])
    a = ap.parse_args()
    {"data": stage_data, "results": stage_results,
     "figures": stage_figures, "fleet": stage_fleet}[a.stage]()
