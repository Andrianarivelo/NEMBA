"""Which self-supervised objective should NEMBA use by default?

Compares, on synthetic recordings with known behaviour (``synthetic.make_dataset``):
  * likelihood: gaussian (squared error on z-scored sqrt counts, v0.2 default)
                vs poisson (raw counts, log-rate head);
  * masking:    time spans only (v0.2 default) vs time + neuron ("both");
for trained backbones, with PCA and FA as no-training references.

Scores (one common evaluator):
  * co-smoothing bits/spike on held-out neurons and held-out time (label free);
  * held-out decoding of phase / speed / state (blocked CV, linear);
  * state recovery ARI (K-means, K = 3).

Statistics: every configuration is compared with the v0.2 default
(gaussian-time) on the same (backbone, seed) pairs: Shapiro-Wilk on the paired
differences, then paired t-test or Wilcoxon, bootstrap 95 % CI, Holm
correction across metrics. Outputs in benchmarks/results/objectives_*.

    python benchmarks/benchmark_objectives.py
"""

# =========================================================================== #
# PARAMETERS (edit here)                                                      #
# =========================================================================== #
BACKBONES = ["tcn", "tst"]
REFERENCES = ["pca", "fa"]
SEEDS = [0, 1, 2]
CONFIGS = {                                   # label -> Embedder kwargs
    "gaussian-time": dict(likelihood="gaussian", mask_mode="time"),
    "gaussian-both": dict(likelihood="gaussian", mask_mode="both"),
    "poisson-time":  dict(likelihood="poisson", mask_mode="time"),
    "poisson-both":  dict(likelihood="poisson", mask_mode="both"),
}
BASELINE = "gaussian-time"
SPEED = "fast"
N_NEURONS, DURATION_S, DT = 60, 600.0, 0.1
K_TRUE = 3
BOOT = 5000
ALPHA = 0.05
FORMATS = ("png", "pdf", "svg")
FONT = 7.5
# =========================================================================== #

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from embedders_backbones import Embedder, synthetic                   # noqa: E402
from benchmark_v01_vs_v02 import evaluate, load_evaluator, holm, paired_stats  # noqa: E402


def run_one(bb, label, kw, seed, ds, dec):
    """Fit, co-smooth and evaluate one configuration."""
    t0 = time.perf_counter()
    emb = Embedder(bb, speed=SPEED, bin_size=DT, n_states=K_TRUE, log=None, seed=0, **kw)
    emb.fit_spikes(ds["spike_times"])
    fit_s = time.perf_counter() - t0
    cs = emb.cosmooth()
    row = dict(config=label, backbone=bb, seed=seed, fit_seconds=fit_s,
               cosmooth_bps=cs["bps"], val_bps=emb.history.get("best_val_bps", np.nan))
    row.update(evaluate(emb.Z, ds, emb.times, dec))
    return row


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    raw_csv = os.path.join(OUT, "objectives_raw.csv")
    rows = pd.read_csv(raw_csv).to_dict("records") if os.path.exists(raw_csv) else []
    done = {(r["config"], r["backbone"], int(r["seed"])) for r in rows}
    dec = load_evaluator()
    for seed in SEEDS:
        ds = synthetic.make_dataset(n_neurons=N_NEURONS, duration=DURATION_S, dt=DT, seed=seed)
        plan = [(bb, "reference", {}) for bb in REFERENCES] + \
               [(bb, lab, kw) for bb in BACKBONES for lab, kw in CONFIGS.items()]
        for bb, lab, kw in plan:
            if (lab, bb, seed) in done:
                continue
            r = run_one(bb, lab, kw, seed, ds, dec)
            rows.append(r)
            pd.DataFrame(rows).to_csv(raw_csv, index=False)
            print(f"{lab:14s} {bb:4s} seed={seed} cosmooth={r['cosmooth_bps']:.3f} "
                  f"phase={r['dec_phase']:.3f} speed={r['dec_speed']:.3f} state={r['dec_state']:.3f} "
                  f"ARI={r['ari_states']:.3f} fit={r['fit_seconds']:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    metrics = ["cosmooth_bps", "dec_phase", "dec_speed", "dec_state", "ari_states", "fit_seconds"]
    rng = np.random.default_rng(0)
    base = df[df.config == BASELINE].set_index(["backbone", "seed"])
    stat_rows = []
    for lab in CONFIGS:
        if lab == BASELINE:
            continue
        a = df[df.config == lab].set_index(["backbone", "seed"])
        idx = a.index.intersection(base.index)
        for m in metrics:
            d = (np.log10(a.loc[idx, m] / base.loc[idx, m]) if m == "fit_seconds"
                 else a.loc[idx, m] - base.loc[idx, m]).to_numpy()
            r = paired_stats(d, m, rng)
            r["comparison"] = f"{lab} - {BASELINE}"
            stat_rows.append(r)
    st = pd.DataFrame(stat_rows)
    for c in st.comparison.unique():
        sel = st.comparison == c
        st.loc[sel, "p_holm"] = holm(st.loc[sel, "p"].to_numpy())
    st.to_csv(os.path.join(OUT, "objectives_stats.csv"), index=False)
    summary = df.groupby(["config", "backbone"])[metrics].agg(["mean", "std"]).round(4)
    summary.to_csv(os.path.join(OUT, "objectives_summary.csv"))
    make_figure(df)
    print(st[["comparison", "metric", "n_pairs", "mean_diff", "ci95_low", "ci95_high",
              "test_used", "p_holm"]].round(4).to_string(index=False))
    print(f"\nresults in {OUT}")


def make_figure(df):
    """Bars = mean over seeds, dots = seeds; references shown as grey lines."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    pal = {"gaussian-time": "#8a8984", "gaussian-both": "#2a78d6",
           "poisson-time": "#eda100", "poisson-both": "#eb6834"}
    metrics = [("cosmooth_bps", "co-smoothing\n(bits/spike)"), ("dec_phase", "phase\nmean cos(Δθ)"),
               ("dec_speed", "speed R²"), ("dec_state", "state\nbalanced acc."),
               ("ari_states", "state ARI\n(K-means)")]
    rc = {"font.size": FONT, "axes.spines.top": False, "axes.spines.right": False,
          "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "legend.frameon": False,
          "svg.fonttype": "none", "pdf.fonttype": 42}
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(9.0, 2.9), layout="constrained")
        axes = fig.subplots(1, len(metrics))
        for ax, (m, lab) in zip(axes, metrics):
            x = 0
            ticks, names = [], []
            for bb in BACKBONES:
                for c in CONFIGS:
                    v = df[(df.backbone == bb) & (df.config == c)][m].to_numpy()
                    ax.bar(x, v.mean(), width=0.7, color=pal[c], label=c if (bb == BACKBONES[0]) else None)
                    ax.plot(np.full(len(v), x), v, "o", ms=2, color="#0b0b0b", alpha=0.6, mec="none")
                    x += 1
                ticks.append(x - len(CONFIGS) / 2 - 0.5); names.append(bb)
                x += 1
            for ref, ls in zip(REFERENCES, ("-", "--")):
                rv = df[(df.backbone == ref)][m].mean()
                ax.axhline(rv, color="#52514e", lw=0.8, ls=ls, label=f"{ref} (no training)")
            ax.set_xticks(ticks); ax.set_xticklabels(names)
            ax.set_title(lab, fontsize=FONT)
            ax.grid(True, axis="y", color="#e6e5e1", lw=0.5); ax.set_axisbelow(True)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="outside lower center", ncols=6)
        fig.suptitle("objective variants on synthetic ground truth (bars = mean of 3 seeds)",
                     fontsize=FONT + 1)
        for f in FORMATS:
            fig.savefig(os.path.join(OUT, f"objectives.{f}"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
