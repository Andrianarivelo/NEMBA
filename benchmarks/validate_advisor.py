"""How often does the data advisor read synthetic data correctly?

For every synthetic generator (known modality, orientation, ring topology and
discrete-state structure) and many seeds, run ``analyze_data`` and score:
  * modality, time axis (data also given transposed), ring flag, cluster flag;
  * accuracy with Wilson 95 % confidence intervals;
  * a modality confusion matrix.
Outputs: benchmarks/results/advisor_validation*.csv and advisor_validation.{png,pdf,svg}.

    python benchmarks/validate_advisor.py
"""

# =========================================================================== #
# PARAMETERS (edit here)                                                      #
# =========================================================================== #
SEEDS = list(range(10))
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

from embedders_backbones import synthetic as S                 # noqa: E402
from embedders_backbones.advisor import analyze_data           # noqa: E402


def cases(seed):
    """(name, data, kwargs, truth) for one seed. truth: modality, axis, ring, clusters."""
    ds = S.make_dataset(n_neurons=50, duration=500, seed=seed)
    m = S.make_metastable(seed=seed)
    sh = S.make_sheet(seed=seed)
    ca = S.make_calcium(duration=400, seed=seed)
    ph = S.make_photometry(duration=400, seed=seed)
    po = S.make_pose(duration=200, seed=seed)
    yield "ring spike times", ds["spike_times"], {}, ("spike_times", 1, True, False)
    yield "ring counts", ds["counts"], {}, ("binned_counts", 1, True, False)
    yield "ring counts (transposed)", ds["counts"].T, {}, ("binned_counts", 0, True, False)
    yield "metastable counts", m["counts"], {}, ("binned_counts", 1, False, True)
    yield "sheet counts", sh["counts"], {}, ("binned_counts", 1, False, False)
    yield "calcium traces", ca["traces"], dict(fs=ca["fs"]), ("calcium", 1, True, False)
    yield "photometry", ph["signal"], dict(fs=ph["fs"]), ("photometry", 1, False, False)
    yield "pose (no names)", po["X"], dict(fs=po["fps"]), ("pose", 0, False, False)


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for seed in SEEDS:
        for name, data, kw, (mod, axis, ring, clus) in cases(seed):
            t0 = time.perf_counter()
            p = analyze_data(data, **kw)
            rows.append(dict(case=name, seed=seed, true_modality=mod, modality=p.modality,
                             true_axis=axis, axis=p.time_axis, true_ring=ring, ring=p.has_ring,
                             true_clusters=clus, clusters=p.has_clusters,
                             ring_persistence=p.ring_persistence, silhouette=p.silhouette_best,
                             drift_frac=p.drift_frac, seconds=time.perf_counter() - t0))
        print(f"seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "advisor_validation_raw.csv"), index=False)
    acc = []
    for case, g in df.groupby("case", sort=False):
        for what in ("modality", "axis", "ring", "clusters"):
            k = int((g[what] == g[f"true_{what}"]).sum()); n = len(g)
            lo, hi = wilson(k, n)
            acc.append(dict(case=case, check=what, correct=k, n=n, accuracy=k / n,
                            ci95_low=lo, ci95_high=hi))
    acc = pd.DataFrame(acc)
    acc.to_csv(os.path.join(OUT, "advisor_validation_accuracy.csv"), index=False)
    conf = pd.crosstab(df.true_modality, df.modality)
    conf.to_csv(os.path.join(OUT, "advisor_validation_confusion.csv"))
    tot = acc.groupby("check")[["correct", "n"]].sum()
    tot["accuracy"] = tot.correct / tot.n
    tot[["ci95_low", "ci95_high"]] = [wilson(k, n) for k, n in zip(tot.correct, tot.n)]
    print(tot.round(3).to_string())
    print(f"median profiling time {df.seconds.median():.1f} s")
    figure(acc, conf, tot)


def figure(acc, conf, tot):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    rc = {"font.size": FONT, "axes.spines.top": False, "axes.spines.right": False,
          "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "legend.frameon": False,
          "svg.fonttype": "none", "pdf.fonttype": 42}
    checks = ["modality", "axis", "ring", "clusters"]
    cols = {"modality": "#2a78d6", "axis": "#1baf7a", "ring": "#eda100", "clusters": "#eb6834"}
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(9.0, 3.6), layout="constrained")
        a, b = fig.subplots(1, 2, width_ratios=[1.7, 1])
        cases = list(dict.fromkeys(acc.case))
        y = np.arange(len(cases))
        h = 0.18
        for j, c in enumerate(checks):
            g = acc[acc.check == c].set_index("case").reindex(cases)
            yy = y + (j - 1.5) * h
            a.barh(yy, g.accuracy, height=h * 0.85, color=cols[c],
                   label=f"{c} ({100 * tot.loc[c, 'accuracy']:.0f} % overall)")
            a.errorbar(g.accuracy, yy, xerr=[g.accuracy - g.ci95_low, g.ci95_high - g.accuracy],
                       fmt="none", ecolor="#52514e", elinewidth=0.6, capsize=1.2)
        a.set_yticks(y); a.set_yticklabels(cases); a.invert_yaxis()
        a.set_xlim(0, 1.05); a.set_xlabel("accuracy (Wilson 95 % CI)")
        a.grid(True, axis="x", color="#e6e5e1", lw=0.5); a.set_axisbelow(True)
        a.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncols=4)
        a.set_title(f"advisor checks, {acc.n.iloc[0]} seeds per case", loc="left", fontsize=FONT + 0.5)
        M = conf.to_numpy()
        b.imshow(M / M.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        b.set_xticks(range(M.shape[1])); b.set_xticklabels(conf.columns, rotation=45, ha="right")
        b.set_yticks(range(M.shape[0])); b.set_yticklabels(conf.index)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j]:
                    b.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=FONT - 1,
                           color="white" if M[i, j] / M[i].sum() > 0.6 else "#0b0b0b")
        b.set_xlabel("detected"); b.set_ylabel("true")
        b.set_title("modality confusion", loc="left", fontsize=FONT + 0.5)
        for f in FORMATS:
            fig.savefig(os.path.join(OUT, f"advisor_validation.{f}"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
