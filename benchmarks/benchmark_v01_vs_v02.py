"""Benchmark NEMBA v0.1 (commit a017c65) against v0.2 on synthetic ground truth.

What is compared
----------------
For every (backbone, dataset seed) the same spike recording is embedded by
  * v0.1 : ``Embedder(backbone, epochs=EPOCHS_V01)`` (the v0.1 CLI default),
  * v0.2 : ``Embedder(backbone, **kw)`` for each configuration in V02_CONFIGS
           (presets fast / balanced, objectives masked / contrastive / hybrid),
and every embedding is scored by ONE common evaluator (v0.2 decoding module,
blocked 5-fold CV with purge gap, linear decoders):
  * phase (circular, mean cos dtheta), speed (R^2), state (balanced accuracy),
  * ARI / NMI between KMeans(K=3) on the embedding and the true states,
  * CPU wall time of fit (train + embed; clustering excluded, it is shared).

Statistics
----------
Paired design: differences v0.2 - v0.1 per (backbone, seed). Normality of the
differences is checked with Shapiro-Wilk; a paired t-test is reported when it
is not rejected (p > 0.05), else the Wilcoxon signed-rank test (the decision
and both p-values are written to the CSV). Effect sizes: mean difference with
95 % bootstrap CI and the matched-pairs rank-biserial correlation. Time is
compared on log scale (ratio). Holm correction across the metric family.

Usage
-----
    python benchmarks/benchmark_v01_vs_v02.py --old <path to v0.1 checkout>
    (or set NEMBA_V01 to that path). Outputs go to benchmarks/results/.
Runs in the background in ~1-2 h on a 16-thread laptop CPU (no GPU).
"""

from __future__ import annotations

# =========================================================================== #
# PARAMETERS (edit here)                                                      #
# =========================================================================== #
BACKBONES = ["pca", "tcn", "embtcn", "c2f_tcn", "tst", "bilstm", "dtc"]
SEEDS = [0, 1, 2]                    # independent synthetic recordings
N_NEURONS = 60
DURATION_S = 600.0                   # 6000 bins at DT
DT = 0.1                             # bin size (s)
EPOCHS_V01 = 80                      # v0.1 CLI default
# v0.2 configurations: label -> (Embedder kwargs, backbones it runs on or None = all)
V02_CONFIGS = {
    "v0.2-fast":             (dict(speed="fast"), None),
    "v0.2-fast-contrastive": (dict(speed="fast", objective="contrastive"), "trained"),
    "v0.2-fast-hybrid":      (dict(speed="fast", objective="hybrid"), "trained"),
    "v0.2-balanced":         (dict(speed="balanced"), ["tcn", "tst", "bilstm"]),
}
DECODE_MODEL = "linear"
N_FOLDS = 5
K_TRUE = 3                           # number of true behavioural states
BOOT = 5000                          # bootstrap resamples for CIs
ALPHA = 0.05
FIG_FORMATS = ("png", "pdf", "svg")
# figure style
FIG_W, FIG_H = 7.2, 4.6
FONT = 7.5
# =========================================================================== #

import argparse
import importlib
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")


def load_package(path):
    """Import ``embedders_backbones`` from ``path`` (one version per process)."""
    sys.path.insert(0, path)
    return importlib.import_module("embedders_backbones")


def load_evaluator():
    """The v0.2 decoding module loaded standalone (it has no package-relative
    imports), so v0.1 and v0.2 embeddings are scored by identical code."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nemba_eval_decoding", os.path.join(ROOT, "embedders_backbones", "decoding.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                 # required by @dataclass
    spec.loader.exec_module(mod)
    return mod


def make_dataset(seed):
    """Ground-truth recording (v0.2 generator, loaded standalone for both versions)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nemba_eval_synthetic", os.path.join(ROOT, "embedders_backbones", "synthetic.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                 # required by @dataclass
    spec.loader.exec_module(mod)
    return mod.make_dataset(n_neurons=N_NEURONS, duration=DURATION_S, dt=DT, seed=seed)


def evaluate(Z, ds, times, dec_mod):
    """Common evaluator (v0.2 decoding + state recovery) for any embedding."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    beh = {k: dec_mod.align_behavior(v, ds["times"], times,
                                     "circular" if k == "phase" else
                                     "classification" if k == "state" else "regression")
           for k, v in ds["behavior"].items()}
    out = {}
    for name, task in (("phase", "circular"), ("speed", "regression"), ("state", "classification")):
        r = dec_mod.decode_target(Z, beh[name], task=task, model=DECODE_MODEL, n_folds=N_FOLDS,
                                  n_perm=0, smooth_sigma=4.0)
        out[f"dec_{name}"] = r["score"]
    lab = KMeans(K_TRUE, n_init=10, random_state=0).fit_predict(Z)
    out["ari_states"] = adjusted_rand_score(beh["state"], lab)
    out["nmi_states"] = normalized_mutual_info_score(beh["state"], lab)
    return out


def run_one(pkg, version, backbone, seed, ds, dec_mod, label=None, kw=None):
    """Fit one backbone with one version; return timing + evaluation row."""
    t0 = time.perf_counter()
    if version == "v0.1":
        emb = pkg.Embedder(backbone, epochs=EPOCHS_V01, bin_size=DT, n_states=K_TRUE,
                           log=None, seed=0)
    else:
        emb = pkg.Embedder(backbone, bin_size=DT, n_states=K_TRUE, log=None, seed=0, **(kw or {}))
    emb.fit_spikes(ds["spike_times"])
    t_fit = time.perf_counter() - t0
    if version != "v0.1":
        t_fit = emb.timings["total_s"] - emb.timings.get("cluster_s", 0.0)
    else:                                   # v0.1 has no timings: subtract re-timed clustering
        tc = time.perf_counter()
        pkg.analysis.cluster(backbone, emb.Z, K_TRUE, 0)
        t_fit -= time.perf_counter() - tc
    row = dict(version=label or version, backbone=backbone, seed=seed,
               fit_seconds=t_fit, n_frames=len(emb.Z))
    row.update(evaluate(emb.Z, ds, emb.times, dec_mod))
    return row


def worker(kind, old_path, raw_csv):
    """Run every pending (backbone, seed) for ONE version in this process."""
    pkg = load_package(old_path if kind == "old" else ROOT)
    dec_mod = load_evaluator()
    rows = pd.read_csv(raw_csv).to_dict("records") if os.path.exists(raw_csv) else []
    done = {(r["version"], r["backbone"], int(r["seed"])) for r in rows}
    for s in SEEDS:
        ds = make_dataset(s)
        for b in BACKBONES:
            configs = [("v0.1", {}, None)] if kind == "old" else                 [(lab, kw, which) for lab, (kw, which) in V02_CONFIGS.items()]
            for ver, kw, which in configs:
                if b == "pca" and (kind == "old" or which is not None):
                    continue                       # v0.1 has no PCA; PCA has no objective
                if isinstance(which, list) and b not in which:
                    continue
                if (ver, b, s) in done:
                    continue
                t0 = time.perf_counter()
                row = run_one(pkg, "v0.1" if kind == "old" else "v0.2", b, s, ds, dec_mod, ver, kw)
                rows = pd.read_csv(raw_csv).to_dict("records") if os.path.exists(raw_csv) else []
                rows.append(row)
                pd.DataFrame(rows).to_csv(raw_csv, index=False)
                print(f"{row['version']:14s} {b:8s} seed={s} fit={row['fit_seconds']:7.1f}s "
                      f"phase={row['dec_phase']:.3f} speed={row['dec_speed']:.3f} "
                      f"state={row['dec_state']:.3f} ARI={row['ari_states']:.3f} "
                      f"(wall {time.perf_counter() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- #
# statistics                                                                   #
# --------------------------------------------------------------------------- #

def paired_stats(d, name, rng):
    """Paired test on differences d (assumption-checked) + effect sizes."""
    from scipy import stats
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    res = dict(metric=name, n_pairs=n, mean_diff=d.mean(), median_diff=np.median(d))
    boots = rng.choice(d, (BOOT, n), replace=True).mean(1)
    res["ci95_low"], res["ci95_high"] = np.percentile(boots, [2.5, 97.5])
    sh = stats.shapiro(d) if n >= 3 else None
    res["shapiro_p"] = sh.pvalue if sh else np.nan
    res["t_p"] = stats.ttest_rel(d, np.zeros(n)).pvalue if n > 1 else np.nan
    nz = d[d != 0]
    w = stats.wilcoxon(nz) if len(nz) > 0 else None
    res["wilcoxon_p"] = w.pvalue if w else np.nan
    res["test_used"] = "paired t" if (sh is not None and sh.pvalue > ALPHA) else "Wilcoxon"
    res["p"] = res["t_p"] if res["test_used"] == "paired t" else res["wilcoxon_p"]
    r = stats.rankdata(np.abs(nz))
    res["rank_biserial"] = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum() if len(nz) else 0.0
    res["cohen_dz"] = d.mean() / (d.std(ddof=1) + 1e-12) if n > 1 else np.nan
    return res


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


# --------------------------------------------------------------------------- #
# figures                                                                      #
# --------------------------------------------------------------------------- #

SHORT = {"v0.2-fast": "fast", "v0.2-fast-contrastive": "contr.",
         "v0.2-fast-hybrid": "hybrid", "v0.2-balanced": "bal."}


def make_figures(df, stats_df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    pal = {"v0.1": "#8a8984", "v0.2-fast": "#2a78d6", "v0.2-fast-contrastive": "#1baf7a",
           "v0.2-fast-hybrid": "#eda100", "v0.2-balanced": "#eb6834"}
    rc = {"font.size": FONT, "axes.titlesize": FONT + 1, "axes.spines.top": False,
          "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
          "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
          "svg.fonttype": "none", "pdf.fonttype": 42}
    versions = [v for v in ["v0.1"] + list(V02_CONFIGS) if v in set(df.version)]
    bbs = [b for b in BACKBONES if b in set(df.backbone)]
    metrics = [("fit_seconds", "fit time (s, log)"), ("dec_phase", "phase decoding\nmean cos(Δθ)"),
               ("dec_speed", "speed decoding R²"), ("dec_state", "state decoding\nbalanced acc."),
               ("ari_states", "state recovery ARI\n(KMeans K=3)")]
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(FIG_W, FIG_H + 0.9), layout="constrained")
        axes = fig.subplots(len(metrics), 1, sharex=True)
        x = np.arange(len(bbs))
        w = 0.8 / len(versions)
        for ax, (m, lab) in zip(axes, metrics):
            for j, v in enumerate(versions):
                sub = df[df.version == v]
                means = [sub[sub.backbone == b][m].mean() for b in bbs]
                xs = x - 0.4 + w * (j + 0.5)
                ax.bar(xs, means, width=w * 0.8, color=pal[v], label=v)
                for xi, b in zip(xs, bbs):
                    pts = sub[sub.backbone == b][m].to_numpy()
                    ax.plot(np.full(len(pts), xi), pts, "o", ms=2, color="#0b0b0b", alpha=0.6, mec="none")
            if m == "fit_seconds":
                ax.set_yscale("log")
            ax.set_ylabel(lab)
            ax.grid(True, axis="y", color="#e6e5e1", lw=0.5); ax.set_axisbelow(True)
        axes[-1].set_xticks(x); axes[-1].set_xticklabels(bbs)
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncols=len(versions))
        fig.suptitle("NEMBA v0.1 vs v0.2 on synthetic ground truth (bars = mean, dots = seeds)",
                     fontsize=FONT + 1.5)
        for f in FIG_FORMATS:
            fig.savefig(os.path.join(outdir, f"benchmark_per_backbone.{f}"), dpi=300, bbox_inches="tight")

        # paired differences with test annotations
        fig = Figure(figsize=(FIG_W, 2.6), layout="constrained")
        comps = sorted(stats_df.comparison.unique())
        axes = fig.subplots(1, len(metrics))
        for ax, (m, lab) in zip(axes, metrics):
            for j, c in enumerate(comps):
                r = stats_df[(stats_df.comparison == c) & (stats_df.metric == m)]
                if not len(r):
                    continue
                r = r.iloc[0]
                col = pal[c.split(" - ")[0]]
                ax.errorbar(j, r.mean_diff, yerr=[[r.mean_diff - r.ci95_low], [r.ci95_high - r.mean_diff]],
                            fmt="o", ms=4, color=col, elinewidth=1.0, capsize=2)
                ax.text(j, r.ci95_high, f"\np={r.p_holm:.2g}" if r.p_holm >= 0.001 else "\np<0.001",
                        ha="center", va="bottom", fontsize=FONT - 1.5)
            ax.axhline(0, color="#8a8984", lw=0.6)
            ax.set_xticks(range(len(comps))); ax.set_xticklabels([SHORT.get(c.split(" - ")[0], c) for c in comps],
                               rotation=45, ha="right")
            ttl = lab.replace("\n", " ")
            ax.set_title(("log10 time ratio" if m == "fit_seconds" else ttl), fontsize=FONT)
            ax.margins(x=0.4)
        fig.suptitle("paired difference v0.2 - v0.1 (mean, 95% bootstrap CI, Holm p)", fontsize=FONT + 1)
        for f in FIG_FORMATS:
            fig.savefig(os.path.join(outdir, f"benchmark_paired_differences.{f}"), dpi=300, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=os.environ.get("NEMBA_V01"), help="path to the v0.1 checkout")
    ap.add_argument("--quick", action="store_true", help="1 seed, 2 backbones (smoke test)")
    ap.add_argument("--stats-only", action="store_true", help="recompute stats/figures from CSV")
    ap.add_argument("--worker", choices=["old", "new"], default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    global SEEDS, BACKBONES, OUT
    if args.quick:
        SEEDS, BACKBONES = [0], ["pca", "tcn"]
        OUT = os.path.join(HERE, "results_quick")
    os.makedirs(OUT, exist_ok=True)
    raw_csv = os.path.join(OUT, "benchmark_raw.csv")
    warnings.filterwarnings("ignore")

    if args.worker:
        worker(args.worker, args.old, raw_csv)
        return
    if not args.stats_only:
        import subprocess
        extra = ["--quick"] if args.quick else []
        for kind in ("new", "old"):             # sequential: no CPU contention between versions
            if kind == "old" and not args.old:
                print("no --old path: skipping v0.1 runs"); continue
            subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__),
                            "--worker", kind] + (["--old", args.old] if args.old else []) + extra,
                           check=True)

    df = pd.read_csv(raw_csv)
    rng = np.random.default_rng(0)
    stat_rows = []
    metric_cols = ["fit_seconds", "dec_phase", "dec_speed", "dec_state", "ari_states", "nmi_states"]
    for lab in V02_CONFIGS:
        a = df[df.version == lab].set_index(["backbone", "seed"])
        b = df[df.version == "v0.1"].set_index(["backbone", "seed"])
        idx = a.index.intersection(b.index)
        if not len(idx):
            continue
        for m in metric_cols:
            if m == "fit_seconds":
                d = np.log10(a.loc[idx, m].to_numpy() / b.loc[idx, m].to_numpy())
            else:
                d = a.loc[idx, m].to_numpy() - b.loc[idx, m].to_numpy()
            r = paired_stats(d, m, rng)
            r["comparison"] = f"{lab} - v0.1"
            stat_rows.append(r)
    st = pd.DataFrame(stat_rows)
    if len(st):
        st["p_holm"] = np.nan
        for c in st.comparison.unique():
            sel = st.comparison == c
            st.loc[sel, "p_holm"] = holm(st.loc[sel, "p"].to_numpy())
        st.to_csv(os.path.join(OUT, "benchmark_paired_stats.csv"), index=False)
    summary = df.groupby(["version", "backbone"])[metric_cols].agg(["mean", "std"]).round(4)
    summary.to_csv(os.path.join(OUT, "benchmark_summary.csv"))
    with open(os.path.join(OUT, "benchmark_params.json"), "w") as fh:
        json.dump(dict(BACKBONES=BACKBONES, SEEDS=SEEDS, N_NEURONS=N_NEURONS, DURATION_S=DURATION_S,
                       DT=DT, EPOCHS_V01=EPOCHS_V01, V02_CONFIGS={k: v[0] for k, v in V02_CONFIGS.items()}, DECODE_MODEL=DECODE_MODEL,
                       N_FOLDS=N_FOLDS, K_TRUE=K_TRUE, BOOT=BOOT), fh, indent=2)
    if len(st):
        make_figures(df, st, OUT)
        print("\n", st[["comparison", "metric", "n_pairs", "mean_diff", "ci95_low", "ci95_high",
                        "shapiro_p", "test_used", "p", "p_holm", "rank_biserial"]].round(4).to_string(index=False))
    print(f"\nresults in {OUT}")


if __name__ == "__main__":
    main()
