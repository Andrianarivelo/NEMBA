"""Validate the temporal state models of ``embedders_backbones.states`` against
frame-wise K-means (what NEMBA v0.2 does in ``analysis.cluster``).

What is compared
----------------
Three ways of turning a feature matrix Z [T, D] into a state sequence:
  * kmeans : K-means on single frames, K = n_init 10 (as ``analysis.cluster``);
             automatic K by the KMeans silhouette (as ``analysis.auto_k``);
  * hmm    : sticky Gaussian HMM (``states.fit_states(method="hmm")``);
  * arhmm  : sticky autoregressive HMM (``states.fit_states(method="arhmm")``);
             automatic K for both HMMs by blocked CV + 1-SE rule (``select_k``).

Conditions (each repeated over SEEDS, a new simulated recording per seed)
  * hmm_k4 / hmm_auto : data drawn from a sticky 4-state Gaussian HMM in 8-D
    with heavily overlapping emissions (the model-matched case);
  * nemba_k3 / nemba_auto : the NEMBA synthetic recording (``synthetic.make_dataset``,
    60 neurons, 600 s, dt 0.1 s, 3 true behavioural states); features = top 10
    PCs of z-scored sqrt(counts), Gaussian-smoothed (sigma 4 frames), z-scored;
  * nemba_raw_k3 : same without the smoothing (diagnostic: smoothing is a
    temporal prior that frame-wise K-means needs and the HMMs do not).

Metrics: ARI and NMI against the true states, number of transitions / true
number, median dwell time / true median dwell, fit seconds, K used.

Statistics
----------
Paired design (same recording for every method). For each condition, metric
and pair of methods, the per-seed differences are tested for normality
(Shapiro-Wilk); the paired t-test and the Wilcoxon signed-rank test are both
reported, and ``test_used`` is the t-test when Shapiro-Wilk does not reject
(p > ALPHA), else Wilcoxon. Holm correction over the 3 comparisons of each
(condition, metric). With n = 5 seeds the exact two-sided Wilcoxon p-value
cannot go below 0.0625: read the effect sizes (mean difference, 95 % CI,
Cohen's dz) together with the p-values. Transition and dwell ratios are tested
as their distance to the truth, |log2(ratio)|; fit time on log10 seconds.

A timing section measures one EM iteration at T = 100,000, K = 12, D = 32 for
both backends (numba / NumPy).

Usage
-----
    python benchmarks/validate_states.py
Outputs (benchmarks/results/): states_validation.csv (raw, one row per
condition x method x seed), states_validation_stats.csv, states_validation_timing.csv,
states_validation_params.json, states_validation.{png,pdf,svg}.
"""

from __future__ import annotations

# =========================================================================== #
# PARAMETERS (edit here)                                                      #
# =========================================================================== #
SEEDS = [0, 1, 2, 3, 4]                    # one simulated recording per seed
METHODS = ["kmeans", "hmm", "arhmm"]
# (a) sticky Gaussian HMM synthetic data
HMM_T, HMM_K, HMM_D = 6000, 4, 8           # frames, states, dimensions
HMM_P_STAY = 0.98                          # self-transition probability (mean dwell 50 frames)
HMM_SEP = 0.5                              # state means ~ N(0, HMM_SEP^2 I)
HMM_NOISE = 1.0                            # emission noise SD (overlapping states)
# (b) NEMBA synthetic recording
N_NEURONS, DURATION_S, DT = 60, 600.0, 0.1
N_PCS = 10                                 # PCs of z-scored sqrt(counts)
SMOOTH_SIGMA = 4.0                         # Gaussian smoothing (frames) of the PCs
# conditions: data source, smoothing (NEMBA only), K (None = automatic)
CONDITIONS = {
    "hmm_k4":       dict(data="hmm", sigma=None, k=4, label="Sticky HMM\nK = 4"),
    "hmm_auto":     dict(data="hmm", sigma=None, k=None, label="Sticky HMM\nauto K"),
    "nemba_k3":     dict(data="nemba", sigma=SMOOTH_SIGMA, k=3, label="NEMBA, σ = 4\nK = 3"),
    "nemba_auto":   dict(data="nemba", sigma=SMOOTH_SIGMA, k=None, label="NEMBA, σ = 4\nauto K"),
    "nemba_raw_k3": dict(data="nemba", sigma=0.0, k=3, label="NEMBA, σ = 0\nK = 3"),
}
K_RANGE = (2, 10)                          # automatic K search range (NEMBA k_min / k_max defaults)
N_FOLDS = 3                                # blocked CV folds of select_k
SE_RULE = "fold"                           # 1-SE rule variant of select_k
STATE_KW = {}                              # extra fit_states kwargs (n_restarts, sticky, ...)
KMEANS_N_INIT = 10                         # as analysis.cluster
SIL_SAMPLE = 3000                          # frames for the silhouette (as analysis.auto_k)
# timing of one EM iteration on a large problem
TIMING = True
TIMING_T, TIMING_K, TIMING_D = 100_000, 12, 32
TIMING_REPEATS = 3
# statistics
ALPHA = 0.05
COMPARISONS = [("hmm", "kmeans"), ("arhmm", "kmeans"), ("arhmm", "hmm")]
STAT_METRICS = ["ari", "nmi", "trans_err", "dwell_err", "log10_fit_s"]
# figure
FIG_FORMATS = ("png", "pdf", "svg")
FIG_W, ROW_H = 7.2, 1.5                    # inches
FONT = 7
DPI = 300
COLORS = {"kmeans": "#2a78d6", "hmm": "#eb6834", "arhmm": "#1baf7a"}   # categorical slots 1-3
NAMES = {"kmeans": "K-means (frame-wise)", "hmm": "Gaussian HMM", "arhmm": "AR-HMM"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BAR_W = 0.12                               # thin bars (group spacing 1)
# figure rows: (raw metric, axis label, scale, stats metric used for the p-values)
FIG_METRICS = [("ari", "ARI vs true states", "linear", "ari"),
               ("trans_ratio", "transitions / true", "log", "trans_err"),
               ("dwell_ratio", "median dwell / true", "log", "dwell_err"),
               ("fit_s", "fit time (s)", "log", "log10_fit_s")]
# =========================================================================== #

import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")
sys.path.insert(0, ROOT)

from embedders_backbones import states as ST            # noqa: E402
from embedders_backbones.synthetic import make_dataset   # noqa: E402


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #

def sticky_hmm_data(seed):
    """Sticky Gaussian HMM: Markov chain with self-transition HMM_P_STAY (jumps
    uniform over the other states), frames = state mean + isotropic noise."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, HMM_SEP, (HMM_K, HMM_D))
    s = np.empty(HMM_T, dtype=int)
    s[0] = rng.integers(HMM_K)
    for t in range(1, HMM_T):
        if rng.random() < HMM_P_STAY:
            s[t] = s[t - 1]
        else:
            s[t] = rng.choice([k for k in range(HMM_K) if k != s[t - 1]])
    return means[s] + HMM_NOISE * rng.standard_normal((HMM_T, HMM_D)), s


def nemba_data(seed, sigma):
    """NEMBA synthetic recording -> features (top PCs of z-scored sqrt counts,
    optionally Gaussian-smoothed, z-scored) and the true behavioural state."""
    d = make_dataset(n_neurons=N_NEURONS, duration=DURATION_S, dt=DT, seed=seed)
    X = np.sqrt(d["counts"].T.astype(np.float64))                  # [T, N] variance-stabilised
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    P = PCA(N_PCS, random_state=0).fit_transform(X)
    if sigma:
        P = gaussian_filter1d(P, sigma=sigma, axis=0)
    return (P - P.mean(0)) / (P.std(0) + 1e-8), np.asarray(d["behavior"]["state"], int)


# --------------------------------------------------------------------------- #
# methods                                                                     #
# --------------------------------------------------------------------------- #

def silhouette_auto_k(Z, k_min, k_max, seed, sample=SIL_SAMPLE):
    """Frozen copy of NEMBA v0.2 ``analysis.auto_k`` (commit 99177cb): K that
    maximises the KMeans silhouette on a random subsample. Kept local so the
    baseline does not move when ``analysis`` is changed."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), sample, replace=False) if len(Z) > sample else np.arange(len(Z))
    Zs = Z[idx]
    best_k, best_s = k_min, -1.0
    for k in range(k_min, min(k_max, len(Zs) - 1) + 1):
        lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(Zs)
        s = float(silhouette_score(Zs, lab))
        if s > best_s:
            best_s, best_k = s, k
    return best_k


def run_method(method, Z, k, seed):
    """Fit one method; returns (labels, K used, seconds). k=None = automatic K."""
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "kmeans":
            k_used = silhouette_auto_k(Z, K_RANGE[0], K_RANGE[1], seed) if k is None else k
            labels = KMeans(n_clusters=k_used, n_init=KMEANS_N_INIT, random_state=seed).fit_predict(Z)
        else:
            res = ST.fit_states(Z, method=method, k=k, k_range=K_RANGE, seed=seed,
                                n_folds=N_FOLDS, se_rule=SE_RULE, **STATE_KW)
            labels, k_used = res["labels"], res["k"]
    return np.asarray(labels), int(k_used), time.perf_counter() - t0


def score(labels, truth):
    """ARI, NMI, transition ratio and median dwell ratio against the truth."""
    n_pred, n_true = ST.n_switches(labels), ST.n_switches(truth)
    dw_pred = np.median(np.concatenate(list(ST.dwell_times(labels).values())))
    dw_true = np.median(np.concatenate(list(ST.dwell_times(truth).values())))
    tr = max(n_pred, 1) / max(n_true, 1)                 # floor 1: a ratio stays finite
    dr = dw_pred / dw_true
    return dict(ari=adjusted_rand_score(truth, labels), nmi=normalized_mutual_info_score(truth, labels),
                n_trans=n_pred, n_trans_true=n_true, trans_ratio=tr,
                median_dwell=float(dw_pred), median_dwell_true=float(dw_true), dwell_ratio=float(dr),
                trans_err=abs(np.log2(tr)), dwell_err=abs(np.log2(dr)))


# --------------------------------------------------------------------------- #
# statistics                                                                  #
# --------------------------------------------------------------------------- #

def holm(p):
    """Holm-Bonferroni adjusted p-values (NaN kept as NaN)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size:
        order = ok[np.argsort(p[ok])]
        m = ok.size
        adj = np.maximum.accumulate([(m - i) * p[j] for i, j in enumerate(order)])
        out[order] = np.minimum(adj, 1.0)
    return out


def paired_test(a, b):
    """Paired comparison a - b: normality of the differences, both tests,
    the test chosen by the Shapiro-Wilk check, effect sizes."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    r = dict(n=n, mean_a=np.mean(a), mean_b=np.mean(b), mean_diff=d.mean(),
             sd_diff=d.std(ddof=1) if n > 1 else np.nan)
    se = r["sd_diff"] / np.sqrt(n)
    tq = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    r["ci95_lo"], r["ci95_hi"] = d.mean() - tq * se, d.mean() + tq * se
    r["cohen_dz"] = d.mean() / r["sd_diff"] if r["sd_diff"] > 0 else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n >= 3 and np.ptp(d) > 0:
            r["shapiro_p"] = float(stats.shapiro(d).pvalue)
        else:
            r["shapiro_p"] = np.nan
        r["t_p"] = float(stats.ttest_rel(a, b).pvalue) if r["sd_diff"] > 0 else (1.0 if d.mean() == 0 else 0.0)
        try:
            r["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            r["wilcoxon_p"] = np.nan
    r["normal"] = bool(np.isfinite(r["shapiro_p"]) and r["shapiro_p"] > ALPHA)
    r["test_used"] = "paired t" if r["normal"] else "wilcoxon"
    r["p_used"] = r["t_p"] if r["normal"] else r["wilcoxon_p"]
    return r


def statistics_table(df):
    """All paired comparisons per condition and metric, Holm within a family."""
    rows = []
    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond]
        for metric in STAT_METRICS:
            fam = []
            for a, b in COMPARISONS:
                A = sub[sub["method"] == a].set_index("seed")[metric]
                B = sub[sub["method"] == b].set_index("seed")[metric]
                common = A.index.intersection(B.index)
                if len(common) < 2:
                    continue
                r = paired_test(A.loc[common].values, B.loc[common].values)
                fam.append(dict(condition=cond, metric=metric, comparison=f"{a} - {b}", **r))
            for r, ph in zip(fam, holm([r["p_used"] for r in fam])):
                r["p_holm"] = ph
                r["significant"] = bool(np.isfinite(ph) and ph < ALPHA)
            rows += fam
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# timing on a large problem                                                   #
# --------------------------------------------------------------------------- #

def timing_table():
    """Median E-step, M-step and Viterbi seconds at T x K x D = TIMING_T x
    TIMING_K x TIMING_D, for both models and both backends."""
    rng = np.random.default_rng(123)
    means = rng.normal(0.0, 1.0, (TIMING_K, TIMING_D))
    s = np.repeat(rng.integers(TIMING_K, size=TIMING_T // 50 + 1), 50)[:TIMING_T]
    Z = means[s] + rng.standard_normal((TIMING_T, TIMING_D))
    rows = []
    backends = ["numba", "numpy"] if ST.HAS_NUMBA else ["numpy"]
    for name, cls in (("hmm", ST.GaussianHMM), ("arhmm", ST.ARHMM)):
        for backend in backends:
            m = cls(TIMING_K, n_restarts=1, max_iter=1, backend=backend, seed=0).fit(Z)
            prep = m._prepare([Z])
            e_s, m_s = [], []
            for _ in range(TIMING_REPEATS):
                t0 = time.perf_counter()
                gammas, xi, start, _ll = m._e_step(prep)
                e_s.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                m._m_step_emission(prep, gammas)
                m._m_step_trans(xi, start)
                m_s.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            m.predict(Z)
            v_s = time.perf_counter() - t0
            rows.append(dict(model=name, backend=backend, T=TIMING_T, K=TIMING_K, D=TIMING_D,
                             e_step_s=np.median(e_s), m_step_s=np.median(m_s),
                             em_iteration_s=np.median(e_s) + np.median(m_s), viterbi_s=v_s))
            print(f"  timing {name:5s} {backend:5s}: E {rows[-1]['e_step_s']:.2f}s  "
                  f"M {rows[-1]['m_step_s']:.2f}s  Viterbi {v_s:.2f}s", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figure                                                                      #
# --------------------------------------------------------------------------- #

def fmt_p(p):
    """Compact p-value text."""
    if not np.isfinite(p):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def make_figure(df, st):
    """One row per metric; groups = conditions; thin bars = mean, dots = seeds;
    brackets = paired tests against K-means (p of the test chosen by Shapiro-Wilk)."""
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory

    conds = list(CONDITIONS)
    rc = {"font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT - 0.5,
          "ytick.labelsize": FONT - 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
          "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
          "xtick.major.width": 0.6, "ytick.major.width": 0.6, "svg.fonttype": "none",
          "pdf.fonttype": 42, "font.family": "DejaVu Sans"}
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(FIG_W, ROW_H * len(FIG_METRICS) + 0.7), layout="constrained")
        axes = fig.subplots(len(FIG_METRICS), 1, sharex=True)
        rng = np.random.default_rng(0)
        offs = {m: (i - (len(METHODS) - 1) / 2) * (BAR_W + 0.07) for i, m in enumerate(METHODS)}
        for ax, (metric, ylabel, scale, smetric) in zip(axes, FIG_METRICS):
            vals = df[metric].values.astype(float)
            if scale == "log":
                ax.set_yscale("log")
                lo, hi = np.log10(np.nanmin(vals)), np.log10(np.nanmax(vals))
                lo, hi = min(lo, -0.1), max(hi, 0.1)            # keep the reference line visible
                span = hi - lo
                ax.set_ylim(10 ** (lo - 0.08 * span), 10 ** (hi + 0.62 * span))
                base = 10 ** (lo - 0.08 * span)
            else:
                ax.set_ylim(0, 1.62)
                ax.set_yticks([0, 0.5, 1.0])
                base = 0.0
            ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if metric in ("trans_ratio", "dwell_ratio"):
                ax.axhline(1.0, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
            for gi, cond in enumerate(conds):
                sub = df[df["condition"] == cond]
                for m in METHODS:
                    v = sub[sub["method"] == m][metric].values.astype(float)
                    if v.size == 0:
                        continue
                    x = gi + offs[m]
                    centre = 10 ** np.mean(np.log10(v)) if scale == "log" else v.mean()  # geometric mean on log axes
                    ax.bar(x, centre - base, bottom=base, width=BAR_W, color=COLORS[m], alpha=0.35,
                           edgecolor=COLORS[m], linewidth=0.8, zorder=2)
                    ax.scatter(x + rng.uniform(-0.05, 0.05, v.size), v, s=7, color=COLORS[m],
                               edgecolor="white", linewidth=0.3, zorder=3)
                # brackets: each HMM against K-means, stacked above the group
                tr = ax.transScale + ax.transLimits               # data -> axes fraction
                top = max(tr.transform((gi, y))[1] for y in sub[metric].values.astype(float))
                bt = blended_transform_factory(ax.transData, ax.transAxes)
                for level, other in enumerate(("hmm", "arhmm")):
                    row = st[(st["condition"] == cond) & (st["metric"] == smetric)
                             & (st["comparison"] == f"{other} - kmeans")]
                    if row.empty:
                        continue
                    p = float(row["p_used"].iloc[0])
                    y = top + 0.07 + level * 0.17
                    x1, x2 = gi + offs["kmeans"], gi + offs[other]
                    ax.plot([x1, x1, x2, x2], [y - 0.025, y, y, y - 0.025], transform=bt,
                            color=INK2, lw=0.5, clip_on=False, zorder=4)
                    ax.text((x1 + x2) / 2, y + 0.01, fmt_p(p), transform=bt, ha="center", va="bottom",
                            fontsize=FONT - 1.6, color=INK, clip_on=False)
            ax.set_ylabel(ylabel)
        axes[-1].set_xticks(range(len(conds)))
        axes[-1].set_xticklabels([CONDITIONS[c]["label"] for c in conds])
        axes[-1].set_xlim(-0.55, len(conds) - 0.45)
        handles = [Patch(facecolor=COLORS[m], alpha=0.6, edgecolor=COLORS[m], label=NAMES[m])
                   for m in METHODS]
        handles.append(Line2D([], [], ls="none", marker="o", ms=3, color=INK2, label=f"seeds (n = {len(SEEDS)})"))
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False,
                   fontsize=FONT, handlelength=1.2, columnspacing=1.4)
        fig.text(0.5, 0.0, "Bars: mean (geometric mean on log axes); dots: seeds. p: each HMM vs K-means, "
                 "paired t-test if Shapiro-Wilk p > 0.05 on the differences, else Wilcoxon signed-rank.\n"
                 f"n = {len(SEEDS)} seeds (exact two-sided Wilcoxon p cannot fall below {2 / 2 ** len(SEEDS):.4f}). "
                 "Transitions and dwell times tested as |log2 ratio| (distance to the truth); "
                 "fit time on log10 seconds.",
                 ha="center", va="bottom", fontsize=FONT - 1.5, color=INK2, linespacing=1.4)
        fig.get_layout_engine().set(h_pad=0.04, rect=(0, 0.035, 1, 0.965))
        for ext in FIG_FORMATS:
            fig.savefig(os.path.join(OUT, f"states_validation.{ext}"), dpi=DPI)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    """Run every condition x method x seed, then statistics, timing, figure."""
    os.makedirs(OUT, exist_ok=True)
    t_start = time.perf_counter()
    raw_path = os.path.join(OUT, "states_validation.csv")
    rows = []
    for seed in SEEDS:
        cache = {}
        for cond, spec in CONDITIONS.items():
            key = (spec["data"], spec["sigma"])
            if key not in cache:                                # data shared across conditions
                cache[key] = sticky_hmm_data(seed) if spec["data"] == "hmm" else nemba_data(seed, spec["sigma"])
            Z, truth = cache[key]
            for method in METHODS:
                labels, k_used, sec = run_method(method, Z, spec["k"], seed)
                r = dict(condition=cond, data=spec["data"], sigma=spec["sigma"], k_mode="auto" if spec["k"] is None else "fixed",
                         method=method, seed=seed, T=len(Z), D=Z.shape[1], k_true=int(truth.max() + 1),
                         k_used=k_used, fit_s=sec, **score(labels, truth))
                r["log10_fit_s"] = np.log10(sec)
                rows.append(r)
                print(f"seed {seed} {cond:13s} {method:6s} K={k_used:2d} ARI {r['ari']:.3f} NMI {r['nmi']:.3f} "
                      f"trans {r['n_trans']:4d}/{r['n_trans_true']:3d} dwell x{r['dwell_ratio']:.2f} "
                      f"{sec:6.2f}s", flush=True)
        pd.DataFrame(rows).to_csv(raw_path, index=False)       # saved after every seed
    df = pd.DataFrame(rows)
    st = statistics_table(df)
    st.to_csv(os.path.join(OUT, "states_validation_stats.csv"), index=False)

    # summary printed for the report
    summ = df.groupby(["condition", "method"]).agg(
        ari=("ari", "mean"), ari_sd=("ari", "std"), nmi=("nmi", "mean"), k_used=("k_used", "median"),
        trans_ratio=("trans_ratio", "median"), dwell_ratio=("dwell_ratio", "median"),
        fit_s=("fit_s", "median")).reset_index()
    print("\n" + summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    keep = st[st["metric"] == "ari"][["condition", "comparison", "mean_diff", "ci95_lo", "ci95_hi",
                                       "shapiro_p", "t_p", "wilcoxon_p", "test_used", "p_used", "p_holm"]]
    print("\nARI paired comparisons\n" + keep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if TIMING:
        print("\ntiming (one EM iteration)")
        timing_table().to_csv(os.path.join(OUT, "states_validation_timing.csv"), index=False)
    make_figure(df, st)
    params = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list, tuple, dict))}
    params.update(sticky_frac=ST.STICKY_FRAC, trans_pseudocount=ST.TRANS_PSEUDOCOUNT, has_numba=ST.HAS_NUMBA,
                  python=platform.python_version(), numpy=np.__version__, cpu_count=os.cpu_count(),
                  total_seconds=time.perf_counter() - t_start)
    with open(os.path.join(OUT, "states_validation_params.json"), "w") as f:
        json.dump(params, f, indent=1, default=str)
    print(f"\ndone in {time.perf_counter() - t_start:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
