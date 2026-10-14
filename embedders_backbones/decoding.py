"""Cross-validated decoding of behaviour from embeddings (time-series aware).

Why not plain K-fold
--------------------
Neighbouring frames are strongly autocorrelated (and embeddings are smoothed),
so random folds put near-duplicates of every test frame in the training set
and inflate scores, especially for flexible decoders such as kNN. Here:

* folds are **contiguous blocks** of time, with a **purge gap** removed on
  both sides of every test block (``gap="auto"`` estimates it from the label
  autocorrelation and the embedding smoothing);
* significance uses a **circular-shift null**: the label series is rolled by a
  random offset, which keeps the autocorrelation of both signals but destroys
  their alignment (a frame-shuffle null would be anti-conservative);
* p-values across several targets are **Holm-corrected**;
* the same pipeline can be run on the raw input (PCA) as a **baseline**, so an
  embedding is only credited with what it adds.

Decoders
--------
``linear``   regression: ridge (alpha by GCV on the real labels, then fixed
             for the null); classification: shrinkage LDA (closed form, fast).
``knn``      k-nearest neighbours (non-linear, CEBRA-style evaluation).
``logistic`` multinomial logistic regression (slower null; classification).

Tasks: ``regression`` (R^2), ``classification`` (balanced accuracy, chance
1/K) and ``circular`` (angles in radians; decoded through sin/cos; score =
mean cos(error), chance 0; also mean absolute error in degrees).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------------------------------- #
# user-configurable defaults                                                   #
# --------------------------------------------------------------------------- #
N_FOLDS = 5                                  # contiguous CV blocks
N_PERM = 200                                 # circular-shift null size
MAX_CLASSES = 20                             # <= this many integer levels -> classification
RIDGE_ALPHAS = np.logspace(-3, 4, 15)        # ridge penalty grid (GCV)
KNN_K = 25                                   # neighbours for the kNN decoder
MIN_SHIFT_FRAC = 0.1                         # null shifts stay >= 10 % of T from 0


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def infer_task(y, max_classes=MAX_CLASSES):
    """'classification' for bool / integer-like labels with few levels, else
    'regression'. Pass the task explicitly when in doubt."""
    y = np.asarray(y)
    if y.dtype.kind == "b":
        return "classification"
    if y.dtype.kind in "iu":
        return "classification" if np.unique(y).size <= max_classes else "regression"
    if y.dtype.kind == "f":
        f = y[np.isfinite(y)]
        if f.size and np.allclose(f, np.round(f)) and np.unique(f).size <= max_classes:
            return "classification"
        return "regression"
    return "classification"                     # strings / objects


def autocorr_length(y):
    """First lag at which the autocorrelation of y drops below 1/e (frames)."""
    y = np.asarray(y, float)
    if y.ndim > 1:
        return max(autocorr_length(c) for c in y.T)
    y = y - y.mean()
    n = len(y)
    if n < 4 or y.std() == 0:
        return 0
    f = np.fft.rfft(y, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / ac[0]
    below = np.flatnonzero(ac < 1.0 / np.e)
    return int(below[0]) if below.size else n


def mean_run_length(labels):
    """Average number of consecutive frames with the same label."""
    s = np.asarray(labels)
    if s.size == 0:
        return 0
    return float(len(s) / (1 + np.count_nonzero(s[1:] != s[:-1])))


def auto_gap(y, task, smooth_sigma=0.0, n_folds=N_FOLDS):
    """Purge gap (frames) on each side of a test block."""
    T = len(y)
    g = mean_run_length(y) if task == "classification" else autocorr_length(
        np.c_[np.sin(y), np.cos(y)] if task == "circular" else y)
    g = max(int(np.ceil(g)), int(np.ceil(3 * (smooth_sigma or 0))))
    return int(min(g, max(T // (2 * n_folds) - 1, 0)))


def blocked_folds(T, n_folds=N_FOLDS, gap=0):
    """Contiguous test blocks; training excludes ``gap`` frames on both sides."""
    edges = np.linspace(0, T, n_folds + 1).astype(int)
    out = []
    for k in range(n_folds):
        a, b = edges[k], edges[k + 1]
        test = np.arange(a, b)
        train = np.r_[np.arange(0, max(0, a - gap)), np.arange(min(T, b + gap), T)]
        if len(test) and len(train):
            out.append((train, test))
    return out


def align_behavior(values, values_times, times, task="regression"):
    """Resample a behaviour series onto the embedding time base.

    Continuous variables are linearly interpolated; discrete labels take the
    nearest sample; circular variables are interpolated through sin/cos."""
    v = np.asarray(values)
    vt = np.asarray(values_times, float)
    t = np.asarray(times, float)
    if task == "classification":
        idx = np.clip(np.searchsorted(vt, t), 1, len(vt) - 1)
        left_closer = (t - vt[idx - 1]) <= (vt[idx] - t)
        return v[np.where(left_closer, idx - 1, idx)]
    if task == "circular":
        s = np.interp(t, vt, np.sin(v)); c = np.interp(t, vt, np.cos(v))
        return np.arctan2(s, c)
    return np.interp(t, vt, v.astype(float))


def _standardize(Ztr, Zte):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    return (Ztr - mu) / sd, (Zte - mu) / sd


def _shift_matrix(y, shifts):
    """[T, 1 + n_shift] columns: the real labels, then circularly shifted copies."""
    return np.stack([y] + [np.roll(y, s) for s in shifts], axis=1)


def _r2_columns(Y, Yhat):
    ss_res = ((Y - Yhat) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# vectorised fold predictors (real labels + every null shift at once)         #
# --------------------------------------------------------------------------- #

def _fill_missing(yy, ok, task):
    """Complete a label series at invalid frames (used only to build the
    shifted null columns; scores are computed on valid frames only)."""
    if ok.all():
        return yy
    t = np.arange(len(yy))
    good = np.flatnonzero(ok)
    if task == "classification":                     # nearest valid label
        j = np.clip(np.searchsorted(good, t), 1, len(good) - 1)
        left = (t - good[j - 1]) <= (good[j] - t)
        return yy[np.where(left, good[j - 1], good[j])]
    if task == "circular":
        s = np.interp(t, good, np.sin(yy[good])); c = np.interp(t, good, np.cos(yy[good]))
        return np.arctan2(s, c)
    return np.interp(t, good, yy[good])


def _predict_regression(Ztr, Ytr, Zte, model, alpha=None, real_cols=(0,)):
    """Predict all target columns Ytr [n_tr, m] for Zte. Returns (Yhat, alpha).

    The ridge penalty is chosen by GCV on the *real* columns only and then
    reused for the null columns (the null must not get its own tuning)."""
    Ztr, Zte = _standardize(Ztr, Zte)
    if model == "knn":
        k = int(min(KNN_K, len(Ztr)))
        nbr = NearestNeighbors(n_neighbors=k).fit(Ztr).kneighbors(Zte, return_distance=False)
        return Ytr[nbr].mean(axis=1), None
    if alpha is None:
        alpha = float(RidgeCV(alphas=RIDGE_ALPHAS).fit(Ztr, Ytr[:, list(real_cols)]).alpha_)
    ym = Ytr.mean(0)
    A = Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1])
    W = np.linalg.solve(A, Ztr.T @ (Ytr - ym))
    return Zte @ W + ym, alpha


def _predict_classification(Ztr, Ytr, Zte, model):
    """Predict every label column of Ytr [n_tr, m] (integers) for Zte."""
    Ztr, Zte = _standardize(Ztr, Zte)
    m = Ytr.shape[1]
    out = np.empty((len(Zte), m), dtype=Ytr.dtype)
    if model == "knn":
        k = int(min(KNN_K, len(Ztr)))
        nbr = NearestNeighbors(n_neighbors=k).fit(Ztr).kneighbors(Zte, return_distance=False)
        classes = np.unique(Ytr)
        ks = np.arange(len(classes))
        for j in range(m):
            lab = np.searchsorted(classes, Ytr[:, j])[nbr]          # [n_te, k]
            votes = (lab[:, :, None] == ks).sum(1)                  # [n_te, K]
            out[:, j] = classes[votes.argmax(1)]
        return out
    for j in range(m):
        y = Ytr[:, j]
        if np.unique(y).size < 2:
            out[:, j] = y[0]
            continue
        if model == "logistic":
            clf = LogisticRegression(max_iter=500, C=1.0)
        else:
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        out[:, j] = clf.fit(Ztr, y).predict(Zte)
    return out


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class DecodingResult:
    table: pd.DataFrame                                  # one row per (target, features)
    predictions: dict = field(default_factory=dict)      # target -> dict(y, y_hat, task)
    nulls: dict = field(default_factory=dict)            # target -> null scores
    folds: list = field(default_factory=list)

    def to_csv(self, path):
        self.table.to_csv(path, index=False)
        return path


def decode_target(Z, y, *, task="auto", model="linear", n_folds=N_FOLDS, gap="auto",
                  n_perm=N_PERM, smooth_sigma=0.0, seed=0):
    """Decode one behaviour series y [T] from features Z [T, D].

    Returns a dict with the pooled out-of-fold score, per-fold scores, the
    null distribution, a permutation p-value and out-of-fold predictions."""
    Z = np.asarray(Z, np.float64)
    y = np.asarray(y)
    T = len(Z)
    if len(y) != T:
        raise ValueError(f"behaviour has {len(y)} samples but the embedding has {T} frames; "
                         "pass behavior_times to resample it")
    ok = np.isfinite(y.astype(float)) if y.dtype.kind in "fiub" else np.ones(T, bool)
    task = infer_task(y[ok]) if task == "auto" else task
    if task == "classification":
        classes, y_int = np.unique(y[ok], return_inverse=True)
        yy = np.full(T, -1, dtype=np.int64); yy[ok] = y_int
    else:
        yy = y.astype(float)
    g = auto_gap(yy[ok], task, smooth_sigma, n_folds) if gap == "auto" else int(gap)
    folds = blocked_folds(T, n_folds, g)

    rng = np.random.default_rng(seed)
    lo = max(1, int(MIN_SHIFT_FRAC * T))
    shifts = rng.integers(lo, T - lo, size=n_perm) if (n_perm and T - lo > lo) else np.array([], int)

    yfull = _fill_missing(yy, ok, task)
    if task == "circular":
        Yall = np.concatenate([_shift_matrix(np.sin(yfull), shifts),
                               _shift_matrix(np.cos(yfull), shifts)], 1)
    else:
        Yall = _shift_matrix(yfull, shifts)
    m = 1 + len(shifts)
    real_cols = (0, m) if task == "circular" else (0,)

    pred = np.full((T, Yall.shape[1]), np.nan)
    fold_scores = []
    for tr, te in folds:
        tr = tr[ok[tr]]
        te_ok = te[ok[te]]
        if len(tr) < 5 or len(te_ok) < 2:
            continue
        if task == "classification":
            pred[te_ok] = _predict_classification(Z[tr], Yall[tr], Z[te_ok], model)
        else:
            pred[te_ok], _ = _predict_regression(Z[tr], Yall[tr], Z[te_ok], model,
                                                 real_cols=real_cols)
        fold_scores.append(_score(task, Yall[te_ok], pred[te_ok], m)[0])

    sel = ok & np.isfinite(pred[:, 0])
    scores = _score(task, Yall[sel], pred[sel], m)
    obs, null = float(scores[0]), np.asarray(scores[1:], float)
    p = float((1 + np.sum(null >= obs)) / (1 + len(null))) if len(null) else float("nan")

    out = dict(task=task, model=model, score=obs, fold_scores=[float(s) for s in fold_scores],
               null=null, p_value=p, gap=g, n_frames=int(sel.sum()), n_folds=len(folds),
               metric={"regression": "r2", "classification": "balanced_accuracy",
                       "circular": "mean_cos_dtheta"}[task])
    if task == "classification":
        out["chance"] = 1.0 / max(len(np.unique(yy[sel])), 1)
        out["y"] = classes[yy[sel]]; out["y_hat"] = classes[pred[sel, 0].astype(int)]
    elif task == "circular":
        th = np.arctan2(pred[sel, 0], pred[sel, m])
        err = np.angle(np.exp(1j * (th - yy[sel])))
        out["chance"] = 0.0
        out["mae_deg"] = float(np.degrees(np.abs(err).mean()))
        out["y"] = yy[sel]; out["y_hat"] = th
    else:
        out["chance"] = 0.0
        out["pearson_r"] = float(np.corrcoef(yy[sel], pred[sel, 0])[0, 1]) if sel.sum() > 2 else float("nan")
        out["y"] = yy[sel]; out["y_hat"] = pred[sel, 0]
    out["frames"] = np.flatnonzero(sel)
    return out


def _score(task, Y, Yhat, m):
    """Score of every column block. Returns array [m] (real first, then nulls)."""
    if task == "classification":
        return np.array([balanced_accuracy_score(Y[:, j], Yhat[:, j].astype(Y.dtype))
                         for j in range(m)])
    if task == "circular":
        th = np.arctan2(Yhat[:, :m], Yhat[:, m:])
        tr = np.arctan2(Y[:, :m], Y[:, m:])
        return np.cos(th - tr).mean(0)
    return _r2_columns(Y, Yhat)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    n = len(pv)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def raw_baseline_features(X, n_components, smooth_sigma=0.0, seed=0):
    """Baseline features: PCA of the model input X [C, T], smoothed like the
    embeddings, so the comparison isolates what the encoder adds."""
    k = int(min(n_components, X.shape[0], X.shape[1]))
    P = PCA(k, random_state=seed).fit_transform(np.asarray(X, float).T)
    if smooth_sigma and smooth_sigma > 0:
        P = gaussian_filter1d(P, smooth_sigma, axis=0)
    return (P - P.mean(0)) / (P.std(0) + 1e-8)


def decode_behavior(Z, behavior, *, tasks=None, circular=(), model="linear",
                    n_folds=N_FOLDS, gap="auto", n_perm=N_PERM, smooth_sigma=0.0,
                    baseline=None, seed=0, feature_name="embedding"):
    """Decode every behaviour variable from Z (and optionally a baseline).

    behavior : array [T], array [T, k] or dict {name: array [T]}
    tasks    : optional dict {name: 'regression'|'classification'|'circular'}
    circular : names of angular variables (radians)
    baseline : optional feature matrix [T, D] decoded identically (e.g. raw PCA)
    """
    beh = _as_dict(behavior)
    tasks = dict(tasks or {})
    for name in circular or ():
        tasks[name] = "circular"
    if n_perm and len(beh) / (n_perm + 1) > 0.05:
        import warnings
        warnings.warn(f"n_perm={n_perm} is too small for {len(beh)} targets: the smallest "
                      f"Holm-corrected p-value is {len(beh) / (n_perm + 1):.3f} > 0.05. "
                      f"Use n_perm >= {int(np.ceil(20 * len(beh)))}.", stacklevel=2)
    feats = {feature_name: Z}
    if baseline is not None:
        feats["raw_pca"] = baseline
    rows, preds, nulls = [], {}, {}
    for fname, F in feats.items():
        for name, y in beh.items():
            r = decode_target(F, y, task=tasks.get(name, "auto"), model=model, n_folds=n_folds,
                              gap=gap, n_perm=n_perm, smooth_sigma=smooth_sigma, seed=seed)
            null = r["null"]
            rows.append(dict(target=name, features=fname, task=r["task"], model=model,
                             metric=r["metric"], score=r["score"],
                             score_sd=float(np.std(r["fold_scores"])) if r["fold_scores"] else np.nan,
                             fold_scores=json.dumps([round(s, 5) for s in r["fold_scores"]]),
                             chance=r["chance"],
                             null_mean=float(null.mean()) if null.size else np.nan,
                             null_p95=float(np.quantile(null, 0.95)) if null.size else np.nan,
                             p_value=r["p_value"], n_perm=int(null.size),
                             mae_deg=r.get("mae_deg", np.nan), pearson_r=r.get("pearson_r", np.nan),
                             n_frames=r["n_frames"], n_folds=r["n_folds"], gap=r["gap"],
                             n_features=int(np.asarray(F).shape[1])))
            if fname == feature_name:
                preds[name] = dict(y=r["y"], y_hat=r["y_hat"], frames=r["frames"], task=r["task"])
                nulls[name] = null
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = np.nan
        for fname in table["features"].unique():
            sel = table["features"] == fname
            table.loc[sel, "p_holm"] = holm(table.loc[sel, "p_value"].to_numpy())
        if baseline is not None:
            base = table[table.features == "raw_pca"].set_index("target")["score"]
            table["delta_vs_raw"] = table.apply(
                lambda r: r["score"] - base.get(r["target"], np.nan), axis=1)
    return DecodingResult(table=table, predictions=preds, nulls=nulls)


def _as_dict(behavior):
    if isinstance(behavior, dict):
        return {str(k): np.asarray(v) for k, v in behavior.items()}
    if isinstance(behavior, pd.DataFrame):
        return {str(c): behavior[c].to_numpy() for c in behavior.columns}
    b = np.asarray(behavior)
    if b.ndim == 1:
        return {"behavior": b}
    if b.ndim == 2:
        if b.shape[0] < b.shape[1]:                 # [k, T] -> [T, k]
            b = b.T
        return {f"behavior_{i}": b[:, i] for i in range(b.shape[1])}
    raise ValueError(f"behaviour must be 1-D, 2-D or a dict, got shape {b.shape}")
