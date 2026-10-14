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
  embedding is only credited with what it adds;
* when raw spike **counts** are available, a Bayesian population decoder
  gives a model-based reference on the neurons themselves.

Decoders
--------
``linear``   regression: ridge (alpha by GCV on the real labels, then fixed
             for the null); classification: shrinkage LDA (closed form, fast).
``knn``      k-nearest neighbours (non-linear, CEBRA-style evaluation).
``logistic`` multinomial logistic regression (slower null; classification).
``gbm``      histogram gradient-boosted trees (sklearn HistGradientBoosting*);
             circular targets through two regressors (sin and cos). Glaser
             et al. 2020 (eNeuro 7:ENEURO.0506-19.2020) found modern decoders
             (neural networks, gradient boosting, ensembles) to beat Wiener and
             Kalman filters on spiking data.
``bayes``    ``bayes_decode_target(counts, y)``: Poisson population decoder on
             raw spike counts (Zhang, Ginzburg, McNaughton & Sejnowski 1998,
             J Neurophysiol 79:1017, one-step Bayesian reconstruction, without
             their continuity constraint).

When to use which
-----------------
* ``linear``: the default. Fast, the full null (N_PERM shifts) is vectorised,
  and "linearly decodable" is the usual operational meaning of a variable
  being explicitly represented in an embedding.
* ``knn``: local, non-linear read-out of the embedding manifold (what a
  neighbourhood in the embedding says about behaviour). Sensitive to the
  choice of k and to the purge gap.
* ``gbm``: non-linear and non-local relations (thresholds, interactions,
  non-monotonic tuning). Strongest read-out but slowest, and its null is
  capped (see below). Use it to ask "is the information there at all", not
  "is it explicit".
* ``bayes``: spike-count data only. A principled reference of what the
  population carries under independent Poisson neurons with stationary
  tuning; it needs no feature engineering and gives a full posterior. It is
  biased when neurons are strongly correlated or over-dispersed.

gbm null-size caveat
--------------------
Boosting cannot be vectorised across null columns: every circular shift is a
separate fit. The gbm null is therefore capped at ``GBM_MAX_PERM`` (49)
shifts, so the smallest attainable p-value is 1 / 50 = 0.02 and, after Holm
correction over k targets, k / 50: with three or more targets no gbm row can
reach p_holm < 0.05. Raise ``decoding.GBM_MAX_PERM`` when needed (cost grows
linearly with it). Early stopping uses a *blocked* validation block (the last
15 % of each training fold, behind a purge gap), applied identically to the
real and the null columns, and the test fold is predicted by the model at its
best round (``GBM_REFIT=True`` refits on the whole fold instead). sklearn's
built-in early stopping draws a random validation split, which is leaky for
autocorrelated data: on shifted (null) labels it never stopped (300 of 300
rounds) in our benchmark, whereas the blocked rule ends most null fits after
a single 20-round fit, which is what keeps gbm affordable. Fits run on one
OpenMP thread (``GBM_N_THREADS``): for these small fits, more threads were
much slower on a busy CPU. Measured cost (6000 frames x 32 features, 5 folds,
49 shifts, laptop CPU shared with another heavy job): regression ~45 s,
circular ~95 s (two regressors), 3-class ~110 s (one tree per class per round).

Tasks: ``regression`` (R^2), ``classification`` (balanced accuracy, chance
1/K) and ``circular`` (angles in radians; decoded through sin/cos; score =
mean cos(error), chance 0; also mean absolute error in degrees).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
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
DECODERS = ("linear", "knn", "logistic", "gbm")   # valid ``model`` names of decode_target

# gradient boosting (model="gbm") ------------------------------------------- #
GBM_MAX_PERM = 49                            # null-size cap: each shift is a separate fit
GBM_PARAMS = dict(learning_rate=0.1,         # shrinkage per boosting round
                  max_leaf_nodes=15,         # small trees: fast, less variance
                  min_samples_leaf=20,       # >= 20 frames per leaf (autocorrelated data)
                  l2_regularization=1.0,     # ridge penalty on leaf values
                  max_bins=63)               # feature histogram resolution (speed)
GBM_MAX_ITER = 300                           # boosting rounds (upper bound)
GBM_EARLY_STOPPING = True                    # blocked early stopping (False: GBM_MAX_ITER rounds)
GBM_VAL_FRAC = 0.15                          # last 15 % of each training fold = validation block
GBM_ES_STEP = 10                             # rounds added between validation checks
GBM_ES_PATIENCE = 2                          # stop after this many checks without improvement
GBM_ES_TOL = 1e-4                            # relative validation-loss gain that counts
GBM_REFIT = False                            # refit on the whole training fold at the chosen
                                             # round count (True: +15 % data, ~1.5x slower)
GBM_N_THREADS = 1                            # OpenMP threads per fit (None = all cores). Fits
                                             # are small: 1 thread was 5x faster than 16 on a
                                             # busy CPU and only ~20 % slower on an idle one

# Bayesian population decoder (bayes_decode_target) ------------------------- #
BAYES_N_BINS = 36                            # behaviour bins (circular: 10 deg; regression: quantiles)
BAYES_PRIOR = "uniform"                      # 'uniform' (visited bins) or 'empirical' (occupancy)
BAYES_PSEUDO_FRAMES = 1.0                    # shrink tuning curves: pseudo-frames at the mean rate
BAYES_TUNING_SIGMA = 1.0                     # Gaussian smoothing of tuning curves (bins; 0 = off)
BAYES_RATE_FLOOR = 1e-6                      # minimum expected count per frame (avoids log 0)
BAYES_CHUNK = 4_000_000                      # max posterior entries (frames x columns x bins) in memory


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


def matched_count_window(smooth_sigma):
    """Odd boxcar width (frames) whose standard deviation matches a Gaussian
    kernel of ``smooth_sigma`` frames (width ~ sigma * sqrt(12)); 1 when the
    embedding is not smoothed. Used to give the Bayesian count decoder the
    same temporal integration as the smoothed embedding it is compared with."""
    return 1 + 2 * int(round(np.sqrt(3.0) * float(smooth_sigma or 0.0)))


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


def _effective_gap(folds):
    """Purge actually present in user-given folds: the smallest distance (in
    frames) between a test frame and a training frame, minus one."""
    g = np.inf
    for tr, te in folds:
        if not len(tr) or not len(te):
            continue
        s = np.sort(tr)
        i = np.searchsorted(s, te)
        left = np.abs(te - s[np.clip(i - 1, 0, len(s) - 1)])
        right = np.abs(s[np.clip(i, 0, len(s) - 1)] - te)
        g = min(g, int(np.minimum(left, right).min()) - 1)
    return int(g) if np.isfinite(g) else 0


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
# gradient boosting (one fit per column: not vectorisable)                    #
# --------------------------------------------------------------------------- #

def _gbm_n_perm(n_perm, stacklevel=3):
    """Cap the gbm null at GBM_MAX_PERM shifts (each shift is one boosting fit)."""
    if n_perm and n_perm > GBM_MAX_PERM:
        warnings.warn(f"model='gbm': null capped at GBM_MAX_PERM={GBM_MAX_PERM} circular "
                      f"shifts (requested {n_perm}); smallest attainable p-value = "
                      f"{1.0 / (GBM_MAX_PERM + 1):.3f}. Raise decoding.GBM_MAX_PERM if needed.",
                      stacklevel=stacklevel)
        return int(GBM_MAX_PERM)
    return n_perm


def _blocked_inner_split(tr_index, gap):
    """Positions (into the training rows) of an inner-training set and a
    *blocked* validation set: the last GBM_VAL_FRAC of the training frames in
    time, with ``gap`` frames purged in front of it. None if too small."""
    n_val = int(round(GBM_VAL_FRAC * len(tr_index)))
    if n_val < 20:
        return None
    cut = np.sort(tr_index)[len(tr_index) - n_val]       # first validation frame
    val = np.flatnonzero(tr_index >= cut)
    inner = np.flatnonzero(tr_index < cut - gap)
    if len(inner) < 40:
        return None
    return inner, val


def _gbm_early_stopped(make, Xi, yi, Xv, yv, classification):
    """Boost on (Xi, yi) and pick the number of rounds minimising the loss
    (squared error or log loss) on the blocked validation block (Xv, yv).

    Returns (best_round, model); best_round 0 means the constant model
    (training mean / class frequencies) is never beaten. The first call fits
    the whole patience window (GBM_ES_PATIENCE * GBM_ES_STEP rounds) so that
    uninformative columns, i.e. most null shifts, need a single fit; then
    rounds are added GBM_ES_STEP at a time with ``warm_start`` until
    GBM_ES_PATIENCE checks bring no relative gain of GBM_ES_TOL, or
    GBM_MAX_ITER is reached."""
    if classification:
        classes = np.unique(yi)
        if len(classes) < 2:
            return 0, None
        known = np.isin(yv, classes)                     # unseen classes get p = 0 (clipped)
        col = np.clip(np.searchsorted(classes, yv), 0, len(classes) - 1)
        rows = np.arange(len(yv))

        def loss(P):
            p = np.where(known, P[rows, col], 0.0)
            return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))

        freq = np.bincount(np.searchsorted(classes, yi), minlength=len(classes)) / len(yi)
        base = loss(np.broadcast_to(freq, (len(yv), len(classes))))
    else:
        if np.ptp(yi) == 0:
            return 0, None

        def loss(p):
            return float(np.mean((yv - p) ** 2))

        base = loss(np.full(len(yv), yi.mean()))
    tol = GBM_ES_TOL * abs(base)
    patience = GBM_ES_PATIENCE * GBM_ES_STEP
    best, best_it, done = base, 0, 0
    est = make(min(patience, GBM_MAX_ITER), warm=True)
    while True:
        est.fit(Xi, yi)                                  # adds rounds done+1 .. max_iter
        stages = est.staged_predict_proba(Xv) if classification else est.staged_predict(Xv)
        for k, P in enumerate(stages, start=1):
            if k <= done:                                # already evaluated
                continue
            l = loss(P)
            if l < best - tol:
                best, best_it = l, k
        done = int(est.n_iter_)
        if done >= GBM_MAX_ITER or done - best_it >= patience:
            return best_it, est
        est.set_params(max_iter=min(done + GBM_ES_STEP, GBM_MAX_ITER))


def _gbm_column(Est, Ztr, y, Zte, split, classification, seed):
    """Fit one boosted model on one label column and predict the test frames.

    With early stopping the number of rounds is chosen on the blocked inner
    split; the test fold is predicted by that early-stopped model at its best
    round (or, with GBM_REFIT, by a refit on the whole training fold)."""
    def make(n_iter, warm=False):
        return Est(max_iter=int(n_iter), early_stopping=False, warm_start=warm,
                   random_state=seed, **GBM_PARAMS)

    if classification:
        y = y.astype(np.int64)
        labels, cnt = np.unique(y, return_counts=True)
        constant, degenerate = labels[np.argmax(cnt)], len(labels) < 2   # majority class
    else:
        constant, degenerate = float(y.mean()), np.ptp(y) == 0
    if degenerate:
        return np.full(len(Zte), constant)
    if split is None:                                    # no early stopping: fixed rounds
        return make(GBM_MAX_ITER).fit(Ztr, y).predict(Zte)
    inner, val = split
    best_it, est = _gbm_early_stopped(make, Ztr[inner], y[inner], Ztr[val], y[val], classification)
    if best_it == 0:
        return np.full(len(Zte), constant)
    if GBM_REFIT:
        return make(best_it).fit(Ztr, y).predict(Zte)
    for k, p in enumerate(est.staged_predict(Zte), start=1):     # prediction at round best_it
        if k == best_it:
            return p


def _predict_gbm(Ztr, Ytr, Zte, tr_index, gap, task, seed):
    """Gradient-boosted trees for every label column of Ytr [n_tr, m'].

    One model per column (real labels and each null shift; sin and cos are two
    separate columns for circular targets). The same early-stopping rule is
    applied to every column, so the null includes the tuning step."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits          # sklearn dependency
    classification = task == "classification"
    Est = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
    split = _blocked_inner_split(tr_index, gap) if GBM_EARLY_STOPPING else None
    out = np.empty((len(Zte), Ytr.shape[1]))
    # many small fits: OpenMP fork/join overhead (and spin-waiting on a busy
    # CPU) dominates, so the thread count is limited for the whole loop
    with threadpool_limits(limits=GBM_N_THREADS, user_api="openmp"):
        for j in range(Ytr.shape[1]):
            out[:, j] = _gbm_column(Est, Ztr, Ytr[:, j], Zte, split, classification, seed)
    return out


# --------------------------------------------------------------------------- #
# Bayesian population decoder (Poisson, vectorised over the null columns)     #
# --------------------------------------------------------------------------- #

def _bayes_fold(Xtr, Ytr, Xte, task, m, n_classes, n_bins, prior, pseudo, sigma):
    """Poisson Bayesian decoding of every label column (real + null shifts).

    Xtr [n_tr, C], Xte [n_te, C]  spike counts (frames x neurons)
    Ytr [n_tr, m'] label columns as built by the CV driver (circular: sin
                   block then cos block, m' = 2m; otherwise m' = m)
    Returns predictions in the same column layout as Ytr.

    Model: n_i(t) ~ Poisson(f_i(x_t)) independently across neurons, so
        log P(x = b | n) = log P(b) + sum_i n_i log f_i(b) - sum_i f_i(b) + const.
    Tuning curves f_i(b) (expected count per frame in behaviour bin b) are
    estimated on the training frames of *each* label column; because they are
    linear in the counts, all columns are obtained with one sparse product,
    and the log-likelihood of all columns with one dense product."""
    n_tr, C = Xtr.shape
    # 1) behaviour bin of every training frame, for every label column ----------
    if task == "classification":
        B = max(int(n_classes), 1)
        bins = Ytr.astype(np.int64)                                  # class index = bin
    elif task == "circular":
        B = int(n_bins)
        theta = np.arctan2(Ytr[:, :m], Ytr[:, m:])                   # recover the angle
        bins = np.floor((theta + np.pi) / (2 * np.pi) * B).astype(np.int64) % B
        centres = -np.pi + (np.arange(B) + 0.5) * 2 * np.pi / B      # equal-width bins
    else:
        B = int(n_bins)
        # quantile (equal-occupancy) edges per column, from that column's own
        # training labels: the null columns go through the identical procedure
        q = np.quantile(Ytr, np.linspace(0, 1, B + 1), axis=0)      # [B+1, m]
        bins = np.empty((n_tr, m), np.int64)
        for j in range(m):
            bins[:, j] = np.searchsorted(q[1:-1, j], Ytr[:, j], side="right")
    flat = bins + B * np.arange(m)                                   # [n_tr, m] -> row of m*B
    occ = np.bincount(flat.ravel(), minlength=m * B).reshape(m, B).astype(float)
    visited = occ > 0
    # 2) summed counts per (column, bin, neuron): sparse one-hot [m*B, n_tr] @ X
    onehot = sparse.csr_matrix((np.ones(flat.size), (flat.ravel(), np.repeat(np.arange(n_tr), m))),
                               shape=(m * B, n_tr))
    S = np.asarray(onehot @ Xtr).reshape(m, B, C)
    if task == "regression":                     # bin value = mean training label in the bin
        values = (np.bincount(flat.ravel(), weights=Ytr.ravel(), minlength=m * B).reshape(m, B)
                  / np.maximum(occ, 1.0))
    # 3) optional smoothing over neighbouring bins (numerator and occupancy) ----
    occ_s = occ
    if sigma and sigma > 0 and task != "classification":
        mode = "wrap" if task == "circular" else "nearest"
        S = gaussian_filter1d(S, sigma, axis=1, mode=mode)
        occ_s = gaussian_filter1d(occ, sigma, axis=1, mode=mode)
    # 4) tuning curves, shrunk toward each neuron's mean count by `pseudo` frames
    lam = Xtr.mean(0)                                                # [C]
    F = np.maximum((S + pseudo * lam) / (occ_s[..., None] + pseudo), BAYES_RATE_FLOOR)
    if prior == "empirical":                                         # training occupancy
        with np.errstate(divide="ignore"):
            logp = np.log(occ / occ.sum(1, keepdims=True))
    else:                                                            # flat over visited bins
        logp = np.where(visited, 0.0, -np.inf)
    logF_T = np.log(F).reshape(m * B, C).T                           # [C, m*B]
    const = (logp - F.sum(-1)).reshape(m * B)
    # 5) posterior over bins for the test frames, chunked to bound memory -----
    out = np.empty((len(Xte), 2 * m if task == "circular" else m))
    if task == "circular":
        s_c, c_c = np.sin(centres), np.cos(centres)
    step = max(1, int(BAYES_CHUNK // (m * B)))
    for a in range(0, len(Xte), step):
        L = (Xte[a:a + step] @ logF_T + const).reshape(-1, m, B)
        if task == "classification":                                 # MAP class
            out[a:a + step] = L.argmax(-1)
            continue
        L -= L.max(-1, keepdims=True)
        P = np.exp(L)
        P /= P.sum(-1, keepdims=True)
        if task == "circular":            # circular posterior mean = angle of E[sin], E[cos]
            out[a:a + step, :m] = P @ s_c
            out[a:a + step, m:] = P @ c_c
        else:                             # posterior mean of the bin values
            out[a:a + step] = np.einsum("tjb,jb->tj", P, values)
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


def _cv_decode(y, T, fold_predict, *, model, task, n_folds, gap, n_perm, smooth_sigma, seed,
               valid=None, min_gap=0, folds=None):
    """Shared engine of every decoder: labels, blocked folds, circular-shift
    null, pooled out-of-fold score, p-value and the output dict.

    fold_predict(tr, te, Yall, ctx) -> predictions [len(te), Yall.shape[1]]
    for every label column at once (column 0 = real labels, then the shifts;
    circular: sin block then cos block). ``ctx`` holds task, m, gap,
    n_classes and seed. ``valid`` marks frames usable as features; ``min_gap``
    raises the automatic purge gap (e.g. half a count-summation window)."""
    ok = np.isfinite(y.astype(float)) if y.dtype.kind in "fiub" else np.ones(T, bool)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    task = infer_task(y[ok]) if task == "auto" else task
    classes = None
    if task == "classification":
        classes, y_int = np.unique(y[ok], return_inverse=True)
        yy = np.full(T, -1, dtype=np.int64); yy[ok] = y_int
    else:
        yy = y.astype(float)
    if folds is None:
        g = auto_gap(yy[ok], task, smooth_sigma, n_folds) if gap == "auto" else int(gap)
        if gap == "auto" and min_gap > g:
            g = int(min(min_gap, max(T // (2 * n_folds) - 1, 0)))
        folds = blocked_folds(T, n_folds, g)
    else:
        folds = [(np.asarray(a, int), np.asarray(b, int)) for a, b in folds]
        g = _effective_gap(folds)

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
    ctx = dict(task=task, m=m, gap=g, seed=seed,
               n_classes=len(classes) if classes is not None else 0)

    pred = np.full((T, Yall.shape[1]), np.nan)
    fold_scores = []
    for tr, te in folds:
        tr = tr[ok[tr]]
        te_ok = te[ok[te]]
        if len(tr) < 5 or len(te_ok) < 2:
            continue
        pred[te_ok] = fold_predict(tr, te_ok, Yall, ctx)
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


def decode_target(Z, y, *, task="auto", model="linear", n_folds=N_FOLDS, gap="auto",
                  n_perm=N_PERM, smooth_sigma=0.0, seed=0):
    """Decode one behaviour series y [T] from features Z [T, D].

    model : 'linear' | 'knn' | 'logistic' | 'gbm' (see the module docstring).
            For 'gbm' the null is capped at GBM_MAX_PERM shifts.

    Returns a dict with the pooled out-of-fold score, per-fold scores, the
    null distribution, a permutation p-value and out-of-fold predictions."""
    if model not in DECODERS:
        raise ValueError(f"unknown model {model!r}; choose from {DECODERS}")
    Z = np.asarray(Z, np.float64)
    y = np.asarray(y)
    T = len(Z)
    if len(y) != T:
        raise ValueError(f"behaviour has {len(y)} samples but the embedding has {T} frames; "
                         "pass behavior_times to resample it")
    if model == "gbm":
        n_perm = _gbm_n_perm(n_perm)

    def fold_predict(tr, te, Yall, ctx):
        if model == "gbm":
            return _predict_gbm(Z[tr], Yall[tr], Z[te], tr, ctx["gap"], ctx["task"], ctx["seed"])
        if ctx["task"] == "classification":
            return _predict_classification(Z[tr], Yall[tr], Z[te], model)
        real_cols = (0, ctx["m"]) if ctx["task"] == "circular" else (0,)
        return _predict_regression(Z[tr], Yall[tr], Z[te], model, real_cols=real_cols)[0]

    return _cv_decode(y, T, fold_predict, model=model, task=task, n_folds=n_folds, gap=gap,
                      n_perm=n_perm, smooth_sigma=smooth_sigma, seed=seed)


def _counts_matrix(counts, T):
    """Validate spike counts and return them as [C, T] float64.

    A [T, C] array is transposed when the orientation is unambiguous."""
    N = np.asarray(counts, np.float64)
    if N.ndim == 1:
        N = N[None, :]
    if N.ndim != 2:
        raise ValueError(f"counts must be 2-D [C, T], got shape {N.shape}")
    if N.shape[1] != T and N.shape[0] == T:
        N = N.T
    if N.shape[1] != T:
        raise ValueError(f"counts have {N.shape[1]} frames but the behaviour / embedding has {T}")
    fin = np.isfinite(N)
    if np.any(N[fin] < 0):
        raise ValueError("counts must be non-negative raw spike counts (got negative values: "
                         "z-scored or PCA features cannot be decoded with a Poisson model)")
    return N


def bayes_decode_target(counts, y, *, task="auto", n_bins=BAYES_N_BINS, n_folds=N_FOLDS,
                        gap="auto", n_perm=N_PERM, smooth_sigma=0.0, seed=0, prior=BAYES_PRIOR,
                        pseudo_count=BAYES_PSEUDO_FRAMES, tuning_sigma=BAYES_TUNING_SIGMA,
                        window=1, folds=None):
    """Bayesian population decoding of y [T] from spike counts [C, T].

    Method (Zhang, Ginzburg, McNaughton & Sejnowski 1998, one-step Bayesian
    reconstruction): on each training fold, estimate every neuron's tuning
    curve as its mean count per behaviour bin; on the test fold compute the
    posterior over bins under independent Poisson neurons and read it out.

    counts        [C, T] raw, non-negative spike counts per frame ([T, C] is
                  accepted when unambiguous). Frames with non-finite counts
                  are excluded.
    task          'auto' | 'regression' | 'classification' | 'circular'
    n_bins        behaviour bins. circular: equal-width bins on [-pi, pi);
                  regression: equal-occupancy (quantile) bins of the training
                  labels, each represented by its mean training label;
                  classification: one bin per class (n_bins ignored).
    prior         'uniform': flat over the bins visited in the training fold
                  (with quantile bins this is close to the empirical prior);
                  'empirical': training-fold occupancy. Unvisited bins get
                  zero prior in both cases (no tuning information there).
    pseudo_count  tuning curves are shrunk toward each neuron's mean count by
                  this many pseudo-frames: f = (S + a * mean) / (N + a).
    tuning_sigma  Gaussian smoothing of the tuning curves over neighbouring
                  bins (in bins; circular wrap for angles; not for classes).
    window        counts are summed over a centred boxcar of this many frames
                  before decoding (a sum of Poisson counts is Poisson); 1 = off.
                  The automatic gap is raised to at least window // 2.
    folds         optional list of (train, test) frame indices; overrides
                  n_folds / gap (the reported gap is then the effective purge).
    The other arguments, the null and the output dict are as in decode_target.

    Read-out: regression = posterior mean of the bin values; circular =
    circular posterior mean; classification = MAP class. The null is fully
    vectorised: tuning curves of all shifted label columns come from one
    sparse product and their log-likelihoods from one dense product."""
    y = np.asarray(y)
    T = len(y)
    N = _counts_matrix(counts, T)
    if prior not in ("uniform", "empirical"):
        raise ValueError("prior must be 'uniform' or 'empirical'")
    valid = np.isfinite(N).all(0)
    N = np.where(np.isfinite(N), N, 0.0)
    window = max(1, int(window))
    if window > 1:                                       # centred boxcar sum
        N = uniform_filter1d(N, size=window, axis=1, mode="nearest") * window
    X = np.ascontiguousarray(N.T)                        # [T, C]

    def fold_predict(tr, te, Yall, ctx):
        return _bayes_fold(X[tr], Yall[tr], X[te], ctx["task"], ctx["m"], ctx["n_classes"],
                           n_bins, prior, float(pseudo_count), float(tuning_sigma or 0.0))

    out = _cv_decode(y, T, fold_predict, model="bayes", task=task, n_folds=n_folds, gap=gap,
                     n_perm=n_perm, smooth_sigma=smooth_sigma, seed=seed, valid=valid,
                     min_gap=window // 2, folds=folds)
    out.update(n_neurons=int(N.shape[0]), prior=prior, window=window,
               n_bins=(int(np.unique(out["y"]).size) if out["task"] == "classification"
                       else int(n_bins)))
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
                    baseline=None, seed=0, feature_name="embedding", counts=None,
                    bayes_kwargs=None):
    """Decode every behaviour variable from Z (and optionally baselines).

    behavior     : array [T], array [T, k] or dict {name: array [T]}
    tasks        : optional dict {name: 'regression'|'classification'|'circular'}
    circular     : names of angular variables (radians)
    baseline     : optional feature matrix [T, D] decoded identically (e.g. raw PCA)
    counts       : optional raw spike counts [C, T] aligned to the frames of Z;
                   adds rows with features='bayes_counts', model='bayes'
                   (Bayesian Poisson decoder, see bayes_decode_target)
    bayes_kwargs : optional overrides for bayes_decode_target (n_bins, prior,
                   pseudo_count, tuning_sigma, window, ...). By default the
                   count window matches the embedding smoothing
                   (matched_count_window(smooth_sigma)), just as the raw PCA
                   baseline is smoothed like the embedding.

    Holm correction is applied within each feature set; ``delta_vs_raw``
    (when a baseline is given) is score minus the raw_pca score of the target.
    """
    beh = _as_dict(behavior)
    tasks = dict(tasks or {})
    for name in circular or ():
        tasks[name] = "circular"
    # gbm rows use a capped null (warned once here, not again per target); the
    # Holm warning below uses the smallest null actually run
    n_perm_model = _gbm_n_perm(n_perm, stacklevel=2) if model == "gbm" else n_perm
    n_eff = n_perm_model
    if n_eff and len(beh) / (n_eff + 1) > 0.05:
        warnings.warn(f"n_perm={n_eff} is too small for {len(beh)} targets: the smallest "
                      f"Holm-corrected p-value is {len(beh) / (n_eff + 1):.3f} > 0.05. "
                      f"Use n_perm >= {int(np.ceil(20 * len(beh)))}.", stacklevel=2)
    feats = {feature_name: Z}
    if baseline is not None:
        feats["raw_pca"] = baseline
    bkw = {}
    if counts is not None:
        counts = _counts_matrix(counts, len(Z))
        feats["bayes_counts"] = counts
        bkw = dict(window=matched_count_window(smooth_sigma))
        bkw.update(bayes_kwargs or {})
    rows, preds, nulls = [], {}, {}
    for fname, F in feats.items():
        is_bayes = counts is not None and fname == "bayes_counts"
        for name, y in beh.items():
            if is_bayes:
                r = bayes_decode_target(F, y, task=tasks.get(name, "auto"), n_folds=n_folds,
                                        gap=gap, n_perm=n_perm, smooth_sigma=smooth_sigma,
                                        seed=seed, **bkw)
            else:
                r = decode_target(F, y, task=tasks.get(name, "auto"), model=model, n_folds=n_folds,
                                  gap=gap, n_perm=n_perm_model, smooth_sigma=smooth_sigma, seed=seed)
            null = r["null"]
            rows.append(dict(target=name, features=fname, task=r["task"], model=r["model"],
                             metric=r["metric"], score=r["score"],
                             score_sd=float(np.std(r["fold_scores"])) if r["fold_scores"] else np.nan,
                             fold_scores=json.dumps([round(s, 5) for s in r["fold_scores"]]),
                             chance=r["chance"],
                             null_mean=float(null.mean()) if null.size else np.nan,
                             null_p95=float(np.quantile(null, 0.95)) if null.size else np.nan,
                             p_value=r["p_value"], n_perm=int(null.size),
                             mae_deg=r.get("mae_deg", np.nan), pearson_r=r.get("pearson_r", np.nan),
                             n_frames=r["n_frames"], n_folds=r["n_folds"], gap=r["gap"],
                             n_features=int(F.shape[0]) if is_bayes else int(np.asarray(F).shape[1])))
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
