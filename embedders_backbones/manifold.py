"""Time-aware manifold projections and topology of NEMBA embeddings.

Contents
--------
``persistence``  persistent homology (ripser) of a subsample of the latent
                 cloud, with bars normalised by the cloud diameter (the same
                 normalisation as ``analysis.circularity``);
``tphate``       a native re-implementation of T-PHATE (Busch et al., Nat
                 Comput Sci 2023), i.e. PHATE (Moon et al., Nat Biotechnol
                 2019) whose diffusion operator is multiplied by a temporal
                 operator built from the data's autocorrelation function;
``project3d``    3-D projections (PCA / UMAP / T-PHATE) of a strided subset.

Everything works on a uniform *strided* subset of frames (the convention of
``analysis.project_subset``): consecutive returned indices are equally spaced
in time, which T-PHATE's temporal operator and the flow-field figure rely on.
"""

from __future__ import annotations

import importlib.util

import numpy as np
from scipy.linalg import toeplitz
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA

from .analysis import subsample_index

_HAS_RIPSER = importlib.util.find_spec("ripser") is not None
_HAS_UMAP = importlib.util.find_spec("umap") is not None

# --------------------------------------------------------------------------- #
# user-configurable defaults (edit here)                                       #
# --------------------------------------------------------------------------- #
TPHATE_DEFAULTS = dict(
    t_max=100,              # largest diffusion time scanned by the entropy knee
    eps=1e-7,               # floor inside the log potential, -log(P^t + eps)
    kernel_threshold=1e-4,  # affinities below this are set to 0 (as graphtools)
    n_pca=100,              # inputs with more dims are first reduced to this many PCs
    max_lag_frac=0.1,       # safety cap of the temporal window, as a fraction of n
    # temporal window rule: "zero" = first zero crossing of the mean
    # autocorrelation (default); "1/e" = first drop below 1/e (shorter window).
    # T-PHATE emphasises SLOW structure: on the NEMBA synthetic recording it
    # separates the behavioural states far better than PCA (2-D silhouette 0.49
    # vs 0.02) but blurs the faster ring phase (phase preservation 0.05 vs 0.91);
    # "1/e" only partly restores it (0.12). On a clean slow ring it preserves
    # the ring (0.995). Use PCA / PHATE for fast cycles, T-PHATE for slow states.
    lag_rule="zero",
    metric_mds_max_points=2000,  # SMACOF refinement only up to this many points
    mds_max_iter=60,        # SMACOF iterations (initialised with classical MDS; ~0.07 s
                            # each at 2000 points, the dominant cost of 'auto' below 2000)
)
PERSISTENCE_DEFAULTS = dict(
    n_perm_h2=250,          # greedy landmarks used when H2 is requested (cost ~ n^3)
)


# --------------------------------------------------------------------------- #
# persistent homology                                                          #
# --------------------------------------------------------------------------- #

def persistence(Z, max_points=600, maxdim=1, seed=0):
    """Persistent homology of the latent point cloud (Vietoris-Rips, ripser).

    A random subset of ``max_points`` frames is used (same draw as
    ``analysis.circularity`` for equal ``seed`` and ``max_points``). Each bar
    length (death - birth) is divided by the diameter of that subset, so the
    numbers are unitless and comparable across embeddings: a clean ring gives
    a single H1 bar of ~0.6-0.8, a torus two H1 bars and one H2 bar (Gardner
    et al., Nature 2022), an unstructured Gaussian cloud only short bars
    (< ~0.15).

    When ``maxdim >= 2`` ripser's greedy-permutation subsampling (``n_perm``,
    ``PERSISTENCE_DEFAULTS['n_perm_h2']`` landmarks) keeps the cost tractable;
    the diagrams are then an approximation bounded by the covering radius
    (returned as ``r_cover``).

    Returns a dict with
        available   False when ripser is not installed (other fields NaN/None)
        dgms        list of (n_bars, 2) birth/death arrays, one per dimension
        diameter    largest pairwise distance of the subset
        h1_norm     most persistent H1 bar / diameter (the circularity score)
        h1_second   second most persistent H1 bar / diameter (torus check)
        h2_norm     most persistent H2 bar / diameter (NaN when maxdim < 2)
        n_points, r_cover, maxdim
    """
    Z = np.asarray(Z, dtype=float)
    rng = np.random.default_rng(seed)
    # random (not strided) subset: topology does not need temporal order and a
    # random draw avoids aliasing with periodic behaviour
    sub = Z[rng.choice(len(Z), max_points, replace=False)] if len(Z) > max_points else Z
    diam = float(pdist(sub).max()) if len(sub) > 1 else 1.0
    out = dict(available=_HAS_RIPSER, dgms=None, diameter=diam, h1_norm=float("nan"),
               h1_second=float("nan"), h2_norm=float("nan"), n_points=int(len(sub)),
               r_cover=0.0, maxdim=int(maxdim))
    if not _HAS_RIPSER or len(sub) < 4:
        return out
    from ripser import ripser
    kw = {}
    if maxdim >= 2 and len(sub) > PERSISTENCE_DEFAULTS["n_perm_h2"]:
        kw["n_perm"] = PERSISTENCE_DEFAULTS["n_perm_h2"]
    res = ripser(sub, maxdim=int(maxdim), **kw)
    dgms = [np.asarray(d, float) for d in res["dgms"]]
    out["dgms"] = dgms
    out["r_cover"] = float(res.get("r_cover", 0.0))

    def _top(dim, rank):
        """rank-th largest finite bar length of dimension ``dim`` / diameter."""
        if dim >= len(dgms):
            return float("nan")
        d = dgms[dim]
        life = d[:, 1] - d[:, 0]
        life = np.sort(life[np.isfinite(life)])[::-1]
        return float(life[rank] / (diam + 1e-12)) if life.size > rank else 0.0

    out["h1_norm"] = _top(1, 0)
    out["h1_second"] = _top(1, 1)
    out["h2_norm"] = _top(2, 0) if maxdim >= 2 else float("nan")
    return out


# --------------------------------------------------------------------------- #
# T-PHATE building blocks                                                      #
# --------------------------------------------------------------------------- #

def mean_autocorrelation(X, max_lag):
    """Mean (over columns) autocorrelation function of X [n, d], lags 0..max_lag.

    Biased estimator via FFT (normalised so that acf[0] = 1); constant
    columns are ignored."""
    X = np.asarray(X, float)
    n = len(X)
    max_lag = int(min(max_lag, n - 1))
    Xc = X - X.mean(0)
    var = (Xc ** 2).sum(0)
    ok = var > 1e-12
    if not ok.any():
        return np.r_[1.0, np.zeros(max_lag)]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))         # zero padding: no circular wrap
    F = np.fft.rfft(Xc[:, ok], nfft, axis=0)
    ac = np.fft.irfft(F * np.conj(F), nfft, axis=0)[: max_lag + 1]
    return (ac / var[ok]).mean(1)


def temporal_lag(acf, rule="zero"):
    """First lag at which the autocorrelation function drops to <= 0
    (T-PHATE's temporal window). Returns len(acf) if it never does."""
    thr = 1.0 / np.e if rule == "1/e" else 0.0
    below = np.flatnonzero(np.asarray(acf)[1:] <= thr)
    return int(below[0] + 1) if below.size else int(len(acf))


def von_neumann_entropy(P, t_max=100):
    """Von Neumann entropy of the diffusion operator for t = 1..t_max.

    As in PHATE, the spectrum is taken as the singular values sigma_i of P;
    H(t) = -sum eta_i log eta_i with eta_i = sigma_i^t / sum_j sigma_j^t.
    Singular values come from ``eigvalsh(P P^T)`` (3-4x faster than an SVD;
    the precision loss only affects values ~1e-8 that carry no entropy)."""
    P = np.asarray(P, float)
    lam = np.linalg.eigvalsh(P @ P.T)
    sig = np.sqrt(np.clip(lam, 0.0, None))
    sig = sig[sig > 1e-12]
    H = np.empty(t_max)
    st = sig.copy()
    for t in range(t_max):
        p = st / st.sum()
        p = p[p > 0]
        H[t] = -np.sum(p * np.log(p))
        st = st * sig
    return H


def knee_point(y, x=None):
    """Knee of a decreasing curve: split point of the best two-segment
    piecewise-linear least-squares fit (the rule PHATE uses on the entropy)."""
    y = np.asarray(y, float)
    x = np.arange(1, len(y) + 1, dtype=float) if x is None else np.asarray(x, float)
    n = len(y)
    if n < 5:
        return x[int(np.argmin(y))] if n else 1.0

    def _sse(xs, ys):
        """Residual sum of squares of a straight-line fit."""
        A = np.c_[xs, np.ones_like(xs)]
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        return float(np.sum((ys - A @ coef) ** 2))

    errs = np.full(n, np.inf)
    for k in range(2, n - 2):                     # each segment keeps >= 3 points
        errs[k] = _sse(x[: k + 1], y[: k + 1]) + _sse(x[k:], y[k:])
    return x[int(np.argmin(errs))]


def _alpha_decay_kernel(X, knn, decay, threshold):
    """PHATE alpha-decay affinity: K_ij = 0.5 exp(-(d_ij/e_i)^a) + 0.5 exp(-(d_ij/e_j)^a),
    e_i = distance from i to its knn-th nearest neighbour (self excluded)."""
    D = cdist(X, X)
    k = int(min(knn, len(X) - 1))
    # partition is O(n^2) instead of a full sort; column 0 is the point itself
    eps = np.partition(D, k, axis=1)[:, k]
    eps = np.maximum(eps, 1e-12 + 1e-9 * (np.median(eps) if eps.size else 1.0))
    with np.errstate(over="ignore", under="ignore"):
        K = 0.5 * np.exp(-np.power(D / eps[:, None], decay)) + \
            0.5 * np.exp(-np.power(D / eps[None, :], decay))
    K[K < threshold] = 0.0
    return K


def _matrix_power_flushed(P, t, floor=1e-30):
    """P^t by repeated squaring in float32, flushing entries below ``floor``
    to 0 after every product.

    Without the flush, sparse operators (temporal=False, large t) can
    underflow into float32 denormals, which slow BLAS down by ~100x. The
    flush is harmless: the potential -log(P^t + 1e-7) cannot distinguish
    1e-30 from 0."""
    base = np.asarray(P, np.float32)
    out = None
    t = int(t)
    while t:
        if t & 1:
            out = base.copy() if out is None else out @ base
            out[out < floor] = 0.0
        t >>= 1
        if t:
            base = base @ base
            base[base < floor] = 0.0
    return out


def _row_normalise(A):
    """Row-stochastic version of a non-negative matrix."""
    s = A.sum(1, keepdims=True)
    return A / np.where(s > 0, s, 1.0)


def _classical_mds(D, n_components, seed=0):
    """Classical (Torgerson) MDS of a distance matrix; deterministic signs."""
    D2 = D ** 2
    rm = D2.mean(1, keepdims=True)
    B = -0.5 * (D2 - rm - rm.T + D2.mean())
    v0 = np.random.default_rng(seed).standard_normal(len(B))   # fixed ARPACK start
    lam, V = eigsh(B, k=n_components, which="LA", v0=v0)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    Y = V * np.sqrt(np.clip(lam, 0.0, None))
    # sign convention: the largest-magnitude coordinate of each axis is positive
    flip = np.sign(Y[np.argmax(np.abs(Y), axis=0), np.arange(Y.shape[1])])
    return Y * np.where(flip == 0, 1.0, flip)


# --------------------------------------------------------------------------- #
# T-PHATE                                                                      #
# --------------------------------------------------------------------------- #

def tphate(Z, n_components=2, knn=5, decay=40, t="auto", max_points=3000, seed=0,
           temporal=True, mds="auto", return_info=False):
    """T-PHATE-style projection of a strided subset of frames.

    NOTE: this is a native re-implementation written from the published
    descriptions of PHATE (Moon et al., Nat Biotechnol 2019) and T-PHATE
    (Busch et al., Nat Comput Sci 2023). It is NOT the reference ``phate`` /
    ``tphate`` package: it uses dense matrices (no landmarks), so results are
    close in spirit but not numerically identical, and the cost is O(n^3)
    (about 5 s for 3000 points on a laptop CPU; between 1000 and 2000 points
    the SMACOF refinement dominates, see ``TPHATE_DEFAULTS``).

    Steps
    -----
    1. strided subset ``idx`` (consecutive frames equally spaced in time);
       inputs with more than 100 dims are reduced to 100 PCs;
    2. alpha-decay kNN affinity (bandwidth = distance to the ``knn``-th
       neighbour, exponent ``decay``), row-normalised -> diffusion operator P_D;
    3. temporal operator P_T: affinity ACF(|i - j|) for lags shorter than the
       temporal window (``TPHATE_DEFAULTS['lag_rule']``: first zero crossing of
       the mean autocorrelation by default, or its first drop below 1/e; capped at
       ``TPHATE_DEFAULTS['max_lag_frac'] * n`` steps of the subset),
       row-normalised;
    4. P = P_D @ P_T (P_D alone when ``temporal=False``, i.e. plain PHATE);
    5. diffusion time t: knee of the von Neumann entropy H(t), t = 1..100,
       when ``t='auto'``;
    6. potential U = -log(P^t + eps); potential distances = Euclidean
       distances between rows of U;
    7. classical MDS, refined by metric MDS (sklearn SMACOF, initialised with
       the classical solution) when n <= 2000 (``mds='auto'``; force with
       ``mds='metric'`` or skip with ``mds='classic'``).

    Caveat: the temporal view smooths along time by construction, so
    T-PHATE trajectories look cleaner than PCA/UMAP ones partly *because* of
    that prior. Judge structure on held-out metrics, not on looks.

    Returns (P [n, n_components], idx [n]) or, with ``return_info=True``,
    (P, idx, info) where info has t, lag, acf, entropy, mds.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be a 2-D array [frames, dims]")
    cfg = TPHATE_DEFAULTS
    idx = subsample_index(len(Z), max_points)
    X = Z[idx]
    n = len(X)
    if n < max(10, knn + 2, n_components + 2):
        raise ValueError(f"tphate needs at least {max(10, knn + 2)} frames, got {n}")
    if X.shape[1] > cfg["n_pca"]:
        X = PCA(cfg["n_pca"], random_state=seed).fit_transform(X)

    # --- 2. data view ---------------------------------------------------------
    P_D = _row_normalise(_alpha_decay_kernel(X, knn, decay, cfg["kernel_threshold"]))

    # --- 3-4. temporal view and combined operator ---------------------------
    lag, acf = 0, None
    if temporal:
        cap = max(2, int(cfg["max_lag_frac"] * n))
        acf = mean_autocorrelation(X, cap)
        lag = min(temporal_lag(acf, rule=cfg.get("lag_rule", "zero")), cap)
        col = np.zeros(n)
        col[:lag] = acf[:lag]                  # positive part of the ACF only
        P_T = _row_normalise(toeplitz(col))    # symmetric Toeplitz band, then row-stochastic
        P = P_D @ P_T
    else:
        P = P_D

    # --- 5. diffusion time ----------------------------------------------------
    H = None
    if t == "auto":
        H = von_neumann_entropy(P, cfg["t_max"])
        t_use = int(knee_point(H))
    else:
        t_use = int(t)
    t_use = max(1, t_use)

    # --- 6. potential distances ----------------------------------------------
    # float32 powers: twice as fast, and values below eps are clipped anyway
    Pt = _matrix_power_flushed(P, t_use).astype(float)
    U = -np.log(np.clip(Pt, 0.0, None) + cfg["eps"])
    sq = (U ** 2).sum(1)
    D = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2.0 * (U @ U.T), 0.0, None))
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)                         # exact symmetry for MDS

    # --- 7. embedding ---------------------------------------------------------
    Y = _classical_mds(D, n_components, seed)
    used = "classic"
    if mds == "metric" or (mds == "auto" and n <= cfg["metric_mds_max_points"]):
        from sklearn.manifold import MDS
        mm = MDS(n_components=n_components, metric=True, n_init=1,
                 max_iter=cfg["mds_max_iter"], dissimilarity="precomputed",
                 random_state=seed, normalized_stress="auto")
        Y = mm.fit_transform(D, init=Y)
        used = "metric"
    Y = np.asarray(Y, float)
    if return_info:
        return Y, idx, dict(t=t_use, lag=int(lag), acf=acf, entropy=H, mds=used,
                            temporal=bool(temporal), n_points=int(n))
    return Y, idx


# --------------------------------------------------------------------------- #
# 3-D projections                                                              #
# --------------------------------------------------------------------------- #

def project3d(Z, method="pca", max_points=4000, seed=0):
    """3-D projection of a strided subset of frames. Returns (P3 [n, 3], idx).

    method  'pca'     linear, all axes interpretable as variance directions;
            'umap'    UMAP (n_neighbors=30, min_dist=0.1, 200 epochs, as
                      ``analysis.project``); falls back to PCA when umap-learn
                      is missing;
            'tphate'  ``tphate(..., n_components=3)``. T-PHATE is O(n^3), so
                      its subset is capped at 3000 frames whatever
                      ``max_points`` says (check ``len(idx)``)."""
    Z = np.asarray(Z, dtype=float)
    m = method.lower().replace("-", "")
    if m == "tphate":
        return tphate(Z, n_components=3, max_points=min(int(max_points), 3000), seed=seed)
    idx = subsample_index(len(Z), max_points)
    X = Z[idx]
    if m == "pca" or (m == "umap" and not _HAS_UMAP):
        return PCA(3, random_state=seed).fit_transform(X), idx
    if m == "umap":
        import umap
        red = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, n_epochs=200,
                        random_state=seed, n_jobs=1)
        return np.asarray(red.fit_transform(X), float), idx
    raise ValueError(f"unknown 3-D projection {method!r}; choose pca, umap or tphate")
