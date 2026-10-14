"""Consistency between embeddings and alignment across sessions.

Two different questions live here:

1. *Consistency* (same recording, several embeddings): can one embedding be
   linearly mapped onto another? Cross-validated R^2 of a ridge map, the
   measure CEBRA uses to compare models and seeds (Schneider, Lee & Mathis,
   Nature 2023). Directional: R^2(a -> b) high and R^2(b -> a) low means b's
   information is contained in a but not the reverse.

2. *Alignment* (different sessions or animals, different neurons): put a new
   session's latent trajectory into the coordinates of a reference session.
   Time-matched data: ``procrustes_align`` (rotation + scale) or
   ``cca_align`` (Gallego et al., Nat Neurosci 2020: CCA on the latent
   dynamics). Not time-matched (the usual case): ``align_sessions`` matches
   the two sessions through behaviour, using one centroid ("anchor") per
   behaviour bin, and reports a leave-one-anchor-out R^2.

Every transform is an affine map stored as a dict with ``A`` (d_src x d_ref)
and ``b`` (d_ref,): ``apply_transform(Z, T) = Z @ A + b``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# --------------------------------------------------------------------------- #
# user-configurable defaults (edit here)                                       #
# --------------------------------------------------------------------------- #
ALIGN_DEFAULTS = dict(
    ridge_alpha=1e-2,     # ridge penalty of the consistency map (inputs z-scored)
    min_count=5,          # frames needed in a behaviour bin to form an anchor
    rank_tol=1e-8,        # relative singular-value threshold for whitening
    cov_ridge=1e-6,       # ridge added to covariances (fraction of mean variance)
)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def apply_transform(Z, T):
    """Apply an alignment transform (dict with 'A' and 'b') to frames Z [n, d_src]."""
    return np.asarray(Z, float) @ T["A"] + T["b"]


def _pooled_r2(Y, Y_hat):
    """R^2 pooled over all output dims (variance-weighted): 1 - SSE / SST."""
    Y = np.asarray(Y, float)
    sst = float(((Y - Y.mean(0)) ** 2).sum())
    sse = float(((Y - Y_hat) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def _contiguous_folds(n, n_folds):
    """Test-index blocks of contiguous frames (no shuffling: respects time)."""
    edges = np.linspace(0, n, int(n_folds) + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(int(n_folds))]


def _cv_map_r2(X, Y, n_folds, alpha, multioutput):
    """Cross-validated R^2 of a ridge map X -> Y over contiguous blocks.

    The inputs are z-scored with training-fold statistics only; R^2 is
    computed on each held-out block (against that block's own mean, so slow
    drifts count as errors) and averaged over folds."""
    n = len(X)
    scores = []
    for te in _contiguous_folds(n, n_folds):
        tr = np.ones(n, bool)
        tr[te] = False
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        model = Ridge(alpha=alpha).fit((X[tr] - mu) / sd, Y[tr])
        pred = model.predict((X[te] - mu) / sd)
        scores.append(r2_score(Y[te], pred, multioutput=multioutput))
    return float(np.mean(scores))


def _is_discrete(b):
    """True when a behaviour vector looks categorical (labels, bools, few
    integer levels), False for a continuous variable."""
    b = np.asarray(b)
    if b.dtype.kind in "OUSb":
        return True
    if b.dtype.kind in "iu":
        return np.unique(b).size <= 50
    fin = b[np.isfinite(b)]
    return bool(fin.size and np.unique(fin).size <= 20 and np.allclose(fin, np.round(fin)))


def _whitener(C, rank_tol):
    """Symmetric inverse square root and square root of a covariance matrix
    (eigen-directions below ``rank_tol`` of the largest eigenvalue dropped)."""
    lam, V = np.linalg.eigh(C)
    keep = lam > rank_tol * max(lam.max(), 1e-300)
    lam, V = lam[keep], V[:, keep]
    W = V / np.sqrt(lam)                 # d x r : whitening (Z @ W has identity covariance)
    Wi = (V * np.sqrt(lam)).T            # r x d : un-whitening
    return W, Wi


# --------------------------------------------------------------------------- #
# consistency between embeddings of the same recording                        #
# --------------------------------------------------------------------------- #

def linear_consistency(Za, Zb, n_folds=2, alpha=None, multioutput="uniform_average"):
    """Cross-validated linear consistency between two time-matched embeddings.

    A ridge map (penalty ``alpha``, default ``ALIGN_DEFAULTS['ridge_alpha']``
    on z-scored inputs, i.e. nearly ordinary least squares) is fitted on
    ``n_folds - 1`` contiguous blocks and scored on the held-out block.
    R^2 is averaged over output dims (``multioutput='uniform_average'``, as
    CEBRA; pass 'variance_weighted' to weight dims by their variance).

    Returns dict(a_to_b, b_to_a, symmetric) where symmetric is the mean of
    the two directions. ~1: the embeddings are linear transforms of each
    other; ~0 (or negative): no linear relation that generalises in time."""
    Za, Zb = np.asarray(Za, float), np.asarray(Zb, float)
    if len(Za) != len(Zb):
        raise ValueError(f"embeddings must be time-matched: {len(Za)} vs {len(Zb)} frames")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2 (held-out blocks are required)")
    alpha = ALIGN_DEFAULTS["ridge_alpha"] if alpha is None else alpha
    ab = _cv_map_r2(Za, Zb, n_folds, alpha, multioutput)
    ba = _cv_map_r2(Zb, Za, n_folds, alpha, multioutput)
    return dict(a_to_b=ab, b_to_a=ba, symmetric=0.5 * (ab + ba))


def consistency_matrix(embeddings, n_folds=2, symmetric=False, alpha=None):
    """Pairwise linear consistency of several time-matched embeddings.

    embeddings  dict {name: Z [n_frames, d]} (all with the same n_frames)
    Returns a square DataFrame; entry [row a, column b] = R^2 of predicting b
    from a (directional), or the symmetric mean when ``symmetric=True``.
    The diagonal is NaN (self-consistency is trivially 1)."""
    names = list(embeddings)
    M = np.full((len(names), len(names)), np.nan)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = linear_consistency(embeddings[names[i]], embeddings[names[j]], n_folds, alpha)
            if symmetric:
                M[i, j] = M[j, i] = r["symmetric"]
            else:
                M[i, j], M[j, i] = r["a_to_b"], r["b_to_a"]
    df = pd.DataFrame(M, index=names, columns=names)
    df.index.name, df.columns.name = "source", "target"
    return df


# --------------------------------------------------------------------------- #
# time-matched alignment                                                       #
# --------------------------------------------------------------------------- #

def _procrustes_fit(A, B, scaling=True):
    """Orthogonal Procrustes (optionally with isotropic scale) mapping rows of
    A [k, d_a] onto rows of B [k, d_b]; dims may differ (semi-orthogonal R)."""
    mu_a, mu_b = A.mean(0), B.mean(0)
    Ac, Bc = A - mu_a, B - mu_b
    # thin SVD of the cross-product: R = U V^T maximises tr(R^T Ac^T Bc)
    U, S, Vt = np.linalg.svd(Ac.T @ Bc, full_matrices=False)
    R = U @ Vt
    ss = float((Ac ** 2).sum())
    scale = float(S.sum() / ss) if (scaling and ss > 0) else 1.0
    A_map = scale * R
    return dict(method="procrustes", A=A_map, b=mu_b - mu_a @ A_map, R=R, scale=scale,
                mu_src=mu_a, mu_ref=mu_b)


def procrustes_align(Z_src, Z_ref, scaling=True):
    """Align time-matched Z_src onto Z_ref with a rotation / reflection
    (+ isotropic scale and translation).

    Distances inside Z_src are preserved (up to the global scale), so a high
    R^2 means the two trajectories have the same *shape*. Embedding dims may
    differ. Returns (aligned [n, d_ref], transform dict, R^2 pooled over dims)."""
    A, B = np.asarray(Z_src, float), np.asarray(Z_ref, float)
    if len(A) != len(B):
        raise ValueError(f"procrustes_align needs time-matched data: {len(A)} vs {len(B)} frames")
    T = _procrustes_fit(A, B, scaling)
    aligned = apply_transform(A, T)
    return aligned, T, _pooled_r2(B, aligned)


def cca_align(Z_src, Z_ref, n_components=None):
    """Canonical correlation alignment of time-matched latents (Gallego et
    al., Nat Neurosci 2020 used it on per-session PCA latents).

    Each session is whitened (rank-deficient directions dropped), the SVD of
    the whitened cross-covariance gives paired directions maximising the
    correlation. Returns
        C_src, C_ref  canonical variables [n, n_components] (unit variance)
        corr          canonical correlations (descending)
        transform     affine map of Z_src into Z_ref coordinates through the
                      shared canonical space (keys A, b, W_src, W_ref, corr)
    Caveat: CCA maximises correlation, not reconstruction; with many dims and
    few frames the leading correlations are inflated (fit and evaluate on
    different trials or blocks when it matters)."""
    A, B = np.asarray(Z_src, float), np.asarray(Z_ref, float)
    if len(A) != len(B):
        raise ValueError(f"cca_align needs time-matched data: {len(A)} vs {len(B)} frames")
    n = len(A)
    mu_a, mu_b = A.mean(0), B.mean(0)
    Ac, Bc = A - mu_a, B - mu_b
    tol = ALIGN_DEFAULTS["rank_tol"]

    def _orth(X):
        """Orthonormal basis of X's column space and the weights producing it."""
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        keep = S > tol * max(S.max(), 1e-300)
        return U[:, keep], Vt[keep].T / S[keep]        # X @ W = U

    Qa, Wa0 = _orth(Ac)
    Qb, Wb0 = _orth(Bc)
    U, S, Vt = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    k = int(min(n_components or len(S), len(S)))
    scale = np.sqrt(max(n - 1, 1))                      # unit-variance canonical variables
    Wa = Wa0 @ U[:, :k] * scale
    Wb = Wb0 @ Vt.T[:, :k] * scale
    C_src, C_ref = Ac @ Wa, Bc @ Wb
    corr = np.clip(S[:k], 0.0, 1.0)
    A_map = Wa @ np.linalg.pinv(Wb)                     # canonical space -> ref coordinates
    T = dict(method="cca", A=A_map, b=mu_b - mu_a @ A_map, W_src=Wa, W_ref=Wb, corr=corr,
             mu_src=mu_a, mu_ref=mu_b)
    return C_src, C_ref, corr, T


# --------------------------------------------------------------------------- #
# behaviour-anchored alignment of sessions that are not time-matched          #
# --------------------------------------------------------------------------- #

def behavior_anchors(Z, behavior, n_bins=12, circular=False, edges=None, levels=None,
                     min_count=None):
    """Mean latent state per behaviour bin ("anchor").

    Continuous behaviour: ``n_bins`` bins, equal width on [-pi, pi) when
    ``circular`` (angles are wrapped first), equal occupancy (quantiles)
    otherwise; pass ``edges`` to impose shared bins across sessions.
    Discrete behaviour (labels, bools, few integer levels): one anchor per
    label; pass ``levels`` to impose the label set and order.

    Returns (C [k, d] centroids, NaN rows for bins with < ``min_count``
    frames; keys [k] bin centres or labels; counts [k])."""
    Z = np.asarray(Z, float)
    b = np.asarray(behavior)
    if len(b) != len(Z):
        raise ValueError(f"behaviour ({len(b)}) and embedding ({len(Z)}) lengths differ")
    min_count = ALIGN_DEFAULTS["min_count"] if min_count is None else min_count
    if levels is not None or (edges is None and _is_discrete(b)):
        keys = np.unique(b) if levels is None else np.asarray(levels)
        lab = _labels(b, None, keys, circular)
    else:
        if edges is None:
            bf = b.astype(float)
            bf = bf[np.isfinite(bf)]
            edges = (np.linspace(-np.pi, np.pi, n_bins + 1) if circular else
                     np.unique(np.quantile(bf, np.linspace(0, 1, n_bins + 1))))
        edges = np.asarray(edges, float)
        lab = _labels(b, edges, None, circular)
        keys = 0.5 * (edges[:-1] + edges[1:])
    C = np.full((len(keys), Z.shape[1]), np.nan)
    counts = np.bincount(lab[lab >= 0], minlength=len(keys))[: len(keys)]
    for i in range(len(keys)):
        if counts[i] >= min_count:
            C[i] = Z[lab == i].mean(0)
    return C, keys, counts


def _bin_stats(Z, lab, keep_bins):
    """Centroid and within-bin covariance of each kept bin (for whitening)."""
    d = Z.shape[1]
    mus, covs = np.zeros((len(keep_bins), d)), np.zeros((len(keep_bins), d, d))
    for i, k in enumerate(keep_bins):
        X = Z[lab == k]
        mus[i] = X.mean(0)
        covs[i] = np.cov(X.T, bias=True).reshape(d, d)
    return mus, covs


def _labels(behavior, edges, levels, circular):
    """Per-frame bin index under given bins (-1 = unassigned).

    ``levels`` (discrete labels) takes precedence over ``edges``; angles are
    wrapped to (-pi, pi] when ``circular``; values outside [edges[0],
    edges[-1]] are left unassigned instead of being forced into edge bins."""
    b = np.asarray(behavior)
    if levels is not None:
        lab = np.full(len(b), -1)
        for i, lv in enumerate(levels):
            lab[b == lv] = i
        return lab
    bf = b.astype(float)
    valid = np.isfinite(bf)
    if circular:
        bf = np.angle(np.exp(1j * np.where(valid, bf, 0.0)))
    lab = np.clip(np.searchsorted(edges, bf, side="right") - 1, 0, len(edges) - 2)
    lab[~valid | (bf < edges[0]) | (bf > edges[-1])] = -1
    return lab


def _fit_anchor_map(method, An, Ar, cov_n, cov_r, n_components):
    """Fit the new -> reference map on matched anchors An [k, d_n], Ar [k, d_r].

    procrustes: similarity transform of the anchors (few parameters, robust
    with few anchors, assumes the same geometry up to rotation and scale).
    cca: whiten each session with its bin-balanced covariance (mean within-bin
    covariance + covariance of the anchors), then the SVD of the whitened
    anchor cross-covariance gives the canonical pairs; the map is
    whiten(new) -> rotate onto the top canonical pairs -> un-whiten(ref).
    With all components kept, this is orthogonal Procrustes between the two
    whitened anchor sets, so it also absorbs anisotropic linear distortions."""
    if method == "procrustes":
        return _procrustes_fit(An, Ar, scaling=True)
    if method != "cca":
        raise ValueError(f"unknown method {method!r}; choose 'procrustes' or 'cca'")
    k = len(An)
    mu_n, mu_r = An.mean(0), Ar.mean(0)
    Anc, Arc = An - mu_n, Ar - mu_r

    def _balanced(Ac, covs):
        """Bin-balanced covariance (robust to different behaviour occupancy)."""
        C = covs.mean(0) + Ac.T @ Ac / len(Ac)
        return C + ALIGN_DEFAULTS["cov_ridge"] * np.trace(C) / len(C) * np.eye(len(C))

    Wn, _ = _whitener(_balanced(Anc, cov_n), ALIGN_DEFAULTS["rank_tol"])
    Wr, Wr_inv = _whitener(_balanced(Arc, cov_r), ALIGN_DEFAULTS["rank_tol"])
    U, S, Vt = np.linalg.svd((Anc @ Wn).T @ (Arc @ Wr) / k, full_matrices=False)
    m = int(min(n_components or (k - 1), len(S)))
    A_map = Wn @ U[:, :m] @ Vt[:m] @ Wr_inv
    return dict(method="cca", A=A_map, b=mu_r - mu_n @ A_map, corr=S[:m], n_components=m,
                mu_src=mu_n, mu_ref=mu_r)


def align_sessions(Z_ref, beh_ref, Z_new, beh_new, method="procrustes", n_bins=12,
                   circular=False, min_count=None, n_components=None, n_null=0, seed=0):
    """Align a new session onto a reference session through behaviour.

    The sessions may have different neurons, embedding dims and lengths. Both
    are binned on *shared* behaviour bins (equal width on the circle when
    ``circular``, pooled quantiles for a linear variable, common labels for a
    discrete one); bins with >= ``min_count`` frames in both sessions become
    matched anchors, and the map new -> reference is fitted on them
    (``method='procrustes'``: rotation + scale; ``'cca'``: canonical
    directions, see ``_fit_anchor_map``) then applied to every frame of Z_new.

    Quality = leave-one-anchor-out R^2: each anchor is predicted by a map
    fitted on the other anchors (whitening included), pooled over anchors and
    dims. Caveat: on a symmetric manifold (a ring) a *wrong* correspondence
    can also align well, because a rotation absorbs it; ``n_null > 0``
    re-fits with randomly permuted anchor labels and stores the null R^2
    values in ``transform['null_r2']`` as a reference.

    Returns (Z_new aligned into reference coordinates [n_new, d_ref],
    transform dict (A, b, anchors, keys, r2_insample, ...), held-out R^2)."""
    Zr, Zn = np.asarray(Z_ref, float), np.asarray(Z_new, float)
    br, bn = np.asarray(beh_ref), np.asarray(beh_new)
    min_count = ALIGN_DEFAULTS["min_count"] if min_count is None else min_count

    # --- shared binning ------------------------------------------------------
    edges = levels = None
    if _is_discrete(br) and _is_discrete(bn):
        levels = np.intersect1d(np.unique(br), np.unique(bn))
    elif circular:
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    else:
        pooled = np.r_[br.astype(float), bn.astype(float)]
        pooled = pooled[np.isfinite(pooled)]
        edges = np.unique(np.quantile(pooled, np.linspace(0, 1, n_bins + 1)))
    lab_r = _labels(br, edges, levels, circular)
    lab_n = _labels(bn, edges, levels, circular)
    n_keys = len(levels) if levels is not None else len(edges) - 1
    cnt_r = np.bincount(lab_r[lab_r >= 0], minlength=n_keys)[:n_keys]
    cnt_n = np.bincount(lab_n[lab_n >= 0], minlength=n_keys)[:n_keys]
    keep = np.flatnonzero((cnt_r >= min_count) & (cnt_n >= min_count))
    if len(keep) < 3:
        raise ValueError(f"only {len(keep)} behaviour bins are populated in both sessions "
                         f"(>= {min_count} frames); need >= 3 anchors")
    Ar, cov_r = _bin_stats(Zr, lab_r, keep)
    An, cov_n = _bin_stats(Zn, lab_n, keep)
    keys = (levels[keep] if levels is not None else 0.5 * (edges[:-1] + edges[1:])[keep])

    # --- leave-one-anchor-out quality ---------------------------------------
    def _heldout_r2(An_, Ar_, cn_, cr_):
        """Pooled R^2 of anchors predicted by maps fitted without them."""
        pred = np.zeros_like(Ar_)
        for j in range(len(Ar_)):
            o = np.arange(len(Ar_)) != j
            T_ = _fit_anchor_map(method, An_[o], Ar_[o], cn_[o], cr_[o], n_components)
            pred[j] = apply_transform(An_[j:j + 1], T_)[0]
        return _pooled_r2(Ar_, pred), pred

    r2_ho, pred = _heldout_r2(An, Ar, cov_n, cov_r)
    T = _fit_anchor_map(method, An, Ar, cov_n, cov_r, n_components)
    T.update(anchors_ref=Ar, anchors_new=An, keys=keys, counts_ref=cnt_r[keep],
             counts_new=cnt_n[keep], anchors_heldout_pred=pred, r2_heldout=r2_ho,
             r2_insample=_pooled_r2(Ar, apply_transform(An, T)), circular=bool(circular))
    if n_null > 0:
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(int(n_null)):
            p = rng.permutation(len(An))
            null.append(_heldout_r2(An[p], Ar, cov_n[p], cov_r)[0])
        T["null_r2"] = np.asarray(null)
    return apply_transform(Zn, T), T, r2_ho
