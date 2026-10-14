"""Clustering (KMeans / DEC / CRF), low-dimensional projections
(PCA / t-SNE / UMAP) and manifold metrics.

Training and inference live in ``training``; ``train`` / ``embed_sequence`` /
``make_windows`` / ``resolve_device`` are re-exported here for compatibility.

Optional deps are imported lazily (importing umap pulls in numba and, when
installed, TensorFlow: ~13 s that v0.1 paid at every ``import``):
umap-learn (UMAP) and ripser (topological circularity) fall back to simpler
equivalents if missing.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import torch
import torch.nn as nn

from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from . import backbones as BK
from .backbones import DECAutoencoder, target_distribution, LinearChainCRF
from .training import train, embed_sequence, make_windows, resolve_device  # noqa: F401  (re-export)

_HAS_UMAP = importlib.util.find_spec("umap") is not None
_HAS_RIPSER = importlib.util.find_spec("ripser") is not None

# Default number of frames shown per projection method. Uniform *strided*
# subsampling keeps trajectories continuous while bounding CPU time
# (t-SNE is O(n log n) with a large constant; UMAP has a numba JIT warm-up).
MAX_POINTS = {"pca": 20000, "umap": 3000, "tsne": 2500, "tphate": 3000}


def smooth_standardize(Z, sigma=4.0, return_stats=False):
    """Gaussian temporal smoothing (sigma in frames, 0 = off) then z-scoring.

    Smoothing is a strong prior: it raises state stability and decoding scores
    of slow variables but blurs fast transients. Keep it explicit."""
    Zs = gaussian_filter1d(Z, sigma=sigma, axis=0) if sigma and sigma > 0 else np.asarray(Z)
    mu, sd = Zs.mean(0), Zs.std(0) + 1e-8
    out = ((Zs - mu) / sd).astype(np.float32)
    return (out, mu.astype(np.float32), sd.astype(np.float32)) if return_stats else out


# --------------------------------------------------------------------------- #
# clustering (kmeans / DEC / CRF, dispatched by backbone)                     #
# --------------------------------------------------------------------------- #

def auto_k(Z, k_min=2, k_max=10, seed=0, sample=3000):
    """Pick the number of states by maximising the KMeans silhouette score over
    K in [k_min, k_max]. Returns (best_k, {k: silhouette}).

    Caveat: silhouette favours compact, convex, well separated clusters and
    small K. On a continuous manifold (e.g. a ring) any K is a discretisation,
    so inspect the whole score curve (saved in the metrics) before trusting it."""
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(Z), sample, replace=False) if len(Z) > sample
           else np.arange(len(Z)))
    Zs = Z[idx]
    scores, best_k, best_s = {}, k_min, -1.0
    for k in range(k_min, min(k_max, len(Zs) - 1) + 1):
        try:
            lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(Zs)
            s = float(silhouette_score(Zs, lab))
        except Exception:  # noqa: BLE001
            s = -1.0
        scores[k] = s
        if s > best_s:
            best_s, best_k = s, k
    return best_k, scores


def cluster_structure(silhouette):
    """Verbal reading of an average silhouette (Kaufman & Rousseeuw, 1990,
    Finding Groups in Data): <=0.25 none, <=0.50 weak, <=0.70 reasonable,
    >0.70 strong."""
    s = float(silhouette)
    if not np.isfinite(s):
        return "undefined"
    return "none" if s <= 0.25 else "weak" if s <= 0.50 else "reasonable" if s <= 0.70 else "strong"


def cluster_silhouette(Z, labels, seed=0, sample=3000):
    """Silhouette of a state labelling on the embedding (cluster separation)."""
    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(Z), sample, replace=False) if len(Z) > sample else np.arange(len(Z)))
    try:
        return float(silhouette_score(Z[idx], labels[idx]))
    except Exception:  # noqa: BLE001
        return float("nan")


def cluster(name, Z, k=8, seed=0):
    """State sequence from the embedding, with the backbone's clustering head."""
    kind = BK.clustering_kind(name)
    if kind == "dec":
        return _deep_embedded_clustering(Z, k, seed)
    if kind == "crf":
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
        return _crf_states(Z, km, k, seed)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z).astype(int)


def _deep_embedded_clustering(H, k, seed):
    torch.manual_seed(seed)
    Ht = torch.from_numpy(StandardScaler().fit_transform(H).astype(np.float32))
    dec = DECAutoencoder(in_dim=Ht.shape[1], bottleneck=min(16, Ht.shape[1]), hidden=64, n_clusters=k)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(120):
        z = dec.encoder(Ht); loss = nn.functional.mse_loss(dec.decoder(z), Ht)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z0 = dec.encoder(Ht).numpy()
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z0)
    dec.cluster.centroids.data = torch.from_numpy(km.cluster_centers_.astype(np.float32))
    opt = torch.optim.AdamW(dec.parameters(), lr=5e-4, weight_decay=1e-4); p = None
    for it in range(160):
        z, rec, q = dec(Ht)
        if it % 10 == 0:
            p = target_distribution(q).detach()
        kl = nn.functional.kl_div(q.clamp_min(1e-12).log(), p, reduction="batchmean")
        loss = nn.functional.mse_loss(rec, Ht) + 0.5 * kl
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return dec(Ht)[2].argmax(1).numpy().astype(int)


def _crf_states(Z, km_labels, k, seed, chunk=256, iters=80):
    """Linear-chain CRF fitted to KMeans pseudo-labels, Viterbi-decoded.

    v0.2: trained on a batch of ``chunk``-frame subsequences (one vectorised
    forward recursion of length ``chunk``) instead of one Python loop over the
    whole recording per iteration; Viterbi runs in NumPy."""
    torch.manual_seed(seed)
    Zs = StandardScaler().fit_transform(Z).astype(np.float32)
    T = len(Zs)
    L = int(min(chunk, T)); n = max(1, T // L)
    crf = LinearChainCRF(emb_dim=Zs.shape[1], num_tags=k)
    opt = torch.optim.AdamW(crf.parameters(), lr=5e-3, weight_decay=1e-4)
    zt = torch.from_numpy(Zs[: n * L].reshape(n, L, -1))
    tags = torch.from_numpy(km_labels[: n * L].astype(np.int64).reshape(n, L))
    for _ in range(iters):
        loss = crf.nll_batch(zt, tags)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(crf.parameters(), 5.0); opt.step()
    return crf.viterbi_np(Zs).astype(int)


# --------------------------------------------------------------------------- #
# projections + metrics                                                       #
# --------------------------------------------------------------------------- #

def subsample_index(n, max_points):
    """Uniform strided frame indices (keeps temporal continuity)."""
    if max_points is None or n <= max_points:
        return np.arange(n)
    stride = int(np.ceil(n / max_points))
    return np.arange(0, n, stride)


def project(Z, method="umap", seed=0):
    """2-D projection of *all* rows of Z (see ``project_subset`` for speed)."""
    method = method.lower()
    if method == "pca":
        return PCA(2, random_state=seed).fit_transform(Z)
    if method in ("tsne", "t-sne"):
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=min(30, max(5, len(Z) // 10)),
                    init="pca", random_state=seed).fit_transform(Z)
    if method == "umap":
        if not _HAS_UMAP:
            return PCA(2, random_state=seed).fit_transform(Z)
        import umap
        # a fixed random_state makes umap single-threaded (reproducible); 200
        # epochs instead of 500 keep it to a few seconds for a few thousand points
        return umap.UMAP(n_neighbors=30, min_dist=0.1, n_epochs=200, random_state=seed,
                         n_jobs=1).fit_transform(Z)
    if method in ("tphate", "t-phate"):
        from .manifold import tphate
        return tphate(Z, max_points=len(Z), seed=seed)[0]
    raise ValueError(f"unknown projection {method!r}")


def project_subset(Z, method="umap", seed=0, max_points=None):
    """Project a strided subset of frames. Returns (P [n, 2], idx [n]).

    'tphate' is the time-aware T-PHATE-style projection (``manifold.tphate``,
    a native re-implementation of Busch et al. 2023)."""
    m = method.lower().replace("t-sne", "tsne").replace("t-phate", "tphate")
    mp = MAX_POINTS.get(m, 5000) if max_points is None else max_points
    if m == "tphate":
        from .manifold import tphate
        return tphate(Z, max_points=mp, seed=seed)
    idx = subsample_index(len(Z), mp)
    return project(Z[idx], m, seed), idx


def circularity(Z, seed=0):
    """Ring-likeness of the embedding: the most persistent H1 bar (ripser)
    divided by the point-cloud diameter. Fallback without ripser: radial
    regularity of the 2-D PCA projection."""
    rng = np.random.default_rng(seed)
    sub = Z[rng.choice(len(Z), min(700, len(Z)), replace=False)] if len(Z) > 700 else Z
    diam = float(pdist(sub).max()) if len(sub) > 1 else 1.0
    if _HAS_RIPSER:
        from ripser import ripser
        dg = ripser(sub, maxdim=1)["dgms"][1]
        return float((dg[:, 1] - dg[:, 0]).max() / (diam + 1e-9)) if len(dg) else 0.0
    P = PCA(2).fit_transform(Z); r = np.linalg.norm(P - P.mean(0), axis=1)
    return float(1.0 - r.std() / (r.mean() + 1e-9))


def metrics(Z, seed=0):
    """Geometry of the embedding cloud (unitless, comparable across backbones)."""
    Zc = Z - Z.mean(0); S = np.linalg.svd(Zc, full_matrices=False)[1]; lam = S ** 2
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(1500, len(Z)), replace=False) if len(Z) > 1500 else np.arange(len(Z))
    v = np.diff(Z, axis=0); speed = np.linalg.norm(v, axis=1)
    return dict(circularity=circularity(Z, seed),
                diameter=float(pdist(Z[idx]).max()),
                participation_ratio=float(lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-12)),
                planarity=float(lam[:2].sum() / (lam.sum() + 1e-12)),
                mean_speed=float(speed.mean()),
                speed_cv=float(speed.std() / (speed.mean() + 1e-9)))


def state_statistics(states, n_states=None):
    """Occupancy, transition matrix (row-stochastic) and dwell times (frames)."""
    s = np.asarray(states, dtype=int)
    K = int(n_states or (s.max() + 1 if s.size else 1))
    occ = np.bincount(s, minlength=K) / max(len(s), 1)
    Tm = np.zeros((K, K))
    np.add.at(Tm, (s[:-1], s[1:]), 1.0)
    Tm = Tm / np.clip(Tm.sum(1, keepdims=True), 1, None)
    change = np.flatnonzero(np.diff(s)) + 1
    bounds = np.r_[0, change, len(s)]
    runs_state = s[bounds[:-1]]
    runs_len = np.diff(bounds)
    dwell = {k: runs_len[runs_state == k] for k in range(K)}
    return dict(occupancy=occ, transitions=Tm, dwell=dwell,
                mean_dwell=np.array([d.mean() if d.size else 0.0 for d in dwell.values()]))
