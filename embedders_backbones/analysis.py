"""Training, embedding, clustering (KMeans / DEC / CRF), low-dimensional
projections (PCA / t-SNE / UMAP) and manifold metrics.

Optional deps (graceful fallback): umap-learn (UMAP), ripser (topological
circularity); both fall back to simpler equivalents if missing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from . import backbones as BK
from .backbones import (make_model, make_span_mask, masked_reconstruction_loss,
                        neuron_graph, DECAutoencoder, target_distribution, LinearChainCRF)

try:
    import umap  # noqa: F401
    _HAS_UMAP = True
except Exception:  # noqa: BLE001
    _HAS_UMAP = False
try:
    from ripser import ripser  # noqa: F401
    _HAS_RIPSER = True
except Exception:  # noqa: BLE001
    _HAS_RIPSER = False


def resolve_device(device="auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --------------------------------------------------------------------------- #
# windowing / training / embedding                                            #
# --------------------------------------------------------------------------- #

def make_windows(X, L, stride):
    C, T = X.shape
    if T <= L:
        pad = np.zeros((C, L), dtype=X.dtype); pad[:, :T] = X
        return pad[None]
    starts = list(range(0, T - L + 1, stride))
    if starts[-1] != T - L:
        starts.append(T - L)
    return np.stack([X[:, s:s + L] for s in starts], axis=0)


def train(name, X, *, embedding_dim=32, epochs=120, win_len=256, stride=128, batch=16,
          lr=1e-3, weight_decay=1e-4, mask_ratio=0.4, mask_span=16, device="auto",
          seed=0, log=print):
    dev = resolve_device(device)
    torch.manual_seed(seed); np.random.seed(seed)
    net = make_model(name, X.shape[0], embedding_dim=embedding_dim, max_len=win_len).to(dev)
    if BK.needs_graph(name):
        net.set_graph(neuron_graph(X, top_k=min(8, max(X.shape[0] - 1, 1))))
    net.train()
    win = torch.from_numpy(make_windows(X, win_len, stride).astype(np.float32))
    W = win.shape[0]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(W, generator=gen); ep_loss, nb = 0.0, 0
        for i in range(0, W, batch):
            xb = win[perm[i:i + batch]].to(dev)
            mask = make_span_mask(xb.shape[0], win_len, mask_ratio, mask_span,
                                  device=dev, generator=gen)
            out = net(xb, mask)
            loss = masked_reconstruction_loss(xb, out.reconstruction, mask)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            ep_loss += float(loss); nb += 1
        losses.append(ep_loss / max(nb, 1))
        if log and (ep % 20 == 0 or ep == epochs - 1):
            log(f"  epoch {ep:3d}/{epochs}  loss={losses[-1]:.4f}")
    return net, losses


@torch.no_grad()
def embed_sequence(net, X, emb_dim, win_len=256, device="auto"):
    dev = resolve_device(device); net.eval()
    C, T = X.shape; Z = np.zeros((emb_dim, T), dtype=np.float32)
    for s in range(0, T, win_len):
        e = min(s + win_len, T)
        seg = np.zeros((C, win_len), dtype=np.float32); seg[:, :e - s] = X[:, s:e]
        z = net.encode(torch.from_numpy(seg)[None].to(dev)).squeeze(0).cpu().numpy()
        Z[:, s:e] = z[:, :e - s]
    return Z.T


def smooth_standardize(Z, sigma=4.0):
    Z = gaussian_filter1d(Z, sigma=sigma, axis=0)
    return StandardScaler().fit_transform(Z).astype(np.float32)


# --------------------------------------------------------------------------- #
# clustering (kmeans / DEC / CRF, dispatched by backbone)                     #
# --------------------------------------------------------------------------- #

def cluster(name, Z, k=8, seed=0):
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


def _crf_states(Z, km_labels, k, seed):
    torch.manual_seed(seed)
    Zs = StandardScaler().fit_transform(Z).astype(np.float32)
    crf = LinearChainCRF(emb_dim=Zs.shape[1], num_tags=k)
    opt = torch.optim.AdamW(crf.parameters(), lr=5e-3, weight_decay=1e-4)
    zt = torch.from_numpy(Zs); tags = torch.from_numpy(km_labels.astype(np.int64))
    for _ in range(80):
        loss = crf.nll(zt, tags); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(crf.parameters(), 5.0); opt.step()
    return crf.viterbi(zt).astype(int)


# --------------------------------------------------------------------------- #
# projections + metrics                                                       #
# --------------------------------------------------------------------------- #

def project(Z, method="umap", seed=0):
    method = method.lower()
    if method == "pca":
        return PCA(2, random_state=seed).fit_transform(Z)
    if method in ("tsne", "t-sne"):
        return TSNE(n_components=2, perplexity=min(30, max(5, len(Z) // 10)),
                    init="pca", random_state=seed).fit_transform(Z)
    if method == "umap":
        if not _HAS_UMAP:
            return PCA(2, random_state=seed).fit_transform(Z)
        import umap
        return umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(Z)
    raise ValueError(f"unknown projection {method!r}")


def circularity(Z, seed=0):
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
