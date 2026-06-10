"""The unified ``Embedder`` API: pick a backbone, fit it to any modality, get
embeddings / states / projections / metrics, and render figures + a GIF.

    from embedders_backbones import Embedder
    emb = Embedder(backbone="dtc", n_states=8, epochs=120)
    emb.fit_spikes(spike_times)              # or fit_features / fit_fiber / fit_matrix
    emb.run("out_dir")                       # embeddings.png + states.png + gif + metrics
"""

from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd

from . import backbones as BK
from . import data as _data
from . import analysis as _an
from . import viz as _viz


class Embedder:
    def __init__(self, backbone="dtc", *, bin_size=0.2, embedding_dim=32, n_states=8,
                 epochs=120, win_len=256, stride=128, batch=16, mask_ratio=0.4,
                 mask_span=16, smooth_sigma=4.0, device="auto", seed=0, log=print):
        if backbone not in BK.BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; choose {BK.list_backbones()}")
        self.backbone = backbone
        self.bin_size = bin_size
        self.embedding_dim = embedding_dim
        self.n_states = n_states
        self.epochs = epochs
        self.win_len = win_len
        self.stride = stride
        self.batch = batch
        self.mask_ratio = mask_ratio
        self.mask_span = mask_span
        self.smooth_sigma = smooth_sigma
        self.device = device
        self.seed = seed
        self.log = log
        self.X = self.times = self.net = self.Z = self.states = self.losses = None
        self._proj = {}

    # ----- fitting from different inputs ----- #
    def fit_spikes(self, spike_times, t_start=None, t_end=None, top_n=None):
        X, times, _ = _data.spikes_to_matrix(spike_times, self.bin_size, t_start, t_end, top_n)
        return self._fit(X, times)

    def fit_features(self, X, time_axis=1, times=None):
        Xz, times = _data.features_to_matrix(X, time_axis=time_axis, times=times)
        return self._fit(Xz, times)

    def fit_fiber(self, signal, time_axis=1, times=None, dff=False, fs=None):
        Xz, times = _data.fiber_photometry_to_matrix(signal, time_axis, times, dff, fs)
        return self._fit(Xz, times)

    def fit_matrix(self, X, times=None):
        X = np.asarray(X, dtype=np.float32)
        if times is None:
            times = np.arange(X.shape[1], dtype=float) * self.bin_size
        return self._fit(X, np.asarray(times, float))

    def _fit(self, X, times):
        self.X, self.times = X.astype(np.float32), times
        if self.log:
            self.log(f"[{self.backbone}] fitting on {X.shape[0]} channels x {X.shape[1]} bins")
        self.net, self.losses = _an.train(
            self.backbone, X, embedding_dim=self.embedding_dim, epochs=self.epochs,
            win_len=self.win_len, stride=self.stride, batch=self.batch,
            mask_ratio=self.mask_ratio, mask_span=self.mask_span,
            device=self.device, seed=self.seed, log=self.log)
        Z = _an.embed_sequence(self.net, X, self.embedding_dim, self.win_len, self.device)
        self.Z = _an.smooth_standardize(Z, self.smooth_sigma)
        self.states = _an.cluster(self.backbone, self.Z, self.n_states, self.seed)
        self._proj = {}
        return self

    # ----- access ----- #
    def embedding(self):
        self._check(); return self.Z

    def state_sequence(self):
        self._check(); return self.states

    def projection(self, method="umap"):
        self._check()
        m = method.lower()
        if m not in self._proj:
            self._proj[m] = _an.project(self.Z, m, self.seed)
        return self._proj[m]

    def metrics(self):
        self._check()
        m = _an.metrics(self.Z, self.seed)
        m.update(backbone=self.backbone, n_channels=int(self.X.shape[0]),
                 n_bins=int(self.X.shape[1]), n_states=int(np.unique(self.states).size))
        return m

    # ----- figures ----- #
    def plot_embeddings(self, save_path, methods=("pca", "tsne", "umap"), color="time"):
        self._check()
        return _viz.embedding_panel(self.Z, save_path, methods=methods, color=color,
                                    states=self.states, seed=self.seed,
                                    title=f"{BK.label(self.backbone)} embedding")

    def figure(self, methods=("pca", "tsne", "umap"), color="time", fig=None):
        """Return a matplotlib Figure of the embedding (for GUI display)."""
        self._check()
        return _viz.embedding_figure(self.Z, methods=methods, color=color,
                                     states=self.states, seed=self.seed,
                                     title=f"{BK.label(self.backbone)} embedding", fig=fig)

    def plot_state_sequence(self, save_path, behavior=None):
        self._check()
        return _viz.state_sequence(self.states, save_path, times=self.times,
                                   behavior=behavior, title=f"{BK.label(self.backbone)} states")

    def animate(self, save_path, method="umap", **kw):
        self._check()
        return _viz.trajectory_gif(self.Z, save_path, method=method, seed=self.seed,
                                   title=f"{BK.label(self.backbone)} trajectory", **kw)

    # ----- one-call pipeline ----- #
    def run(self, outdir, behavior=None, animate=True, gif_method="umap", prefix=None):
        self._check()
        prefix = prefix or self.backbone
        os.makedirs(outdir, exist_ok=True)
        self.plot_embeddings(os.path.join(outdir, f"{prefix}_embeddings.png"))
        self.plot_state_sequence(os.path.join(outdir, f"{prefix}_states.png"), behavior=behavior)
        if animate:
            self.animate(os.path.join(outdir, f"{prefix}_trajectory.gif"), method=gif_method)
        met = self.metrics()
        pd.DataFrame([met]).to_csv(os.path.join(outdir, f"{prefix}_metrics.csv"), index=False)
        with open(os.path.join(outdir, f"{prefix}_metrics.json"), "w") as fh:
            json.dump(met, fh, indent=2)
        np.savez_compressed(os.path.join(outdir, f"{prefix}_embedding.npz"),
                            Z=self.Z, states=self.states, times=self.times)
        if self.log:
            self.log(f"[{self.backbone}] wrote outputs to {outdir}")
        return met

    def _check(self):
        if self.Z is None:
            raise RuntimeError("call .fit_spikes()/.fit_features()/.fit_fiber()/.fit_matrix() first")
