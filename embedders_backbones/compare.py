"""Run several backbones on the same data and compare them.

    from embedders_backbones import compare_embedders
    cmp = compare_embedders(spike_times, input="spikes",
                            backbones=["dtc", "tst", "bilstm"], epochs=80)
    cmp.run("compare_out")        # metrics table + figures
    print(cmp.metrics)            # DataFrame, one row per backbone
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .core import Embedder
from . import viz as _viz
from .backbones import list_backbones

_METRIC_COLS = ["backbone", "n_states", "silhouette", "circularity", "diameter",
                "participation_ratio", "planarity", "mean_speed", "speed_cv"]


def compare_embedders(data, input="spikes", backbones=None, *, time_axis=1, fs=None,
                      dff=False, behavior=None, log=print, **embedder_kw):
    """Fit several backbones on the same data; return a ``Comparison``.

    ``input`` is one of spikes | pose | mask | fiber_photometry | timeseries.
    ``backbones`` defaults to a representative set. Extra kwargs (epochs, n_states,
    device, seed, ...) are passed to every ``Embedder``.
    """
    backbones = backbones or ["dtc", "tst", "bilstm", "stt", "gatr"]
    embedders = {}
    for bb in backbones:
        if log:
            log(f"\n=== fitting {bb} ===")
        emb = Embedder(bb, log=log, **embedder_kw)
        if input == "spikes":
            emb.fit_spikes(data)
        elif input == "fiber_photometry":
            emb.fit_fiber(data, time_axis=time_axis, fs=fs, dff=dff)
        else:
            emb.fit_features(data, time_axis=time_axis)
        embedders[bb] = emb
    return Comparison(embedders, behavior=behavior)


class Comparison:
    def __init__(self, embedders: dict, behavior=None):
        self.embedders = embedders
        self.behavior = behavior
        rows = []
        for bb, emb in embedders.items():
            m = emb.metrics()
            rows.append({c: m.get(c) for c in _METRIC_COLS})
        self.metrics = pd.DataFrame(rows)
        # shared time axis (first embedder's)
        self.times = next(iter(embedders.values())).times

    # ----- figures ----- #
    def plot_embeddings(self, save_path, method="umap", color="state"):
        return _viz.compare_embeddings_panel(self.embedders, save_path, method=method,
                                             color=color)

    def plot_state_sequences(self, save_path, behavior=None):
        seqs = {bb: e.state_sequence() for bb, e in self.embedders.items()}
        return _viz.state_sequences_panel(seqs, save_path, times=self.times,
                                          behavior=behavior if behavior is not None else self.behavior,
                                          title="state sequences by backbone")

    def plot_metric(self, metric, save_path):
        return _viz.compare_bar(self.metrics, metric, save_path)

    def best(self, metric="silhouette"):
        d = self.metrics.dropna(subset=[metric])
        return d.loc[d[metric].idxmax(), "backbone"] if len(d) else None

    def run(self, outdir, behavior=None):
        os.makedirs(outdir, exist_ok=True)
        self.metrics.to_csv(os.path.join(outdir, "comparison_metrics.csv"), index=False)
        self.plot_embeddings(os.path.join(outdir, "compare_embeddings.png"))
        self.plot_state_sequences(os.path.join(outdir, "compare_state_sequences.png"),
                                  behavior=behavior)
        for metric in ("silhouette", "circularity"):
            if self.metrics[metric].notna().any():
                self.plot_metric(metric, os.path.join(outdir, f"compare_{metric}.png"))
        return self.metrics
