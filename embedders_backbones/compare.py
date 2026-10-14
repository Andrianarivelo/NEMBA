"""Run several backbones on the same data and compare them.

    from embedders_backbones import compare_embedders
    cmp = compare_embedders(spike_times, input="spikes",
                            backbones=["pca", "tcn", "tst", "bilstm"], speed="fast",
                            behavior={"x": x, "speed": v})
    cmp.run("compare_out")        # metrics + decoding tables + figures
    print(cmp.metrics)            # DataFrame, one row per backbone

Always include ``"pca"`` (no training) in a comparison: a deep backbone that
does not beat it on held-out decoding is not worth its CPU time.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .core import Embedder
from . import viz as _viz

_METRIC_COLS = ["backbone", "n_states", "silhouette", "circularity", "diameter",
                "participation_ratio", "planarity", "mean_speed", "speed_cv",
                "train_steps", "stopped_early", "n_params", "train_seconds", "total_seconds"]


def compare_embedders(data, input="spikes", backbones=None, *, time_axis=1, fs=None,
                      dff=False, behavior=None, behavior_times=None, circular=(),
                      log=print, mask_resize=(64, 64), mask_channel_key=None,
                      decode_model="linear", n_perm=200, **embedder_kw):
    """Fit several backbones on the same data; return a ``Comparison``.

    ``input`` is one of spikes | pose | mask | coco_mask | fiber_photometry | timeseries.
    ``backbones`` defaults to a CPU-friendly representative set. Extra kwargs
    (speed, objective, n_states, device, seed, ...) go to every ``Embedder``.
    When ``behavior`` is given every backbone is also decoded identically.
    """
    backbones = backbones or ["pca", "tcn", "tst", "bilstm", "dtc"]
    embedders = {}
    for bb in backbones:
        if log:
            log(f"\n=== fitting {bb} ===")
        emb = Embedder(bb, log=log, **embedder_kw)
        if input == "spikes":
            emb.fit_spikes(data)
        elif input == "coco_mask":
            emb.fit_coco_masks(data, resize=mask_resize, channel_key=mask_channel_key, fps=fs)
        elif input == "fiber_photometry":
            emb.fit_fiber(data, time_axis=time_axis, fs=fs, dff=dff)
        elif input == "matrix":
            emb.fit_matrix(data)
        else:
            emb.fit_features(data, time_axis=time_axis)
        if behavior is not None:
            emb.decode(behavior, behavior_times, circular=circular, model=decode_model,
                       n_perm=n_perm, baseline=(bb == backbones[0]))
        embedders[bb] = emb
    return Comparison(embedders, behavior=behavior, behavior_times=behavior_times,
                      circular=circular)


class Comparison:
    def __init__(self, embedders: dict, behavior=None, behavior_times=None, circular=()):
        self.embedders = embedders
        self.behavior = behavior
        self.behavior_times = behavior_times
        self.circular = circular
        rows = []
        for bb, emb in embedders.items():
            m = emb.metrics()
            rows.append({c: m.get(c) for c in _METRIC_COLS})
        self.metrics = pd.DataFrame(rows)
        tabs = [e.decoding.table for e in embedders.values() if e.decoding is not None]
        self.decoding = pd.concat(tabs, ignore_index=True) if tabs else None
        # shared time axis (first embedder's)
        self.times = next(iter(embedders.values())).times

    # ----- figures ----- #
    def plot_embeddings(self, save_path, method="umap", color="state", formats=None):
        return _viz.compare_embeddings_panel(self.embedders, save_path, method=method,
                                             color=color, formats=formats)

    def plot_state_sequences(self, save_path, behavior=None, formats=None):
        seqs = {bb: e.state_sequence() for bb, e in self.embedders.items()}
        beh = behavior if behavior is not None else self._first_behavior()
        return _viz.state_sequences_panel(seqs, save_path, times=self.times,
                                          behavior=beh[1] if beh else None,
                                          behavior_name=beh[0] if beh else "behaviour",
                                          title="state sequences by backbone", formats=formats)

    def plot_metric(self, metric, save_path, formats=None):
        return _viz.compare_bar(self.metrics, metric, save_path, formats=formats)

    def decoding_table(self):
        """Wide table: decoding score per backbone (rows) x target (columns)."""
        if self.decoding is None:
            return None
        d = self.decoding[self.decoding.features == "embedding"]
        return d.pivot_table(index="backbone", columns="target", values="score")

    def best(self, metric="silhouette"):
        """Backbone with the highest ``metric`` (a metrics column, or
        ``decoding:<target>`` for a held-out decoding score)."""
        if metric.startswith("decoding:") and self.decoding is not None:
            tgt = metric.split(":", 1)[1]
            d = self.decoding[(self.decoding.features == "embedding") & (self.decoding.target == tgt)]
            return d.loc[d["score"].idxmax(), "backbone"] if len(d) else None
        d = self.metrics.dropna(subset=[metric])
        return d.loc[d[metric].idxmax(), "backbone"] if len(d) else None

    def _first_behavior(self):
        if self.behavior is None:
            return None
        first = next(iter(self.embedders.values()))
        beh = first._align(self.behavior, self.behavior_times, None, self.circular)
        return next(iter(beh.items()))

    def consistency(self):
        """Pairwise linear consistency (cross-validated R^2 of a linear map
        between embeddings; row predicts column), as in CEBRA."""
        from .align import consistency_matrix
        return consistency_matrix({bb: e.embedding() for bb, e in self.embedders.items()})

    def run(self, outdir, behavior=None, formats=None):
        os.makedirs(outdir, exist_ok=True)
        if len(self.embedders) >= 2:
            try:
                from .viz_dynamics import consistency_figure
                cm = self.consistency()
                cm.to_csv(os.path.join(outdir, "comparison_consistency.csv"))
                _viz.save_figure(consistency_figure(cm), os.path.join(outdir, "compare_consistency"),
                                 formats)
            except Exception:  # noqa: BLE001  (optional figure)
                pass
        self.metrics.to_csv(os.path.join(outdir, "comparison_metrics.csv"), index=False)
        self.plot_embeddings(os.path.join(outdir, "compare_embeddings"), formats=formats)
        self.plot_state_sequences(os.path.join(outdir, "compare_state_sequences"),
                                  behavior=behavior, formats=formats)
        for metric in ("silhouette", "circularity", "train_seconds"):
            if metric in self.metrics and self.metrics[metric].notna().any():
                self.plot_metric(metric, os.path.join(outdir, f"compare_{metric}"), formats)
        if self.decoding is not None:
            self.decoding.to_csv(os.path.join(outdir, "comparison_decoding.csv"), index=False)
            emb_rows = self.decoding[self.decoding.features == "embedding"]
            base = self.decoding[self.decoding.features == "raw_pca"]
            for tgt in emb_rows["target"].unique():
                d = emb_rows[emb_rows.target == tgt][["backbone", "score", "score_sd"]].copy()
                if len(base[base.target == tgt]):
                    b = base[base.target == tgt].iloc[0]
                    d = pd.concat([d, pd.DataFrame([dict(backbone="raw PCA (input)",
                                                         score=b["score"], score_sd=b["score_sd"])])])
                metric = emb_rows[emb_rows.target == tgt]["metric"].iloc[0]
                d = d.rename(columns={"score": metric})
                _viz.compare_bar(d, metric, os.path.join(outdir, f"compare_decoding_{tgt}"),
                                 title=f"held-out decoding of {tgt}", formats=formats,
                                 err_col="score_sd")
        return self.metrics
