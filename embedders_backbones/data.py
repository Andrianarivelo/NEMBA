"""Input adapters: turn any modality into a model-ready [C, T] matrix
(channels x time), z-scored per channel.

  spike trains      -> spikes_to_matrix         (list of spike-time arrays)
  pose / mask / any -> features_to_matrix        ([features, time] or [time, features])
  fiber photometry  -> fiber_photometry_to_matrix(continuous signal(s), optional dF/F)
  generic series    -> features_to_matrix / load_array

File loaders accept .npy / .npz / .csv (and pickle for spikes).
"""

from __future__ import annotations

import os
import numpy as np


def zscore_rows(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-9
    return ((X - mu) / sd).astype(np.float32)


# --------------------------------------------------------------------------- #
# spike trains                                                                 #
# --------------------------------------------------------------------------- #

def spikes_to_matrix(spike_times, bin_size: float = 0.2, t_start=None, t_end=None,
                     top_n=None, variance_stabilize: bool = True):
    """Bin spike trains -> [N, T] z-scored features. Returns (X, times, counts)."""
    spikes = [np.asarray(s, dtype=float) for s in spike_times]
    if t_start is None:
        t_start = min((s.min() for s in spikes if s.size), default=0.0)
    if t_end is None:
        t_end = max((s.max() for s in spikes if s.size), default=t_start + bin_size)
    edges = np.arange(t_start, t_end + bin_size, bin_size)
    times = edges[:-1] + bin_size / 2.0
    counts = np.zeros((len(spikes), times.size), dtype=np.float64)
    totals = np.zeros(len(spikes))
    for j, s in enumerate(spikes):
        s = s[(s >= t_start) & (s <= t_end)]
        c, _ = np.histogram(s, bins=edges)
        counts[j] = c; totals[j] = c.sum()
    keep = np.argsort(totals)[::-1]; keep = keep[totals[keep] > 0]
    if top_n is not None:
        keep = keep[:top_n]
    counts = counts[keep]
    feats = np.sqrt(counts) if variance_stabilize else counts
    return zscore_rows(feats), times, counts


# --------------------------------------------------------------------------- #
# pose / mask / generic feature matrices                                       #
# --------------------------------------------------------------------------- #

def features_to_matrix(X, time_axis: int = 1, times=None):
    """Z-score a [features, time] (or [time, features]) matrix. Returns (X, times)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {X.shape}")
    if time_axis == 0:
        X = X.T
    Xz = zscore_rows(X)
    if times is None:
        times = np.arange(Xz.shape[1], dtype=float)
    return Xz, np.asarray(times, float)


# --------------------------------------------------------------------------- #
# fiber photometry / continuous signals                                        #
# --------------------------------------------------------------------------- #

def fiber_photometry_to_matrix(signal, time_axis: int = 1, times=None,
                               dff: bool = False, fs: float | None = None,
                               detrend_window_s: float = 30.0):
    """Continuous photometry signal(s) -> [C, T] z-scored.

    ``dff`` computes a robust dF/F (subtract & divide by a sliding median baseline)
    before z-scoring; ``fs`` (Hz) sets the baseline window length.
    """
    X = np.asarray(signal, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if time_axis == 0:
        X = X.T
    if dff:
        win = max(int((fs or 1.0) * detrend_window_s), 5)
        base = np.apply_along_axis(lambda v: _sliding_median(v, win), 1, X)
        X = (X - base) / (np.abs(base) + 1e-9)
    Xz = zscore_rows(X)
    if times is None:
        times = (np.arange(Xz.shape[1]) / fs) if fs else np.arange(Xz.shape[1], dtype=float)
    return Xz, np.asarray(times, float)


def _sliding_median(v, win):
    pad = win // 2
    vp = np.pad(v, pad, mode="edge")
    return np.array([np.median(vp[i:i + win]) for i in range(len(v))])


# --------------------------------------------------------------------------- #
# file loaders                                                                 #
# --------------------------------------------------------------------------- #

def load_array(path: str) -> np.ndarray:
    """Load a 2-D array from .npy / .npz / .csv."""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        return np.asarray(d[d.files[0]])
    if path.endswith(".csv"):
        import pandas as pd
        return pd.read_csv(path).to_numpy(dtype=float)
    return np.load(path, allow_pickle=False)


def load_spikes(path: str):
    """Load spike trains: .npy object-array / .npz['spike_times'] / pickle."""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        key = "spike_times" if "spike_times" in d else d.files[0]
        return list(d[key])
    if path.endswith(".npy"):
        return list(np.load(path, allow_pickle=True))
    import pickle
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return list(obj["spike_times"]) if isinstance(obj, dict) else list(obj)
