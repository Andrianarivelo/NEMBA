"""Automatic data profiling and embedding advice.

    from embedders_backbones import advise
    rec = advise(spike_times)                 # or a matrix, a file path, a DataFrame
    print(rec.report())                       # what the data is, what to do, and why
    emb = rec.fit(spike_times)                # run the suggested Embedder
    rec.run(spike_times, "out", behavior=b)   # ...and write every suggested output

What it measures (``analyze_data`` -> ``DataProfile``)
-------------------------------------------------------
* input kind and orientation (which axis is time), value type (counts /
  binary / continuous), missing values, dead or constant channels;
* the likely modality (spike times, binned counts, calcium traces, fiber
  photometry, pose keypoints, raw masks / pixels, generic signals);
* timescales: autocorrelation time of the dominant component, spectral peak,
  slow drift / photobleaching, non-stationarity;
* population structure: participation ratio and number of components for 80 /
  90 % of the variance, mean pairwise correlation;
* geometry of a quick smoothed PCA: a persistent loop (ring topology, ripser)
  and discrete-cluster structure (silhouette);
* behaviour variables (if given): task type (regression / classification /
  circular), timescale and alignment with the data.

How it decides (``suggest`` -> ``Recommendation``)
--------------------------------------------------
Explicit, documented rules. Every suggestion carries the measured value that
triggered it and its rationale (the NEMBA v0.1 vs v0.2 benchmark in
``benchmarks/results`` and the literature in ``docs/EMBEDDERS_REVIEW.md``).
The rules are heuristics, not ground truth: the report says so, and the
recommended comparison always includes a no-training baseline (PCA), because
on our benchmark it decoded behaviour as well as any trained model.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import data as _data
from . import decoding as _dec

# --------------------------------------------------------------------------- #
# thresholds (user-configurable; each is used by exactly one rule below)       #
# --------------------------------------------------------------------------- #
TH = dict(
    integer_frac=0.999,       # finite values that are integers -> counts
    wide_channels=128,        # more channels -> input PCA
    few_channels=3,           # at most this many channels -> "few-channel" signal
    short_frames=2000,        # fewer frames -> linear models only
    long_frames=200_000,      # more frames -> conv backbone, fast preset
    # normalised H1 persistence -> ring-like manifold. Calibrated on synthetic
    # seeds 0-4 (midpoint between the highest non-ring, 0.134, and the lowest
    # ring, 0.226), then checked on held-out seeds 5-9 (25/25 correct, but the
    # lowest ring there was 0.185: a thin margin). Only meaningful together
    # with the drift and cluster gates in ``_interpret`` (pose and metastable
    # data reach 0.33-0.50 through drift loops and state transitions).
    ring_persistence=0.18,
    strong_silhouette=0.50,   # Kaufman & Rousseeuw: reasonable cluster structure
    weak_silhouette=0.25,     # below: no substantial cluster structure
    drift_frac=0.25,          # variance fraction in the slow trend -> detrend / dF/F
    nan_warn=0.01,            # missing values fraction worth reporting
    calcium_skew=1.0,         # median channel skewness of calcium-like traces
    min_count_per_bin=0.1,    # spike binning target (median neuron)
    max_profile_frames=20000, # frames used for the timescale / spectrum analysis
    max_geometry_points=3000, # frames used for the geometry analysis
)
BIN_CANDIDATES = (0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0)


# --------------------------------------------------------------------------- #
# profile                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class DataProfile:
    """Measured facts about a dataset (no recommendations here)."""
    kind: str                                  # spike_times | matrix | table | pose | coco
    modality: str                              # see MODALITIES below
    modality_evidence: list = field(default_factory=list)
    n_channels: int = 0
    n_frames: int = 0
    fs: float | None = None
    duration_s: float | None = None
    time_axis: int = 1                         # axis of the ORIGINAL array that is time
    time_axis_confidence: str = "given"
    value_type: str = "continuous"             # counts | binary | continuous
    nan_frac: float = 0.0
    dead_channels: list = field(default_factory=list)
    columns: list | None = None
    # spikes
    bin_size: float | None = None
    rate_median_hz: float | None = None
    count_per_bin_median: float | None = None
    zero_frac: float | None = None
    fano_median: float | None = None
    # signal
    skew_median: float | None = None
    tau_frames: float | None = None
    dominant_freq_hz: float | None = None
    spectral_peak_ratio: float | None = None
    drift_frac: float | None = None
    bleaching: bool = False
    nonstationarity: float | None = None
    # population geometry
    participation_ratio: float | None = None
    n_pc80: int | None = None
    n_pc90: int | None = None
    mean_abs_corr: float | None = None
    ring_persistence: float | None = None
    silhouette_best: float | None = None
    k_best: int | None = None
    # interpreted flags (set by ``_interpret``; used by report and rules alike)
    drift_dominated: bool = False
    has_clusters: bool = False
    has_ring: bool = False
    # behaviour
    behavior: dict = field(default_factory=dict)
    seconds: float = 0.0
    _plots: dict = field(default_factory=dict, repr=False)

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return json.loads(json.dumps(d, default=_json_default))


MODALITIES = {
    "spike_times": "spike times (sorted units)",
    "binned_counts": "binned spike / event counts",
    "binary_events": "binary event raster",
    "calcium": "calcium-imaging traces (dF/F or similar)",
    "photometry": "fiber photometry / few-channel continuous signal",
    "pose": "pose keypoints (DeepLabCut / SLEAP)",
    "pixels": "raw masks / pixels (very wide binary or image data)",
    "oscillatory": "oscillatory continuous signal (LFP / EEG like)",
    "continuous": "generic continuous multichannel signal",
}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _lag1_median(A, max_series=200, max_len=20000):
    """Median lag-1 autocorrelation of the rows of A."""
    A = np.asarray(A, float)[:max_series, :max_len]
    A = A - np.nanmean(A, axis=1, keepdims=True)
    A = np.nan_to_num(A)
    num = (A[:, 1:] * A[:, :-1]).sum(1)
    den = (A * A).sum(1) + 1e-12
    return float(np.median(num / den))


def detect_time_axis(A):
    """Which axis of a 2-D array is time: (axis, confidence).

    Time is the axis along which neighbouring samples are most similar (higher
    lag-1 autocorrelation); ties go to the longer axis, which is time in
    almost every recording."""
    ac_rows = _lag1_median(A)               # time along axis 1
    ac_cols = _lag1_median(np.asarray(A).T)  # time along axis 0
    n0, n1 = A.shape
    if abs(ac_rows - ac_cols) >= 0.1:
        axis = 1 if ac_rows > ac_cols else 0
        conf = "high" if abs(ac_rows - ac_cols) >= 0.25 else "medium"
    else:
        axis = 1 if n1 >= n0 else 0
        conf = "medium" if max(n0, n1) >= 10 * min(n0, n1) else "low"
    return axis, conf


def _autocorr_tau(x, max_lag=None):
    """First lag where the autocorrelation of x drops below 1/e (frames)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if n < 8 or x.std() == 0:
        return 0.0, np.array([1.0])
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / ac[0]
    max_lag = max_lag or min(n - 1, 2000)
    below = np.flatnonzero(ac[:max_lag] < 1.0 / np.e)
    return float(below[0] if below.size else max_lag), ac[:max_lag]


def _choose_bin(spikes, duration):
    """Smallest candidate bin with >= min_count_per_bin spikes for the median neuron."""
    rates = np.array([len(s) / duration for s in spikes if len(s)])
    med = float(np.median(rates)) if rates.size else 0.0
    for b in BIN_CANDIDATES:
        if med * b >= TH["min_count_per_bin"] and duration / b >= 1000:
            return b, med
    return BIN_CANDIDATES[-1], med


def _circular_like(v):
    """Angles: range within [-pi, pi] or [0, 2 pi] and frequent wrap-around jumps."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 20:
        return False
    lo, hi = v.min(), v.max()
    in_range = (lo >= -math.pi - 1e-6 and hi <= math.pi + 1e-6) or (lo >= -1e-6 and hi <= 2 * math.pi + 1e-6)
    if not in_range or hi - lo < 1.5 * math.pi:
        return False
    jumps = np.abs(np.diff(v)) > math.pi
    return bool(jumps.sum() >= 2)


def analyze_data(data, *, fs=None, times=None, time_axis="auto", columns=None,
                 behavior=None, behavior_times=None, bin_size=None, seed=0):
    """Profile ``data`` (spike times, a 2-D array, a DataFrame, a COCO dict or
    a file path). Returns a ``DataProfile``. Nothing is modified."""
    t0 = time.perf_counter()
    hints = dict(kind=None, columns=columns, times=times, fs=fs)
    if isinstance(data, (str, bytes)) or hasattr(data, "__fspath__"):
        data, h = _data.load_auto(str(data))
        hints.update({k: v for k, v in h.items() if v is not None and hints.get(k) is None})
        hints["kind"] = h["kind"]
    if isinstance(data, pd.DataFrame):
        hints["columns"] = hints["columns"] or [str(c) for c in data.columns]
        data = data.select_dtypes(include=[np.number]).to_numpy(float)
        hints["kind"] = hints["kind"] or "table"
        time_axis = 0 if time_axis == "auto" else time_axis
    if isinstance(data, dict) and "annotations" in data:
        hints["kind"] = "coco"
    is_spikes = hints["kind"] == "spike_times" or (
        isinstance(data, (list, tuple)) and len(data) > 0 and
        all(np.ndim(s) == 1 for s in data) and len({len(s) for s in data}) > 1)
    fs = hints.get("fs") if fs is None else fs
    if fs is None and hints.get("times") is not None:
        fs = _data._fs_from_times(hints["times"])

    if hints["kind"] == "coco":
        prof = DataProfile(kind="coco", modality="pixels",
                           modality_evidence=["COCO segmentation annotations"], fs=fs)
        prof.seconds = time.perf_counter() - t0
        return prof

    if is_spikes:
        spikes = [np.asarray(s, float) for s in data]
        t_lo = min((s.min() for s in spikes if s.size), default=0.0)
        t_hi = max((s.max() for s in spikes if s.size), default=1.0)
        duration = max(t_hi - t_lo, 1e-9)
        b, med_rate = _choose_bin(spikes, duration) if bin_size is None else (bin_size, None)
        X, tt, counts = _data.spikes_to_matrix(spikes, b)
        prof = DataProfile(kind="spike_times", modality="spike_times",
                           modality_evidence=[f"{len(spikes)} spike trains (list of spike-time arrays)"],
                           bin_size=b, fs=1.0 / b, duration_s=float(duration),
                           time_axis=1, time_axis_confidence="given", value_type="counts")
        rates = np.array([len(s) / duration for s in spikes])
        prof.rate_median_hz = float(np.median(rates))
        prof.dead_channels = [int(i) for i in np.flatnonzero(rates == 0)]
        model_input, raw = X, counts
    else:
        A = np.asarray(data, dtype=np.float64)
        if A.ndim == 1:
            A = A[None, :]
        if A.ndim != 2:
            raise ValueError(f"expected spike times or a 2-D array, got shape {A.shape}")
        if time_axis == "auto":
            if hints["kind"] in ("table", "pose"):
                axis, conf = 0, "given (rows of a table are time)"
            else:
                axis, conf = detect_time_axis(A)
        else:
            axis, conf = int(time_axis), "given"
        M = A.T if axis == 0 else A                          # [C, T]
        prof = DataProfile(kind=hints["kind"] or "matrix", modality="continuous",
                           time_axis=axis, time_axis_confidence=conf, fs=fs,
                           columns=hints.get("columns"))
        if fs:
            prof.duration_s = M.shape[1] / fs
        fin = M[np.isfinite(M)]
        prof.nan_frac = float(1.0 - fin.size / max(M.size, 1))
        sd = np.nanstd(M, axis=1)
        prof.dead_channels = [int(i) for i in np.flatnonzero(~(sd > 1e-12))]
        if fin.size and np.mean(np.isclose(fin, np.round(fin))) >= TH["integer_frac"] and fin.min() >= 0:
            prof.value_type = "binary" if fin.max() <= 1 else "counts"
        _guess_modality(prof, M, hints)
        model_input = _data.zscore_rows(_data.interpolate_nans(M) if prof.nan_frac > 0 else M)
        if prof.value_type == "counts":
            model_input = _data.zscore_rows(np.sqrt(np.nan_to_num(M)))
        raw = M

    C, T = model_input.shape
    prof.n_channels, prof.n_frames = int(C), int(T)
    if prof.value_type in ("counts", "binary") or prof.kind == "spike_times":
        R = np.nan_to_num(np.asarray(raw, float))
        mu = R.mean(1)
        prof.count_per_bin_median = float(np.median(mu))
        prof.zero_frac = float((R == 0).mean())
        var = R.var(1)
        prof.fano_median = float(np.median(var[mu > 0] / mu[mu > 0])) if (mu > 0).any() else None
        if prof.fs and prof.rate_median_hz is None:
            prof.rate_median_hz = float(np.median(mu) * prof.fs)
    _signal_stats(prof, model_input, raw)
    _geometry(prof, model_input, seed)
    if behavior is not None:
        _behavior_profile(prof, behavior, behavior_times, times, T)
    _interpret(prof)
    prof.seconds = time.perf_counter() - t0
    return prof


def _interpret(prof):
    """Turn raw measurements into the three structural flags.

    * drift-dominated: the slow trend holds > ``drift_frac`` of the variance.
      Clusters and loops of such data mostly reflect elapsed time.
    * clusters: silhouette >= ``strong_silhouette`` AND not drift-dominated.
    * ring: loop persistence >= ``ring_persistence`` AND no strong clusters
      (transitions between a few discrete states also draw loops)."""
    prof.drift_dominated = bool((prof.drift_frac or 0) > TH["drift_frac"])
    prof.has_clusters = bool((prof.silhouette_best or 0) >= TH["strong_silhouette"]
                             and not prof.drift_dominated)
    prof.has_ring = bool((prof.ring_persistence or 0) >= TH["ring_persistence"]
                         and not prof.has_clusters and not prof.drift_dominated)


def _guess_modality(prof, M, hints):
    """Modality from value type, shape, names and NaN structure (in that order)."""
    C, T = M.shape
    ev = prof.modality_evidence
    names = [str(c).lower() for c in (hints.get("columns") or [])]
    if hints.get("kind") == "pose" or (names and sum(n.endswith(("_x", "_y", " x", " y")) for n in names) >= 0.6 * len(names)):
        prof.modality = "pose"
        ev.append("x / y keypoint columns" if names else "DeepLabCut file")
        return
    if prof.value_type == "binary":
        if C > TH["wide_channels"]:
            prof.modality = "pixels"; ev.append(f"binary values on {C} channels (masks / pixels)")
        else:
            prof.modality = "binary_events"; ev.append("binary 0 / 1 values")
        return
    if prof.value_type == "counts":
        prof.modality = "binned_counts"; ev.append("non-negative integers (counts)")
        return
    # NaN pattern shared by channel pairs: x / y of the same keypoint
    if prof.nan_frac > 0 and C % 2 == 0 and C >= 4:
        nanm = ~np.isfinite(M)
        same = np.mean([np.mean(nanm[2 * k] == nanm[2 * k + 1]) for k in range(C // 2)])
        if same > 0.99 and nanm.any():
            prof.modality = "pose"; ev.append("missing values shared by channel pairs (x / y)")
            return
    Mf = np.nan_to_num(M)
    if C > TH["wide_channels"] * 4:
        prof.modality = "pixels"; ev.append(f"{C} channels (image-like width)")
        return
    from scipy.stats import skew
    sk = skew(Mf, axis=1)
    prof.skew_median = float(np.median(sk))
    if C <= TH["few_channels"] * 2 and T > 1000 and _lag1_median(Mf) > 0.9:
        prof.modality = "photometry"
        ev.append(f"{C} smooth channel(s) (lag-1 autocorrelation {_lag1_median(Mf):.2f})")
        return
    if C >= 5 and prof.skew_median > TH["calcium_skew"]:
        prof.modality = "calcium"
        ev.append(f"positively skewed transients (median skewness {prof.skew_median:.1f})")
        return
    prof.modality = "continuous"
    ev.append("continuous values")


def _signal_stats(prof, X, raw):
    """Timescales, spectrum, drift and non-stationarity of the dominant component."""
    C, T = X.shape
    n = min(T, TH["max_profile_frames"])
    s0 = (T - n) // 2
    seg = np.nan_to_num(X[:, s0:s0 + n])
    if C > 1:
        u, s, vt = np.linalg.svd(seg - seg.mean(1, keepdims=True), full_matrices=False)
        pc1 = vt[0] * s[0]
    else:
        pc1 = seg[0]
    tau, ac = _autocorr_tau(pc1)
    prof.tau_frames = tau
    from scipy.signal import welch
    f, pxx = welch(pc1, fs=prof.fs or 1.0, nperseg=min(1024, n))
    if pxx.size > 8:
        # ignore the lowest bins: there a "peak" is just slow drift, not a rhythm
        lo = 4
        i = int(np.argmax(pxx[lo:]) + lo)
        prof.dominant_freq_hz = float(f[i]) if prof.fs else None
        prof.spectral_peak_ratio = float(pxx[i] / (np.median(pxx[1:]) + 1e-12))
    # slow trend: moving average over 10 % of the recording, per channel
    R = np.nan_to_num(np.asarray(raw, float))
    w = max(3, T // 10)
    trend = gaussian_filter1d(R, sigma=w / 3, axis=1)
    var_tot = R.var(1) + 1e-12
    prof.drift_frac = float(np.median(trend.var(1) / var_tot))
    # photobleaching: a continuous signal whose heavily smoothed mean decays
    # (almost) monotonically and whose slow trend carries much of the variance
    mean_sig = R.mean(0)
    slow = gaussian_filter1d(mean_sig, sigma=max(1.0, T / 30))[:: max(1, T // 100)]
    decreasing = float(np.mean(np.diff(slow) < 0)) if slow.size > 2 else 0.0
    prof.bleaching = bool(prof.value_type == "continuous" and decreasing > 0.8
                          and slow[0] > slow[-1] and prof.drift_frac > TH["drift_frac"])
    blocks = np.array_split(pc1, 4)
    v = np.array([b.var() for b in blocks]) + 1e-12
    prof.nonstationarity = float(v.max() / v.min())
    prof._plots.update(acf=ac[:min(len(ac), 300)], spec_f=f, spec_p=pxx,
                       heat=np.nan_to_num(X[:min(C, 64), :: max(1, T // 1500)]))


def _geometry(prof, X, seed):
    """Dimensionality, ring topology and cluster structure of a quick PCA."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from .analysis import subsample_index, circularity
    X = np.nan_to_num(np.asarray(X, float))
    C, T = X.shape
    if C > 256:                                  # very wide data: work on 64 PCs
        from sklearn.decomposition import PCA
        sub = X[:, :: max(1, T // 5000)].T
        pca = PCA(64, svd_solver="randomized", random_state=seed).fit(sub)
        X = pca.transform(X.T).T
        C = X.shape[0]
    idx = subsample_index(T, TH["max_geometry_points"])
    # smooth at full resolution with half the dominant timescale, then subsample
    sigma = max(1.0, (prof.tau_frames or 2.0) / 2.0)
    Xs = gaussian_filter1d(X, sigma=sigma, axis=1)[:, idx].T                # [n, C]
    Xs = Xs - Xs.mean(0)
    if C > 1:
        u, s, vt = np.linalg.svd(Xs, full_matrices=False)
        lam = s ** 2
        ev = lam / lam.sum()
        cum = np.cumsum(ev)
        prof.participation_ratio = float(lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-12))
        prof.n_pc80 = int(np.searchsorted(cum, 0.8) + 1)
        prof.n_pc90 = int(np.searchsorted(cum, 0.9) + 1)
        k = int(min(max(prof.n_pc90, 2), 10, C))
        P = u[:, :k] * s[:k]
        cc = np.corrcoef(X[:, idx])
        prof.mean_abs_corr = float(np.nanmean(np.abs(cc[np.triu_indices(C, 1)])))
        prof._plots.update(cumvar=cum[:min(30, len(cum))], P=P[:, :2], idx=idx)
    else:
        P = Xs
        prof.participation_ratio, prof.n_pc80, prof.n_pc90 = 1.0, 1, 1
    if P.shape[1] >= 2:
        try:
            prof.ring_persistence = float(circularity(P[:, :min(6, P.shape[1])], seed))
        except Exception:  # noqa: BLE001
            prof.ring_persistence = None
    best, kbest = -1.0, None
    sub = P[:: max(1, len(P) // 2000)]
    for k in range(2, 7):
        if len(sub) <= k + 1:
            break
        lab = KMeans(k, n_init=3, random_state=seed).fit_predict(sub)
        sc = float(silhouette_score(sub, lab))
        if sc > best:
            best, kbest = sc, k
    prof.silhouette_best, prof.k_best = (best if kbest else None), kbest


def _behavior_profile(prof, behavior, behavior_times, times, T):
    """Task type, timescale and alignment of each behaviour variable."""
    beh = _dec._as_dict(behavior)
    for name, v in beh.items():
        v = np.asarray(v)
        task = "circular" if _circular_like(v) else _dec.infer_task(v)
        tau = None
        if task != "classification" and v.dtype.kind in "fiu":
            tau, _ = _autocorr_tau(np.nan_to_num(v.astype(float)))
        aligned = len(v) == T or behavior_times is not None
        prof.behavior[name] = dict(task=task, n=int(len(v)), aligned=bool(aligned),
                                   levels=int(np.unique(v).size) if task == "classification" else None,
                                   tau_samples=tau)


# --------------------------------------------------------------------------- #
# recommendation                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class Recommendation:
    """A concrete, explained pipeline for one dataset."""
    profile: DataProfile
    goal: str
    fit_method: str
    fit_kwargs: dict
    embedder_kwargs: dict
    compare: list
    projections: list
    color_by: list
    visuals: list
    decoding: dict
    preprocessing: list
    reasons: list            # (topic, decision, evidence)
    warnings: list

    # ---------------- run ----------------
    def build(self, **overrides):
        """An ``Embedder`` configured as recommended."""
        from .core import Embedder
        kw = dict(self.embedder_kwargs)
        kw.update(overrides)
        return Embedder(**kw)

    def fit(self, data, **overrides):
        """Build and fit the recommended Embedder on ``data`` (same input as
        given to ``advise``); the recommendation is attached as
        ``emb.recommendation``."""
        if isinstance(data, (str, bytes)) or hasattr(data, "__fspath__"):
            data, _ = _data.load_auto(str(data))
        emb = self.build(**overrides)
        getattr(emb, self.fit_method)(data, **self.fit_kwargs)
        emb.recommendation = self
        return emb

    def run(self, data, outdir, behavior=None, behavior_times=None, **overrides):
        """Fit, then write the recommended outputs plus the advice itself
        (advice.md, advice.json, data_profile.png/pdf/svg)."""
        import os
        from . import viz as _viz
        emb = self.fit(data, **overrides)
        os.makedirs(outdir, exist_ok=True)
        circ = [n for n, d in self.profile.behavior.items() if d["task"] == "circular"]
        emb.run(outdir, behavior=behavior, behavior_times=behavior_times,
                methods=tuple(self.projections), circular=circ,
                n_perm=self.decoding.get("n_perm", 200))
        self.save(os.path.join(outdir, "advice"))
        _viz.save_figure(profile_figure(self.profile), os.path.join(outdir, "data_profile"))
        return emb

    # ---------------- describe ----------------
    def to_dict(self):
        return json.loads(json.dumps(dict(
            goal=self.goal, fit_method=self.fit_method, fit_kwargs=self.fit_kwargs,
            embedder_kwargs=self.embedder_kwargs, compare=self.compare,
            projections=self.projections, color_by=self.color_by, visuals=self.visuals,
            decoding=self.decoding, preprocessing=self.preprocessing,
            reasons=[dict(topic=t, decision=d, evidence=e) for t, d, e in self.reasons],
            warnings=self.warnings, profile=self.profile.to_dict()), default=_json_default))

    def save(self, stem):
        """Write ``stem``.md and ``stem``.json."""
        with open(f"{stem}.md", "w", encoding="utf-8") as fh:
            fh.write(self.report())
        with open(f"{stem}.json", "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return stem

    def code(self, data_name="data"):
        """Python snippet reproducing the recommendation."""
        kw = ", ".join(f"{k}={v!r}" for k, v in self.embedder_kwargs.items() if k != "backbone")
        fk = ", ".join(f"{k}={v!r}" for k, v in self.fit_kwargs.items())
        lines = ["from embedders_backbones import Embedder, compare_embedders",
                 f"emb = Embedder({self.embedder_kwargs['backbone']!r}, {kw})",
                 f"emb.{self.fit_method}({data_name}{', ' + fk if fk else ''})",
                 f"emb.run('out', behavior=behavior, methods={tuple(self.projections)!r})",
                 f"# honest check against no-training baselines:",
                 f"cmp = compare_embedders({data_name}, input={_INPUT_OF.get(self.fit_method, 'timeseries')!r}, "
                 f"backbones={self.compare!r}, speed={self.embedder_kwargs.get('speed', 'fast')!r})"]
        return "\n".join(lines)

    def report(self):
        """Human-readable advice (markdown)."""
        p = self.profile
        L = ["# NEMBA data advisor", "", "## What the data looks like", ""]
        L.append(f"* **{MODALITIES.get(p.modality, p.modality)}**: "
                 + "; ".join(p.modality_evidence))
        L.append(f"* {p.n_channels} channels x {p.n_frames:,} frames"
                 + (f" ({p.duration_s:.0f} s at {p.fs:.3g} Hz)" if p.fs and p.duration_s else "")
                 + f"; time axis {p.time_axis} ({p.time_axis_confidence} confidence)")
        if p.value_type != "continuous":
            L.append(f"* values: {p.value_type}; median count/bin {p.count_per_bin_median:.3g}"
                     + (f", median rate {p.rate_median_hz:.3g} Hz" if p.rate_median_hz else "")
                     + (f", Fano factor {p.fano_median:.2f}" if p.fano_median else ""))
        if p.nan_frac:
            L.append(f"* missing values: {100 * p.nan_frac:.2f} %")
        if p.dead_channels:
            L.append(f"* {len(p.dead_channels)} silent / constant channel(s)")
        if p.tau_frames is not None:
            tau_s = f" ({p.tau_frames / p.fs:.3g} s)" if p.fs else ""
            L.append(f"* dominant timescale: {p.tau_frames:.0f} frames{tau_s}"
                     + (f"; spectral peak {p.dominant_freq_hz:.3g} Hz (x{p.spectral_peak_ratio:.0f} median power)"
                        if p.dominant_freq_hz and p.spectral_peak_ratio and p.spectral_peak_ratio > 20 else ""))
        if p.drift_frac is not None:
            L.append(f"* slow drift: {100 * p.drift_frac:.0f} % of variance"
                     + ("; photobleaching-like decay" if p.bleaching else ""))
        if p.participation_ratio is not None and p.n_channels > 1:
            L.append(f"* dimensionality: participation ratio {p.participation_ratio:.1f}; "
                     f"{p.n_pc80} / {p.n_pc90} PCs for 80 / 90 % variance")
        if p.ring_persistence is not None:
            L.append(f"* loop (ring) persistence: {p.ring_persistence:.2f}"
                     + (" -> ring-like manifold" if p.has_ring else ""))
        if p.silhouette_best is not None:
            from .analysis import cluster_structure
            L.append(f"* cluster structure: silhouette {p.silhouette_best:.2f} at K={p.k_best} "
                     f"({cluster_structure(p.silhouette_best)})"
                     + (" -> discrete states" if p.has_clusters else
                        " (ignored: slow drift dominates)" if p.drift_dominated and
                        p.silhouette_best >= TH["strong_silhouette"] else ""))
        for name, b in p.behavior.items():
            L.append(f"* behaviour `{name}`: {b['task']}"
                     + (f", {b['levels']} levels" if b.get("levels") else "")
                     + ("" if b["aligned"] else " (NOT aligned: pass behavior_times)"))
        L += ["", "## Recommended pipeline", "", "```python", self.code(), "```", ""]
        L += ["| setting | value |", "|---|---|"]
        for k, v in self.embedder_kwargs.items():
            L.append(f"| `{k}` | `{v!r}` |")
        for k, v in self.fit_kwargs.items():
            L.append(f"| `{self.fit_method}.{k}` | `{v!r}` |")
        L += [f"| compare with | {', '.join(self.compare)} |",
              f"| projections | {', '.join(self.projections)} |",
              f"| colour by | {', '.join(self.color_by)} |",
              f"| figures | {', '.join(self.visuals)} |", ""]
        if self.preprocessing:
            L += ["## Preprocessing", ""] + [f"* {x}" for x in self.preprocessing] + [""]
        L += ["## Why", "", "| topic | decision | evidence |", "|---|---|---|"]
        for t, d, e in self.reasons:
            L.append(f"| {t} | {d} | {e} |")
        if self.warnings:
            L += ["", "## Warnings", ""] + [f"* {w}" for w in self.warnings]
        L += ["", "_These are rule-based suggestions from measured properties, not "
              "guarantees. Always check the recommended no-training baseline._", ""]
        return "\n".join(L)


_INPUT_OF = {"fit_spikes": "spikes", "fit_fiber": "fiber_photometry", "fit_features": "timeseries",
             "fit_coco_masks": "coco_mask", "fit_matrix": "matrix"}


def suggest(profile: DataProfile, goal="auto"):
    """Turn a ``DataProfile`` into a ``Recommendation``.

    goal: 'auto' | 'decode' (behaviour readout) | 'states' (discrete states) |
    'visualize' (geometry / trajectories)."""
    p = profile
    R, W, pre = [], [], []
    kw = dict(speed="fast", objective="masked", likelihood="gaussian", mask_mode="time",
              n_states=None, input_pca="auto", embedding_dim=32, smooth_sigma=4.0,
              state_model="auto")
    fit_kwargs = {}

    # ---------------------------------------------------------------- adapter
    if p.kind == "coco":
        fit_method = "fit_coco_masks"
        fit_kwargs.update(resize=(64, 64))
        R.append(("input", "raw masks rasterised to 64 x 64, input PCA to 64 components",
                  "4096 pixel channels are too wide for every backbone on a CPU"))
    elif p.kind == "spike_times":
        fit_method = "fit_spikes"
        kw["bin_size"] = p.bin_size
        R.append(("binning", f"bin_size = {p.bin_size} s",
                  f"smallest bin with >= {TH['min_count_per_bin']} spikes per bin for the median "
                  f"neuron ({p.rate_median_hz:.2g} Hz) and >= 1000 frames"))
    elif p.modality == "photometry":
        fit_method = "fit_fiber"
        fit_kwargs.update(time_axis=1 if p.time_axis == 1 else 0, fs=p.fs,
                          dff=bool(p.bleaching or (p.drift_frac or 0) > TH["drift_frac"]))
        if fit_kwargs["dff"]:
            pre.append("dF/F with a running-median baseline (or pass control= for an isosbestic "
                       "channel, which also removes motion artefacts)")
            R.append(("preprocessing", "dF/F", f"slow drift holds {100 * p.drift_frac:.0f} % of "
                      "the variance" + (" with photobleaching-like decay" if p.bleaching else "")))
    else:
        fit_method = "fit_features"
        fit_kwargs.update(time_axis=p.time_axis)
    if p.modality == "pose":
        pre.append("missing keypoints are interpolated linearly; consider egocentric "
                   "alignment (centre + heading) if absolute position is not of interest")
    if p.dead_channels:
        pre.append(f"{len(p.dead_channels)} silent / constant channel(s) carry no information "
                   "(they are z-scored to 0; drop them to save time)")
    if p.nan_frac > TH["nan_warn"]:
        W.append(f"{100 * p.nan_frac:.1f} % missing values: interpolation can create plateaus; "
                 "check the gaps before trusting fine timescales")
    if p.time_axis_confidence == "low":
        W.append("the time axis was ambiguous: pass time_axis= explicitly")

    # ---------------------------------------------------------------- backbone
    wide = p.n_channels > TH["wide_channels"]
    few = p.n_channels <= TH["few_channels"]
    ring, clustered = p.has_ring, p.has_clusters
    if p.drift_dominated and p.modality != "photometry":
        W.append(f"slow drift holds {100 * p.drift_frac:.0f} % of the variance: clusters and "
                 "trajectories may reflect elapsed time; detrend or high-pass the data, or check "
                 "that the drift is a real behavioural variable (e.g. position in an arena)")
    goal_eff = goal
    if goal == "auto":
        goal_eff = "states" if clustered else ("visualize" if ring else "decode")
    if p.n_frames < TH["short_frames"]:
        kw["backbone"] = "pca"
        R.append(("backbone", "pca (no training)",
                  f"only {p.n_frames} frames: deep encoders overfit; in the NEMBA benchmark PCA "
                  "decoded behaviour as well as any trained model even with 6000 frames"))
    elif few:
        kw["backbone"] = "tcn"
        kw["embedding_dim"] = 8
        R.append(("backbone", "tcn, embedding_dim = 8",
                  f"{p.n_channels} channel(s): the embedding can only describe the temporal "
                  "shape of the signal (a small convolutional encoder is enough)"))
        W.append("with <= 3 channels an embedding re-describes temporal waveform features; "
                 "event-aligned (peri-event) analysis is often more informative")
    elif goal_eff == "states":
        kw["backbone"] = "dtc"
        kw["state_model"] = "hmm"
        R.append(("backbone", "dtc (Transformer-AE + deep embedded clustering)",
                  f"clear cluster structure (silhouette {p.silhouette_best:.2f}); best mean state "
                  "recovery (ARI) in the NEMBA benchmark"))
        R.append(("states", "sticky HMM, K by held-out likelihood",
                  "temporal state models avoid jitter-driven transitions (keypoint-MoSeq, rSLDS); NEMBA "
                  "validation on unsmoothed embeddings: ARI 0.998 vs 0.837 for frame-wise K-means"))
    elif p.n_frames > TH["long_frames"]:
        kw["backbone"] = "tcn"
        R.append(("backbone", "tcn", f"{p.n_frames:,} frames: convolutional encoders scale "
                  "linearly and run fastest on a CPU"))
    else:
        kw["backbone"] = "tcn"
        R.append(("backbone", "tcn (masked, fast)",
                  "fast on CPU; no loss vs v0.1 in the benchmark; compare with pca / fa"))
    if p.modality == "pose" and kw["backbone"] != "pca":
        kw["state_model"] = "arhmm"
        R.append(("states", "AR-HMM", "pose states differ in dynamics rather than position "
                  "(MoSeq / keypoint-MoSeq use autoregressive HMMs)"))
    if wide:
        kw["input_pca"] = "auto"
        R.append(("input", "input PCA to 64 components",
                  f"{p.n_channels} channels: wide inputs make every backbone slow on a CPU"))
    # objective: follows benchmarks/results/objectives_stats.csv (6 paired fits,
    # 2 backbones x 3 recordings): no variant beat gaussian + time masks after
    # Holm correction; Poisson + neuron masks only *leaned* towards better
    # state recovery. So: default objective, except when states are the goal.
    counts_like = p.value_type == "counts" or p.kind == "spike_times"
    if counts_like and goal_eff == "states" and kw["backbone"] not in ("pca", "fa"):
        kw["likelihood"], kw["mask_mode"] = "auto", "both"
        R.append(("objective", "Poisson likelihood + time and neuron masks",
                  "goal = states: in the NEMBA objective benchmark this combination leaned "
                  "towards better state recovery (ARI +0.14, 95 % CI -0.08 to +0.35, not "
                  "significant) at a small cost in speed decoding"))
    elif counts_like and kw["backbone"] not in ("pca", "fa"):
        R.append(("objective", "gaussian likelihood + time masks (default)",
                  "no objective variant beat the default in the NEMBA objective benchmark "
                  "(6 paired fits, all Holm-corrected p > 0.4); Poisson + neuron masks remain "
                  "an option for state recovery"))

    # ---------------------------------------------------------------- dims, smoothing, window
    if p.n_pc90:
        kw["embedding_dim"] = int(min(32, max(kw["embedding_dim"] if few else 8, 2 * p.n_pc90)))
        R.append(("embedding_dim", str(kw["embedding_dim"]),
                  f"twice the {p.n_pc90} components needed for 90 % of the (smoothed) variance"))
    tau = p.tau_frames or 0.0
    kw["smooth_sigma"] = float(np.clip(round(tau / 2.0), 0, 8))
    capped = " (capped at 8 frames)" if tau / 2.0 > 8 else ""
    R.append(("smoothing", f"smooth_sigma = {kw['smooth_sigma']:.0f} frames",
              f"half the autocorrelation time ({tau:.0f} frames){capped}: smooths noise "
              "without blurring the dominant timescale"))
    if tau >= 32:
        win = int(min(512, 2 ** math.ceil(math.log2(4 * tau))))
        kw["win_len"] = win
        R.append(("window", f"win_len = {win}", f"windows must span several autocorrelation "
                  f"times ({tau:.0f} frames)"))

    # ---------------------------------------------------------------- states
    if clustered:
        kw["n_states"] = int(p.k_best)
        R.append(("n_states", str(p.k_best), f"silhouette {p.silhouette_best:.2f} (reasonable "
                  "or strong structure) on a quick PCA"))
    elif (p.silhouette_best or 0) < TH["weak_silhouette"]:
        W.append(f"no substantial cluster structure (silhouette {p.silhouette_best:.2f}): "
                 "states will be an arbitrary slicing of a continuous manifold")

    # ---------------------------------------------------------------- comparison
    compare = ["pca", "fa", kw["backbone"]] if kw["backbone"] not in ("pca", "fa") else ["pca", "fa", "tcn"]
    if p.kind == "spike_times" or p.value_type == "counts":
        R.append(("model choice", "rank backbones by co-smoothing (bits/spike)",
                  "label-free: no selection bias from choosing the best decoder on the same labels"))
    if few:
        compare = ["pca", "tcn"]

    # ---------------------------------------------------------------- representation
    projections = ["pca", "umap"] + (["tsne"] if p.n_frames <= 20000 else [])
    color_by = ["time", "state"] + list(p.behavior)
    visuals = ["dashboard", "explorer", "decoding" if p.behavior else "training"]
    if ring:
        visuals += ["persistence barcode", "flow field"]
        color_by.append("circular coordinate (ring angle)")
        R.append(("representation", "persistence barcode, flow field, cyclic colours",
                  f"loop persistence {p.ring_persistence:.2f} >= {TH['ring_persistence']}: "
                  "ring-like manifold (head-direction ring, Chaudhuri 2019; grid torus, Gardner 2022)"))
    if clustered or kw.get("state_model") in ("hmm", "arhmm"):
        visuals += ["state dynamics", "state-behaviour contingency" if p.behavior else "dwell times"]
    if tau >= 3:
        visuals += ["trajectory GIF", "recurrence plot"]
    if p.n_channels >= 20 and p.modality in ("spike_times", "binned_counts", "calcium"):
        visuals.append("Rastermap raster")
        R.append(("representation", "Rastermap-sorted raster",
                  f"{p.n_channels} neurons: sorting by activity similarity reveals sequences "
                  "and assemblies (Stringer et al. 2025)"))
    if p.n_frames > 20000:
        R.append(("projections", "pca + umap (t-SNE skipped)",
                  f"{p.n_frames:,} frames: t-SNE is slow and only a strided subset is shown"))

    # ---------------------------------------------------------------- decoding
    n_t = max(len(p.behavior), 1)
    decoding = dict(n_perm=int(max(200, 20 * n_t)), model="linear",
                    circular=[n for n, b in p.behavior.items() if b["task"] == "circular"],
                    tasks={n: b["task"] for n, b in p.behavior.items()})
    if p.behavior:
        R.append(("decoding", f"linear, blocked CV, {decoding['n_perm']} circular shifts",
                  "linear first (interpretable); n_perm >= 20 x targets so Holm-corrected p can "
                  "reach 0.05"))
    for n, b in p.behavior.items():
        if not b["aligned"]:
            W.append(f"behaviour '{n}' has {b['n']} samples for {p.n_frames} frames: pass "
                     "behavior_times to resample it")

    kw = {"backbone": kw.pop("backbone"), **kw}
    return Recommendation(profile=p, goal=goal_eff, fit_method=fit_method, fit_kwargs=fit_kwargs,
                          embedder_kwargs=kw, compare=compare, projections=projections,
                          color_by=color_by, visuals=visuals, decoding=decoding,
                          preprocessing=pre, reasons=R, warnings=W)


def advise(data, *, goal="auto", fs=None, times=None, time_axis="auto", columns=None,
           behavior=None, behavior_times=None, bin_size=None, seed=0):
    """Profile ``data`` and return a ``Recommendation`` (see module docstring)."""
    prof = analyze_data(data, fs=fs, times=times, time_axis=time_axis, columns=columns,
                        behavior=behavior, behavior_times=behavior_times, bin_size=bin_size,
                        seed=seed)
    return suggest(prof, goal=goal)


# --------------------------------------------------------------------------- #
# diagnostic figure                                                            #
# --------------------------------------------------------------------------- #

def profile_figure(profile: DataProfile):
    """Six-panel diagnostic figure of a ``DataProfile``."""
    import matplotlib
    from . import viz as V
    p, P = profile, profile._plots
    fig = V._fig(7.4, 4.6)
    with matplotlib.rc_context(V.RC):
        axes = fig.subplots(2, 3)
        a = axes[0, 0]
        if "heat" in P:
            a.imshow(P["heat"], aspect="auto", cmap=V.SEQ_CMAP, interpolation="nearest",
                     vmin=-2, vmax=2)
            a.set_xlabel("time (subsampled)"); a.set_ylabel("channel")
        a.set_title("activity (z-scored)", loc="left")
        a = axes[0, 1]
        if "acf" in P:
            ac = P["acf"]
            lag = np.arange(len(ac)) / (p.fs or 1.0)
            a.plot(lag, ac, color=V.CATEGORICAL[0], lw=1)
            a.axhline(1 / np.e, color=V.STYLE["ink_3"], lw=0.6)
            if p.tau_frames:
                a.axvline(p.tau_frames / (p.fs or 1.0), color=V.CATEGORICAL[7], lw=0.8)
            a.set_xlabel("lag (s)" if p.fs else "lag (frames)"); a.set_ylabel("autocorrelation")
            V._clean(a, grid_axis="y")
        a.set_title(f"timescale tau = {p.tau_frames:.0f} frames" if p.tau_frames else "timescale", loc="left")
        a = axes[0, 2]
        if "spec_f" in P and len(P["spec_f"]) > 2:
            a.semilogy(P["spec_f"][1:], P["spec_p"][1:], color=V.CATEGORICAL[0], lw=1)
            a.set_xlabel("frequency (Hz)" if p.fs else "frequency (1/frame)"); a.set_ylabel("power")
            V._clean(a, grid_axis="y")
        a.set_title("spectrum of PC1", loc="left")
        a = axes[1, 0]
        if "cumvar" in P:
            cv = P["cumvar"]
            a.plot(np.arange(1, len(cv) + 1), cv, "-o", ms=2.5, color=V.CATEGORICAL[0], lw=1)
            for lvl in (0.8, 0.9):
                a.axhline(lvl, color=V.STYLE["ink_3"], lw=0.6)
            a.set_xlabel("components"); a.set_ylabel("cumulative variance")
            V._clean(a, grid_axis="y")
        a.set_title(f"dimensionality (PR {p.participation_ratio:.1f})" if p.participation_ratio
                    else "dimensionality", loc="left")
        a = axes[1, 1]
        if "P" in P and P["P"].shape[1] >= 2:
            pts = P["P"]
            a.scatter(pts[:, 0], pts[:, 1], c=np.arange(len(pts)), cmap=V.SEQ_CMAP, s=2, lw=0,
                      rasterized=True)
            a.set_xticks([]); a.set_yticks([]); V._clean(a, keep=())
            a.set_aspect("equal", adjustable="box")
        ring = f"loop {p.ring_persistence:.2f}" if p.ring_persistence is not None else ""
        a.set_title(f"smoothed PCA ({ring})", loc="left")
        a = axes[1, 2]
        a.axis("off")
        lines = [MODALITIES.get(p.modality, p.modality),
                 f"{p.n_channels} ch x {p.n_frames:,} frames",
                 f"values: {p.value_type}",
                 f"drift: {100 * (p.drift_frac or 0):.0f} % var" + (" (bleaching)" if p.bleaching else ""),
                 f"silhouette: {p.silhouette_best:.2f} (K={p.k_best})" if p.silhouette_best is not None else "",
                 f"missing: {100 * p.nan_frac:.1f} %" if p.nan_frac else ""]
        a.text(0.0, 1.0, "\n".join(l for l in lines if l), va="top", ha="left",
               fontsize=V.STYLE["font_size"], color=V.STYLE["ink_2"], transform=a.transAxes,
               linespacing=1.6)
        a.set_title("summary", loc="left")
        fig.suptitle("data profile", fontsize=V.STYLE["title_size"])
    return fig
