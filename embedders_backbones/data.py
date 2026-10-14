"""Input adapters: turn any modality into a model-ready [C, T] matrix
(channels x time), z-scored per channel.

  spike trains      -> spikes_to_matrix         (list of spike-time arrays)
  pose / any feature -> features_to_matrix       ([features, time] or [time, features])
  COCO masks      -> coco_masks_to_matrix        (raw raster masks -> [pixels, time])
  fiber photometry  -> fiber_photometry_to_matrix(continuous signal(s), optional dF/F)
  generic series    -> features_to_matrix / load_array

File loaders accept .npy / .npz / .csv / COCO .json (and pickle for spikes).
"""

from __future__ import annotations

import json
import os
import numpy as np
from PIL import Image, ImageDraw

_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")


def zscore_rows(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-9
    return ((X - mu) / sd).astype(np.float32)


# --------------------------------------------------------------------------- #
# spike trains                                                                 #
# --------------------------------------------------------------------------- #

def spikes_to_matrix(spike_times, bin_size: float = 0.2, t_start=None, t_end=None,
                     top_n=None, variance_stabilize: bool = True, return_index: bool = False):
    """Bin spike trains -> [N, T] z-scored features. Returns (X, times, counts).

    Silent neurons are dropped and rows are ordered by total spike count;
    ``return_index=True`` also returns the original index of every kept row
    (needed to map channels back to neuron identities)."""
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
    if return_index:
        return zscore_rows(feats), times, counts, keep
    return zscore_rows(feats), times, counts


# --------------------------------------------------------------------------- #
# pose / mask / generic feature matrices                                       #
# --------------------------------------------------------------------------- #

def interpolate_nans(X: np.ndarray, max_gap=None) -> np.ndarray:
    """Linearly interpolate NaN / inf gaps along time in every row of X [C, T].

    Pose trackers (DeepLabCut, SLEAP) emit NaN for low-confidence frames;
    replacing them by the row mean (what z-scoring + nan_to_num does) creates
    artificial jumps. Rows that are entirely missing are left as NaN (they
    become 0 after z-scoring). ``max_gap`` (frames) leaves longer gaps as NaN."""
    X = np.array(X, dtype=np.float64, copy=True)
    t = np.arange(X.shape[1])
    for i in range(X.shape[0]):
        bad = ~np.isfinite(X[i])
        if not bad.any() or bad.all():
            continue
        good = ~bad
        filled = np.interp(t, t[good], X[i, good])
        if max_gap is not None:
            run_id = np.cumsum(np.r_[0, np.diff(bad.astype(int)) != 0])
            for r in np.unique(run_id[bad]):
                if np.count_nonzero(run_id == r) > max_gap:
                    filled[run_id == r] = np.nan
        X[i] = filled
    return X


def features_to_matrix(X, time_axis: int = 1, times=None, interpolate: bool = True):
    """Z-score a [features, time] (or [time, features]) matrix. Returns (X, times).

    NaN gaps are linearly interpolated first (``interpolate=True``)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {X.shape}")
    if time_axis == 0:
        X = X.T
    if interpolate and not np.isfinite(X).all():
        X = interpolate_nans(X)
    Xz = zscore_rows(X)
    if times is None:
        times = np.arange(Xz.shape[1], dtype=float)
    return Xz, np.asarray(times, float)


# --------------------------------------------------------------------------- #
# direct COCO segmentation masks                                               #
# --------------------------------------------------------------------------- #

def coco_masks_to_matrix(coco, *, resize=(64, 64), channel_key: str | None = None,
                         category_ids=None, track_ids=None, times=None,
                         fps: float | None = None, standardize: bool = True,
                         bbox_fallback: bool = True):
    """Rasterize COCO JSON masks directly into a model-ready [C, T] matrix.

    Each time step is a binary mask image, optionally split into stable channels
    by ``track_id`` or ``category_id``, then flattened to pixels-over-time. No
    geometry features such as area, centroid, contours, or skeletons are
    extracted.
    """
    if isinstance(coco, (str, os.PathLike)):
        with open(coco, "r", encoding="utf-8") as fh:
            coco = json.load(fh)
    if not isinstance(coco, dict):
        raise ValueError("coco must be a dict or a path to a COCO JSON file")

    resize_hw = _parse_resize(resize)
    images = list(coco.get("images", []))
    if not images:
        raise ValueError("COCO payload has no images")
    images = sorted(images, key=_image_sort_key)
    image_by_id = {img.get("id"): img for img in images}
    frame_index = {img.get("id"): idx for idx, img in enumerate(images)}

    cat_filter = set(category_ids) if category_ids is not None else None
    track_filter = set(track_ids) if track_ids is not None else None
    anns_by_image: dict[object, list[dict]] = {img.get("id"): [] for img in images}
    channel_values = []
    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id not in image_by_id:
            continue
        if cat_filter is not None and ann.get("category_id") not in cat_filter:
            continue
        if track_filter is not None and ann.get("track_id") not in track_filter:
            continue
        anns_by_image.setdefault(image_id, []).append(ann)
        if channel_key not in (None, "union"):
            channel_values.append(ann.get(channel_key, "missing"))

    if channel_key in (None, "union"):
        channels = ["union"]
    else:
        channels = sorted(set(channel_values), key=lambda x: (str(type(x)), str(x)))
        if not channels:
            channels = ["missing"]
    channel_to_idx = {value: idx for idx, value in enumerate(channels)}

    height, width = resize_hw
    video = np.zeros((len(channels), len(images), height, width), dtype=np.float32)
    for image_id, anns in anns_by_image.items():
        img = image_by_id[image_id]
        frame = frame_index[image_id]
        src_hw = (int(img.get("height", height)), int(img.get("width", width)))
        for ann in anns:
            key = "union" if channel_key in (None, "union") else ann.get(channel_key, "missing")
            channel = channel_to_idx.get(key)
            if channel is None:
                continue
            mask = _annotation_mask(ann, src_hw, resize_hw, bbox_fallback=bbox_fallback)
            video[channel, frame] = np.maximum(video[channel, frame], mask)

    X = video.transpose(0, 2, 3, 1).reshape(len(channels) * height * width, len(images))
    if standardize:
        X = zscore_rows(X)
    if times is None:
        times = _times_from_images(images, fps=fps)
    return X.astype(np.float32), np.asarray(times, float)


def load_coco_masks(path: str, **kwargs):
    """Load a COCO JSON file as direct raw mask pixels over time."""
    return coco_masks_to_matrix(path, **kwargs)


def _parse_resize(resize) -> tuple[int, int]:
    if isinstance(resize, int):
        return int(resize), int(resize)
    if len(resize) != 2:
        raise ValueError("resize must be an int or a (height, width) pair")
    return int(resize[0]), int(resize[1])


def _image_sort_key(img: dict):
    for key in ("frame_index", "frame_id", "frame", "index"):
        if key in img:
            return (0, int(img[key]))
    file_name = str(img.get("file_name", ""))
    digits = "".join(ch if ch.isdigit() else " " for ch in file_name).split()
    if digits:
        return (1, int(digits[-1]))
    return (2, int(img.get("id", 0)))


def _times_from_images(images: list[dict], fps: float | None = None) -> np.ndarray:
    values = []
    has_time = False
    for idx, img in enumerate(images):
        if "time" in img:
            values.append(float(img["time"])); has_time = True
        elif "timestamp" in img:
            values.append(float(img["timestamp"])); has_time = True
        elif fps:
            values.append(idx / float(fps))
        else:
            for key in ("frame_index", "frame_id", "frame", "index"):
                if key in img:
                    values.append(float(img[key])); break
            else:
                values.append(float(idx))
    return np.asarray(values if has_time or fps else np.arange(len(images), dtype=float), float)


def _annotation_mask(ann: dict, src_hw: tuple[int, int], dst_hw: tuple[int, int],
                     bbox_fallback: bool = True) -> np.ndarray:
    segmentation = ann.get("segmentation")
    if segmentation:
        mask = _decode_segmentation(segmentation, src_hw)
    elif bbox_fallback and ann.get("bbox") is not None:
        mask = _bbox_mask(ann["bbox"], src_hw)
    else:
        mask = np.zeros(src_hw, dtype=np.uint8)
    if src_hw != dst_hw:
        img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        img = img.resize((dst_hw[1], dst_hw[0]), _NEAREST)
        mask = (np.asarray(img) > 0).astype(np.float32)
    return mask.astype(np.float32)


def _decode_segmentation(segmentation, src_hw: tuple[int, int]) -> np.ndarray:
    if isinstance(segmentation, list):
        return _polygon_mask(segmentation, src_hw)
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        size = tuple(segmentation.get("size", src_hw))
        if isinstance(counts, list):
            return _rle_counts_mask(counts, size, src_hw)
        return _compressed_rle_mask(segmentation, src_hw)
    raise ValueError(f"unsupported COCO segmentation type: {type(segmentation).__name__}")


def _polygon_mask(polygons, src_hw: tuple[int, int]) -> np.ndarray:
    height, width = src_hw
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        if not poly or len(poly) < 6:
            continue
        coords = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
        draw.polygon(coords, outline=1, fill=1)
    return np.asarray(img, dtype=np.uint8)


def _bbox_mask(bbox, src_hw: tuple[int, int]) -> np.ndarray:
    height, width = src_hw
    x, y, w, h = [float(v) for v in bbox]
    x0, y0 = max(int(np.floor(x)), 0), max(int(np.floor(y)), 0)
    x1, y1 = min(int(np.ceil(x + w)), width), min(int(np.ceil(y + h)), height)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def _rle_counts_mask(counts: list[int], size: tuple[int, int], src_hw: tuple[int, int]) -> np.ndarray:
    height, width = int(size[0]), int(size[1])
    flat = np.zeros(height * width, dtype=np.uint8)
    idx = 0
    value = 0
    for run in counts:
        run = int(run)
        if value == 1:
            flat[idx:idx + run] = 1
        idx += run
        value = 1 - value
    mask = flat.reshape((height, width), order="F")
    if (height, width) != src_hw:
        img = Image.fromarray(mask * 255)
        img = img.resize((src_hw[1], src_hw[0]), _NEAREST)
        mask = (np.asarray(img) > 0).astype(np.uint8)
    return mask


def _compressed_rle_mask(segmentation: dict, src_hw: tuple[int, int]) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "compressed COCO RLE requires pycocotools; install it or export "
            "polygon/uncompressed RLE segmentations"
        ) from exc
    decoded = mask_utils.decode(segmentation)
    if decoded.ndim == 3:
        decoded = decoded.max(axis=2)
    if decoded.shape != src_hw:
        img = Image.fromarray((decoded > 0).astype(np.uint8) * 255)
        img = img.resize((src_hw[1], src_hw[0]), _NEAREST)
        decoded = np.asarray(img) > 0
    return decoded.astype(np.uint8)


# --------------------------------------------------------------------------- #
# fiber photometry / continuous signals                                        #
# --------------------------------------------------------------------------- #

def fiber_photometry_to_matrix(signal, time_axis: int = 1, times=None,
                               dff: bool = False, fs: float | None = None,
                               detrend_window_s: float = 30.0, control=None):
    """Continuous photometry signal(s) -> [C, T] z-scored.

    ``dff`` computes a robust dF/F (subtract & divide by a sliding median baseline)
    before z-scoring; ``fs`` (Hz) sets the baseline window length.
    ``control`` (isosbestic ~405 nm channel, same shape as ``signal``) instead
    fits the control to the signal by least squares and uses
    dF/F = (signal - fit) / fit, removing motion / bleaching artefacts shared by
    both channels (Lerner et al. 2015). This takes precedence over ``dff``.
    """
    X = np.asarray(signal, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if time_axis == 0:
        X = X.T
    if control is not None:
        Cn = np.asarray(control, dtype=np.float64)
        Cn = Cn[None, :] if Cn.ndim == 1 else (Cn.T if time_axis == 0 else Cn)
        if Cn.shape != X.shape:
            raise ValueError(f"control shape {Cn.shape} != signal shape {X.shape}")
        fit = np.empty_like(X)
        for i in range(X.shape[0]):
            a, b = np.polyfit(Cn[i], X[i], 1)
            fit[i] = a * Cn[i] + b
        X = (X - fit) / (np.abs(fit) + 1e-9)
    elif dff:
        win = max(int((fs or 1.0) * detrend_window_s), 5)
        base = np.vstack([_sliding_median(v, win) for v in X])
        X = (X - base) / (np.abs(base) + 1e-9)
    Xz = zscore_rows(X)
    if times is None:
        times = (np.arange(Xz.shape[1]) / fs) if fs else np.arange(Xz.shape[1], dtype=float)
    return Xz, np.asarray(times, float)


def _sliding_median(v, win):
    """Running median baseline (edge-padded), fast for any window length.

    Short windows use SciPy's C median filter directly. Long windows (a 30 s
    window at 1 kHz is 30 000 samples) first take medians of blocks of
    ``win // 50`` samples, run the median filter on the block series and
    interpolate back: a close approximation of the exact running median for
    slowly varying baselines, at a tiny fraction of the cost."""
    from scipy.ndimage import median_filter
    v = np.asarray(v, dtype=np.float64)
    win = int(max(3, win))
    if win <= 301 or len(v) <= 2 * win:
        return median_filter(v, size=win, mode="nearest")
    step = max(1, win // 50)
    n = len(v) // step
    blocks = np.median(v[: n * step].reshape(n, step), axis=1)
    bw = max(3, (win // step) | 1)
    coarse = median_filter(blocks, size=bw, mode="nearest")
    centres = (np.arange(n) + 0.5) * step - 0.5
    return np.interp(np.arange(len(v)), centres, coarse)


# --------------------------------------------------------------------------- #
# dimensionality reduction of wide inputs                                      #
# --------------------------------------------------------------------------- #

class InputReducer:
    """PCA compression of a wide model input X [C, T] -> [k, T] (z-scored).

    Raw mask pixels (64 x 64 = 4096 channels) or dense electrode arrays make
    every backbone slow on a CPU, and channel-token models (stt, gatr,
    patchtst) scale with channels x time^2. Randomised PCA keeps the dominant
    covariance structure at a fraction of the cost. ``n_components='auto'``
    keeps 64 components when C > 128 channels and does nothing otherwise."""

    AUTO_THRESHOLD = 128
    AUTO_COMPONENTS = 64

    def __init__(self, n_components="auto", seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self.pca = None
        self.out_mean = self.out_std = None

    def resolve(self, n_channels: int):
        """Number of components to keep for C channels (None = no reduction)."""
        k = self.n_components
        if k in (None, False, 0, "none"):
            return None
        if k == "auto":
            return self.AUTO_COMPONENTS if n_channels > self.AUTO_THRESHOLD else None
        return int(k) if int(k) < n_channels else None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        k = self.resolve(X.shape[0])
        if k is None:
            return X
        self.pca = PCA(k, svd_solver="randomized", random_state=self.seed)
        Xt = np.asarray(X, np.float32).T
        # fit then project (not fit_transform): randomized fit_transform returns
        # the approximate U*S, which differs from transform() on new data
        Y = self.pca.fit(Xt).transform(Xt).T
        self.out_mean = Y.mean(1, keepdims=True)
        self.out_std = Y.std(1, keepdims=True) + 1e-9
        return ((Y - self.out_mean) / self.out_std).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            return np.asarray(X, np.float32)
        Y = self.pca.transform(np.asarray(X, np.float32).T).T
        return ((Y - self.out_mean) / self.out_std).astype(np.float32)

    @property
    def explained_variance(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum()) if self.pca is not None else 1.0


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


DLC_PCUTOFF = 0.6        # DeepLabCut likelihood below which a keypoint is set to NaN
_TIME_KEYS = ("time", "times", "timestamp", "timestamps", "t", "frame_times")


def _fs_from_times(t):
    t = np.asarray(t, float)
    d = np.diff(t[np.isfinite(t)])
    return float(1.0 / np.median(d)) if d.size and np.median(d) > 0 else None


def load_dlc_csv(path: str, pcutoff: float = DLC_PCUTOFF):
    """Read a DeepLabCut CSV (3 header rows: scorer / bodyparts / coords).

    Returns (X [T, 2K] with x/y per body part, column names '<part>_x',
    '<part>_y'); points with likelihood < ``pcutoff`` become NaN (multi-animal
    files with an extra 'individuals' row are also handled)."""
    import pandas as pd
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        head = [fh.readline() for _ in range(4)]
    n_header = 4 if head[1].lower().startswith("individuals") else 3
    df = pd.read_csv(path, header=list(range(n_header)), index_col=0)
    cols, names = [], []
    levels = df.columns.nlevels
    parts = []
    for c in df.columns:
        key = tuple(c[1:levels - 1])                 # (bodypart,) or (individual, bodypart)
        if key not in parts:
            parts.append(key)
    for key in parts:
        sub = {c[-1]: df[c].to_numpy(float) for c in df.columns if tuple(c[1:levels - 1]) == key}
        if "x" not in sub or "y" not in sub:
            continue
        x, y = sub["x"].copy(), sub["y"].copy()
        if "likelihood" in sub:
            bad = sub["likelihood"] < pcutoff
            x[bad] = np.nan; y[bad] = np.nan
        base = "_".join(str(k) for k in key)
        cols += [x, y]; names += [f"{base}_x", f"{base}_y"]
    return np.stack(cols, axis=1), names


def load_auto(path: str):
    """Load any supported file and describe it for the advisor.

    Returns (data, hints) where hints = dict(kind, columns, times, fs, source):
    kind is 'spike_times' (list of arrays), 'coco' (path, raw masks),
    'pose' (DeepLabCut), 'table' (CSV with named columns, time along rows) or
    'matrix' (numeric array; orientation decided later by the advisor)."""
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    hints = dict(kind="matrix", columns=None, times=None, fs=None, source=path)
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            head = json.load(fh)
        if isinstance(head, dict) and "annotations" in head and "images" in head:
            hints["kind"] = "coco"
            return path, hints
        raise ValueError("JSON file is not a COCO annotation file")
    if ext in (".pkl", ".pickle"):
        return load_spikes(path), dict(hints, kind="spike_times")
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            return [np.asarray(a, float) for a in arr], dict(hints, kind="spike_times")
        return arr, hints
    if ext == ".npz":
        d = np.load(path, allow_pickle=True)
        if "spike_times" in d.files:
            return list(d["spike_times"]), dict(hints, kind="spike_times")
        for k in d.files:
            if k.lower() in _TIME_KEYS:
                hints["times"] = np.asarray(d[k], float)
                hints["fs"] = _fs_from_times(hints["times"])
        arrays = {k: d[k] for k in d.files if k.lower() not in _TIME_KEYS}
        obj = [k for k, v in arrays.items() if v.dtype == object]
        if obj:
            return [np.asarray(a, float) for a in arrays[obj[0]]], dict(hints, kind="spike_times")
        key = max(arrays, key=lambda k: arrays[k].size)            # largest array = data
        return np.asarray(arrays[key]), dict(hints, data_key=key)
    if ext == ".csv":
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            first = [fh.readline().lower() for _ in range(4)]
        if first[0].startswith("scorer") and any(l.startswith("coords") for l in first):
            X, names = load_dlc_csv(path)
            return X, dict(hints, kind="pose", columns=names)
        df = pd.read_csv(path)
        tcol = next((c for c in df.columns if str(c).strip().lower() in _TIME_KEYS), None)
        if tcol is not None:
            hints["times"] = df[tcol].to_numpy(float)
            hints["fs"] = _fs_from_times(hints["times"])
            df = df.drop(columns=[tcol])
        num = df.select_dtypes(include=[np.number])
        return num.to_numpy(float), dict(hints, kind="table", columns=[str(c) for c in num.columns])
    raise ValueError(f"unsupported file type {ext!r}")


def load_behavior(path: str):
    """Load behaviour variables for decoding / colouring.

    .csv  -> {column: values}            (header row required)
    .npz  -> {key: values}               (e.g. x, y, speed, head_direction)
    .npy  -> array [T] or [T, k]
    A column/key named ``time`` / ``times`` / ``timestamp`` is returned
    separately as the behaviour time base (for resampling).
    Returns (behavior, behavior_times | None)."""
    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        beh = {str(c): df[c].to_numpy() for c in df.columns}
    elif path.endswith(".npz"):
        d = np.load(path, allow_pickle=False)
        beh = {k: np.asarray(d[k]) for k in d.files}
    else:
        arr = np.load(path, allow_pickle=False)
        return arr, None
    t = None
    for key in ("time", "times", "timestamp", "timestamps", "t"):
        if key in beh:
            t = np.asarray(beh.pop(key), float)
            break
    return beh, t


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
