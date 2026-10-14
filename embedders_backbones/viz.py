"""Figures, animation and an interactive explorer for NEMBA embeddings.

Every figure is built on a bare ``matplotlib.figure.Figure`` (no pyplot global
state), so it can be drawn from a GUI worker thread and saved in several
formats at once (png / pdf / svg by default, see ``STYLE['formats']``).

Visual rules (kept consistent across all figures)
-------------------------------------------------
* categorical identity (states, backbones): fixed-order, colour-blind-checked
  palette ``CATEGORICAL`` and always a legend;
* magnitude / time: one-hue blue ramp (light -> dark), never a rainbow;
* angles (circular behaviour): cyclic ``twilight`` map;
* thin bars, hairline solid grids, text in neutral ink (never in series colour).
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .analysis import project, project_subset, state_statistics

# --------------------------------------------------------------------------- #
# user-configurable style (edit here)                                          #
# --------------------------------------------------------------------------- #
STYLE = dict(
    formats=("png", "pdf", "svg"),   # every saved figure is written in these formats
    dpi=300,                         # raster resolution for png
    font="DejaVu Sans",
    font_size=7.5,
    title_size=8.5,
    point_size=3.0,                  # scatter marker area (pt^2)
    point_alpha=0.65,
    line_width=1.0,
    bar_height=0.38,                 # thin bars (fraction of the category slot)
    ink="#0b0b0b", ink_2="#52514e", ink_3="#8a8984",
    grid="#e6e5e1", surface="#ffffff",
    gif_dpi=90, gif_frames=120, gif_tail=45, gif_fps=20,
)
# reference categorical palette (slot order is the colour-blind-safety mechanism)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# extra slots for K > 8 states (darker steps of the first hues; legend required)
CATEGORICAL_EXTRA = ["#184f95", "#a8461f", "#0f6e4c", "#9a6a00", "#a24b6e", "#2e2483"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("nemba_blue", BLUE_RAMP)
CYCLIC_CMAP = matplotlib.colormaps["twilight"]

RC = {
    "font.family": STYLE["font"], "font.size": STYLE["font_size"],
    "axes.titlesize": STYLE["title_size"], "axes.labelsize": STYLE["font_size"],
    "axes.edgecolor": STYLE["ink_3"], "axes.labelcolor": STYLE["ink_2"],
    "axes.linewidth": 0.6, "axes.titleweight": "regular", "axes.titlecolor": STYLE["ink"],
    "xtick.color": STYLE["ink_2"], "ytick.color": STYLE["ink_2"],
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "legend.fontsize": STYLE["font_size"] - 0.5,
    "figure.facecolor": STYLE["surface"], "axes.facecolor": STYLE["surface"],
    "savefig.facecolor": STYLE["surface"], "svg.fonttype": "none", "pdf.fonttype": 42,
}


def state_colors(K):
    """Fixed-order colours for K states (identity follows the state index)."""
    pal = CATEGORICAL + CATEGORICAL_EXTRA
    return [pal[i % len(pal)] for i in range(max(int(K), 1))]


def _fig(w, h):
    """New constrained-layout Figure with the NEMBA style applied."""
    with matplotlib.rc_context(RC):
        return Figure(figsize=(w, h), layout="constrained")


def _clean(ax, grid_axis=None, keep=("left", "bottom")):
    """Recessive axes: hide unused spines, optional hairline grid."""
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(sp in keep)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=STYLE["grid"], lw=0.5, ls="-")
        ax.set_axisbelow(True)


def save_figure(fig, path, formats=None):
    """Save ``fig`` as path.{png,pdf,svg} (extension of ``path`` is ignored).
    Returns the list of written files."""
    formats = tuple(formats or STYLE["formats"])
    stem, ext = os.path.splitext(str(path))
    if ext.lower() not in (".png", ".pdf", ".svg", ".jpg", ".tif", ""):
        stem = str(path)
    d = os.path.dirname(stem)
    if d:
        os.makedirs(d, exist_ok=True)
    out = []
    with matplotlib.rc_context(RC):
        for f in formats:
            p = f"{stem}.{f}"
            fig.savefig(p, dpi=STYLE["dpi"], bbox_inches="tight", pad_inches=0.03)
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# colouring                                                                    #
# --------------------------------------------------------------------------- #

def _color_spec(n, color, states=None, behavior=None, circular=False):
    """Resolve what the points are coloured by.

    Returns (values, cmap, norm, kind) with kind in {'time', 'state',
    'continuous', 'circular', 'discrete'}."""
    if color in (None, "time"):
        return np.arange(n), SEQ_CMAP, Normalize(0, max(n - 1, 1)), "time"
    if color == "state" and states is not None:
        s = np.asarray(states).astype(int)
        K = int(s.max()) + 1 if s.size else 1
        return s, ListedColormap(state_colors(K)), BoundaryNorm(np.arange(-0.5, K + 0.5), K), "state"
    arr = np.asarray(behavior if color == "behavior" else color)
    if circular:
        return arr, CYCLIC_CMAP, Normalize(-np.pi, np.pi), "circular"
    if arr.dtype.kind in "iub" or (arr.dtype.kind == "f" and np.unique(arr[np.isfinite(arr)]).size <= 12
                                   and np.allclose(arr[np.isfinite(arr)], np.round(arr[np.isfinite(arr)]))):
        lv, inv = np.unique(arr, return_inverse=True)
        K = len(lv)
        return inv, ListedColormap(state_colors(K)), BoundaryNorm(np.arange(-0.5, K + 0.5), K), "discrete"
    f = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
    lo, hi = (np.percentile(f, [1, 99]) if f.size else (0, 1))
    return arr, SEQ_CMAP, Normalize(lo, hi if hi > lo else lo + 1), "continuous"


def _legend_or_colorbar(fig, axes, vals, cmap, norm, kind, label, times=None, levels=None):
    """Attach the identity key: a legend for categories, a colorbar otherwise."""
    if kind in ("state", "discrete"):
        K = int(norm.boundaries[-1] + 0.5)
        names = [f"state {i}" for i in range(K)] if kind == "state" else \
            [str(l) for l in (levels if levels is not None else range(K))]
        handles = [Line2D([], [], marker="o", ls="", ms=4.5, mfc=cmap(i), mec="none")
                   for i in range(K)]
        fig.legend(handles, names, loc="outside right center", title=label,
                   title_fontsize=STYLE["font_size"], handletextpad=0.2, labelspacing=0.35)
        return
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, shrink=0.75, aspect=28, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.5)
    if kind == "time" and times is not None and len(times) > 1:
        n = len(times)
        ticks = np.linspace(0, n - 1, 4).astype(int)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{times[t]:.0f}" for t in ticks])
        cb.set_label("time (s)" if np.median(np.diff(times)) < 5 else "time")
    elif kind == "circular":
        cb.set_ticks([-np.pi, 0, np.pi]); cb.set_ticklabels(["-π", "0", "π"])
        cb.set_label(label)
    else:
        cb.set_label(label)


# --------------------------------------------------------------------------- #
# embedding scatter                                                            #
# --------------------------------------------------------------------------- #

def embedding_figure(Z, methods=("pca", "tsne", "umap"), color="time", states=None,
                     seed=0, title="embedding", fig=None, projections=None, times=None,
                     behavior=None, behavior_name=None, circular=False):
    """Side-by-side 2-D projections coloured by time, state or a behaviour.

    ``projections`` = {method: (P, frame_idx)} reuses cached projections; when
    missing, a strided subset is projected (fast on CPU)."""
    methods = tuple(methods)
    if fig is None:
        fig = _fig(2.35 * len(methods) + 0.9, 2.45)
    else:
        fig.clear()
        try:
            fig.set_layout_engine("constrained")
        except Exception:  # noqa: BLE001
            pass
    n = len(Z)
    vals, cmap, norm, kind = _color_spec(n, color, states, behavior, circular)
    levels = np.unique(np.asarray(behavior)) if kind == "discrete" else None
    with matplotlib.rc_context(RC):
        axes = np.atleast_1d(fig.subplots(1, len(methods)))
        for ax, m in zip(axes, methods):
            P, idx = (projections or {}).get(m) or project_subset(Z, m, seed)
            ax.scatter(P[:, 0], P[:, 1], c=np.asarray(vals)[idx], cmap=cmap, norm=norm,
                       s=STYLE["point_size"], alpha=STYLE["point_alpha"], linewidths=0,
                       rasterized=True)
            pretty = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}.get(m, m.upper())
            ax.set_title(f"{pretty}" + (f"  (n={len(idx):,} of {n:,} frames)" if len(idx) < n else ""),
                         fontsize=STYLE["font_size"])
            ax.set_xlabel(f"{pretty} 1"); ax.set_ylabel(f"{pretty} 2")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            _clean(ax)
        label = {"time": "time", "state": "state"}.get(kind, behavior_name or "behaviour")
        _legend_or_colorbar(fig, list(axes), vals, cmap, norm, kind, label, times, levels)
        fig.suptitle(title, fontsize=STYLE["title_size"], color=STYLE["ink"])
    return fig


def embedding_panel(Z, save_path, formats=None, **kw):
    """Compatibility wrapper: build ``embedding_figure`` and save it."""
    return save_figure(embedding_figure(Z, **kw), save_path, formats)


# --------------------------------------------------------------------------- #
# state rasters                                                                #
# --------------------------------------------------------------------------- #

def _raster(ax, s, t, K):
    cols = state_colors(K)
    ax.imshow(np.asarray(s)[None, :], aspect="auto", cmap=ListedColormap(cols),
              norm=BoundaryNorm(np.arange(-0.5, K + 0.5), K),
              extent=[t[0], t[-1], 0, 1], interpolation="nearest")
    ax.set_yticks([])
    _clean(ax, keep=("bottom",))


def _behaviour_track(ax, beh, t, name, circular=False):
    beh = np.asarray(beh)
    vals, cmap, norm, kind = _color_spec(len(beh), "behavior", behavior=beh, circular=circular)
    if kind == "discrete":
        K = int(norm.boundaries[-1] + 0.5)
        ax.imshow(vals[None, :], aspect="auto", cmap=cmap, norm=norm,
                  extent=[t[0], t[-1], 0, 1], interpolation="nearest")
        ax.set_yticks([])
        _clean(ax, keep=("bottom",))
    else:
        ax.plot(t, beh, color=STYLE["ink_2"], lw=0.7)
        ax.set_xlim(t[0], t[-1])
        _clean(ax, grid_axis="y")
    ax.set_ylabel(name, rotation=0, ha="right", va="center")


def state_sequence(states, save_path, times=None, behavior=None, title="state sequence",
                   formats=None, behavior_name="behaviour"):
    """State raster (+ optional behaviour track) with a state legend."""
    states = np.asarray(states).astype(int)
    t = np.arange(len(states)) if times is None else np.asarray(times)
    K = int(states.max()) + 1 if states.size else 1
    nrow = 1 if behavior is None else 2
    fig = _fig(7.2, 0.75 * nrow + 0.65)
    with matplotlib.rc_context(RC):
        axes = fig.subplots(nrow, 1, sharex=True, squeeze=False)[:, 0]
        _raster(axes[0], states, t, K)
        axes[0].set_ylabel("state", rotation=0, ha="right", va="center")
        if behavior is not None:
            _behaviour_track(axes[1], behavior, t, behavior_name)
        axes[-1].set_xlabel("time (s)" if times is not None else "frame")
        handles = [Patch(color=c, label=f"{i}") for i, c in enumerate(state_colors(K))]
        fig.legend(handles=handles, loc="outside right center", title="state",
                   title_fontsize=STYLE["font_size"], handlelength=1.0, labelspacing=0.3)
        fig.suptitle(title, fontsize=STYLE["title_size"])
    return save_figure(fig, save_path, formats)


def state_sequences_panel(seqs, save_path, times=None, behavior=None, behavior_name="behaviour",
                          title="state sequences", formats=None):
    """Time-aligned rasters of several state sequences (dict name -> states),
    optional behaviour track at the bottom: how do backbones segment the same
    recording?"""
    names = list(seqs)
    n = len(names)
    t = (np.arange(len(seqs[names[0]])) if times is None else np.asarray(times))
    nrow = n + (1 if behavior is not None else 0)
    fig = _fig(7.2, 0.38 * nrow + 0.75)
    with matplotlib.rc_context(RC):
        axes = fig.subplots(nrow, 1, sharex=True, squeeze=False)[:, 0]
        Kmax = 1
        for i, nm in enumerate(names):
            s = np.asarray(seqs[nm]).astype(int); K = int(s.max()) + 1 if s.size else 1
            Kmax = max(Kmax, K)
            _raster(axes[i], s, t, K)
            axes[i].set_ylabel(nm, rotation=0, ha="right", va="center")
            if i < nrow - 1:
                axes[i].tick_params(labelbottom=False, length=0)
        if behavior is not None:
            _behaviour_track(axes[n], behavior, t, behavior_name)
        axes[-1].set_xlabel("time (s)" if times is not None else "frame")
        handles = [Patch(color=c, label=f"{i}") for i, c in enumerate(state_colors(Kmax))]
        fig.legend(handles=handles, loc="outside right center", title="state",
                   title_fontsize=STYLE["font_size"], handlelength=1.0, labelspacing=0.3)
        fig.suptitle(title, fontsize=STYLE["title_size"])
    return save_figure(fig, save_path, formats)


# --------------------------------------------------------------------------- #
# comparison figures                                                           #
# --------------------------------------------------------------------------- #

def compare_embeddings_figure(embedders, method="umap", color="state", seed=0, fig=None,
                              title="embeddings compared"):
    """Grid of each backbone's 2-D embedding (projections cached per embedder)."""
    names = list(embedders)
    n = len(names); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    if fig is None:
        fig = _fig(2.1 * ncol + 0.8, 2.1 * nrow + 0.3)
    else:
        fig.clear()
        try:
            fig.set_layout_engine("constrained")
        except Exception:  # noqa: BLE001
            pass
    with matplotlib.rc_context(RC):
        axes = np.atleast_1d(fig.subplots(nrow, ncol, squeeze=False)).ravel()
        for ax in axes:
            ax.axis("off")
        Kmax = 1
        for i, nm in enumerate(names):
            emb = embedders[nm]; Z = emb.embedding(); st = emb.state_sequence()
            P, idx = emb.projection(method)
            vals, cmap, norm, kind = _color_spec(len(Z), color, st)
            if kind == "state":
                Kmax = max(Kmax, int(st.max()) + 1)
            ax = axes[i]; ax.axis("on")
            ax.scatter(P[:, 0], P[:, 1], c=np.asarray(vals)[idx], cmap=cmap, norm=norm,
                       s=STYLE["point_size"], alpha=STYLE["point_alpha"], linewidths=0,
                       rasterized=True)
            ax.set_title(nm, fontsize=STYLE["font_size"])
            ax.set_xticks([]); ax.set_yticks([]); _clean(ax)
        if color == "state":
            handles = [Line2D([], [], marker="o", ls="", ms=4.5, mfc=c, mec="none")
                       for c in state_colors(Kmax)]
            fig.legend(handles, [f"{k}" for k in range(Kmax)], loc="outside right center",
                       title="state", title_fontsize=STYLE["font_size"])
        else:
            sm = matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap=SEQ_CMAP)
            cb = fig.colorbar(sm, ax=list(axes), shrink=0.6, aspect=30, pad=0.01)
            cb.set_ticks([0, 1]); cb.set_ticklabels(["start", "end"]); cb.outline.set_visible(False)
        pretty = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}.get(method, method.upper())
        fig.suptitle(f"{title} ({pretty}, coloured by {color})", fontsize=STYLE["title_size"])
    return fig


def compare_embeddings_panel(embedders, save_path, formats=None, **kw):
    return save_figure(compare_embeddings_figure(embedders, **kw), save_path, formats)


def compare_bar(df, metric, save_path=None, title=None, formats=None, err_col=None, fig=None):
    """Thin horizontal bars of ``metric`` per backbone, sorted, value at the tip."""
    d = df.dropna(subset=[metric]).sort_values(metric)
    if fig is None:
        fig = _fig(3.6, 0.24 * len(d) + 0.8)
    with matplotlib.rc_context(RC):
        ax = fig.subplots()
        y = np.arange(len(d))
        xerr = d[err_col].fillna(0).to_numpy() if err_col and err_col in d else np.zeros(len(d))
        names = d["backbone"].astype(str).to_numpy()
        # baselines (no training / raw input) in neutral grey, models in the accent colour
        cols = [STYLE["ink_3"] if (n == "pca" or n.startswith("raw")) else CATEGORICAL[0]
                for n in names]
        ax.barh(y, d[metric].to_numpy(), height=STYLE["bar_height"], color=cols,
                xerr=xerr, error_kw=dict(lw=0.6, capsize=1.5, ecolor=STYLE["ink_2"]))
        ax.set_yticks(y); ax.set_yticklabels(names)
        span = float(np.nanmax(np.abs(d[metric].to_numpy()) + xerr)) if len(d) else 1.0
        for yi, v, e in zip(y, d[metric].to_numpy(), xerr):
            x = v + e if v >= 0 else v - e                 # label past the error bar
            ax.text(x + (0.02 if v >= 0 else -0.02) * span, yi, f"{v:.3f}", va="center",
                    ha="left" if v >= 0 else "right", color=STYLE["ink_2"],
                    fontsize=STYLE["font_size"] - 1)
        ax.set_xlim(right=ax.get_xlim()[1] + 0.12 * span)
        ax.axvline(0, color=STYLE["ink_3"], lw=0.6)
        ax.set_xlabel(metric.replace("_", " "))
        _clean(ax, grid_axis="x")
        ax.set_title(title or f"{metric.replace('_', ' ')} by backbone", loc="left")
    return save_figure(fig, save_path, formats) if save_path else fig


# --------------------------------------------------------------------------- #
# training, state dynamics, decoding                                           #
# --------------------------------------------------------------------------- #

def training_figure(history, title="training"):
    """Train / validation loss per evaluation step, best step marked."""
    fig = _fig(3.4, 2.2)
    with matplotlib.rc_context(RC):
        ax = fig.subplots()
        steps = np.asarray(history.get("step", []))
        if steps.size:
            ax.plot(steps, history["train_loss"], color=CATEGORICAL[0], lw=STYLE["line_width"],
                    label="train")
            v = np.asarray(history["val_loss"], float)
            if np.isfinite(v).any():
                ax.plot(steps, v, color=CATEGORICAL[1], lw=STYLE["line_width"], label="validation")
                b = history.get("best_step")
                if b:
                    ax.axvline(b, color=STYLE["ink_3"], lw=0.6)
                    ax.text(b, ax.get_ylim()[1], " best", va="top", ha="left",
                            color=STYLE["ink_2"], fontsize=STYLE["font_size"] - 1)
            ax.legend(loc="upper right")
        else:
            ax.text(0.5, 0.5, "closed-form fit: no training curve", ha="center", va="center",
                    transform=ax.transAxes, color=STYLE["ink_2"])
            ax.set_xticks([]); ax.set_yticks([]); _clean(ax, keep=())
        ax.set_xlabel("optimisation step"); ax.set_ylabel(f"{history.get('objective', '')} loss")
        _clean(ax, grid_axis="y")
        ax.set_title(f"{title}: {history.get('steps_done', 0)} steps, "
                     f"{history.get('seconds', 0):.1f} s", loc="left")
    return fig


def state_dynamics_figure(states, dt=1.0, fig=None, axes=None):
    """Occupancy, transition matrix and dwell-time distribution of the states."""
    st = state_statistics(states)
    K = len(st["occupancy"])
    cols = state_colors(K)
    own = axes is None
    if own:
        fig = _fig(7.0, 2.3)
    with matplotlib.rc_context(RC):
        if own:
            axes = fig.subplots(1, 3, width_ratios=[1, 1.15, 1.3])
        a0, a1, a2 = axes
        y = np.arange(K)
        a0.barh(y, st["occupancy"] * 100, height=STYLE["bar_height"], color=cols)
        a0.set_yticks(y); a0.set_yticklabels([f"{k}" for k in y]); a0.invert_yaxis()
        a0.set_xlabel("occupancy (%)"); a0.set_ylabel("state")
        _clean(a0, grid_axis="x"); a0.set_title("occupancy", loc="left")

        # off-diagonal transitions only: self-transitions (~1 at fine time bins)
        # would saturate the colour scale and hide the structure
        Tm = st["transitions"].copy()
        np.fill_diagonal(Tm, np.nan)
        rowsum = np.nansum(Tm, axis=1, keepdims=True)
        Tm = Tm / np.where(rowsum > 0, rowsum, 1.0)
        im = a1.imshow(Tm, cmap=SEQ_CMAP, vmin=0, vmax=max(0.5, float(np.nanmax(Tm)) if np.isfinite(Tm).any() else 1))
        tick = y if K <= 8 else y[::2]
        a1.set_xticks(tick); a1.set_yticks(tick)
        a1.tick_params(length=0)
        a1.set_xlabel("to state"); a1.set_ylabel("from state")
        if K <= 6:
            for i in range(K):
                for j in range(K):
                    v = Tm[i, j]
                    if np.isfinite(v) and v >= 0.05:
                        a1.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                                fontsize=STYLE["font_size"] - 2,
                                color="#ffffff" if v > 0.6 * im.norm.vmax else STYLE["ink"])
        for sp in a1.spines.values():
            sp.set_visible(False)
        a1.set_title("next state (leaving)", loc="left")
        cax = a1.inset_axes([1.04, 0.0, 0.05, 1.0])
        cb = a1.figure.colorbar(im, cax=cax)
        cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=STYLE["font_size"] - 1.5)

        data = [st["dwell"][k] * dt for k in range(K)]
        bp = a2.boxplot([d if d.size else [np.nan] for d in data], vert=False, widths=0.45,
                        patch_artist=True, showfliers=False, medianprops=dict(color=STYLE["ink"], lw=0.8),
                        whiskerprops=dict(lw=0.6, color=STYLE["ink_3"]),
                        capprops=dict(lw=0.6, color=STYLE["ink_3"]))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("none")
        a2.set_yticks(np.arange(1, K + 1)); a2.set_yticklabels([f"{k}" for k in range(K)])
        a2.invert_yaxis()
        a2.set_xscale("log")
        a2.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=4))
        a2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a2.set_xlabel("dwell time (s)" if dt != 1.0 else "dwell time (frames)")
        _clean(a2, grid_axis="x"); a2.set_title("dwell times", loc="left")
    return fig if own else axes


def _ptext(p):
    """Compact p-value label (Holm-corrected permutation p)."""
    if not np.isfinite(p):
        return ""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


METRIC_LABELS = {"r2": "R² (held-out)", "balanced_accuracy": "balanced accuracy",
                 "mean_cos_dtheta": "mean cos(Δθ)"}


# decoding feature sets: display order, label and fixed identity colour
FEATURE_ORDER = {"embedding": 0, "raw_pca": 1, "bayes_counts": 2, "psid": 3}
FEATURE_LABELS = {"embedding": "embedding", "raw_pca": "raw PCA",
                  "bayes_counts": "Bayes (counts)", "psid": "PSID"}
FEATURE_COLORS = {"embedding": CATEGORICAL[0], "raw_pca": STYLE["ink_3"],
                  "bayes_counts": CATEGORICAL[2], "psid": CATEGORICAL[6]}


def _wrap_breaks(t, y):
    """Insert NaN where an angle wraps around (no vertical jump lines)."""
    y = np.asarray(y, float).copy()
    jumps = np.flatnonzero(np.abs(np.diff(y)) > np.pi)
    return np.insert(np.asarray(t, float), jumps + 1, np.nan), np.insert(y, jumps + 1, np.nan)


def _mini_cbar(ax, cmap, norm, kind, n=None):
    """Small horizontal colour key inside the lower-left corner of ``ax``."""
    cax = ax.inset_axes([0.04, 0.03, 0.36, 0.035])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=STYLE["font_size"] - 1.5, pad=1)
    if kind == "time":
        cb.set_ticks([norm.vmin, norm.vmax]); cb.set_ticklabels(["start", "end"])
    elif kind == "circular":
        cb.set_ticks([-np.pi, np.pi]); cb.set_ticklabels(["-π", "π"])
    else:
        cb.set_ticks([norm.vmin, norm.vmax])
        cb.set_ticklabels([f"{norm.vmin:.3g}", f"{norm.vmax:.3g}"])
    return cb


def decoding_figure(result, times=None, max_targets=4, excerpt_frac=0.25):
    """Per target: true vs out-of-fold prediction (excerpt) and the score of the
    embedding vs the raw-input baseline against the circular-shift null."""
    tab = result.table
    targets = list(result.predictions)[:max_targets]
    n = max(len(targets), 1)
    fig = _fig(7.2, 1.55 * n + 0.35)
    with matplotlib.rc_context(RC):
        axes = fig.subplots(n, 2, width_ratios=[2.6, 1], squeeze=False)
        for i, name in enumerate(targets):
            pr = result.predictions[name]
            fr = pr["frames"]
            t = (np.asarray(times)[fr] if times is not None else fr).astype(float)
            m = max(10, int(len(fr) * excerpt_frac))
            a = axes[i, 0]
            if pr["task"] == "classification":
                a.step(t[:m], pr["y"][:m], where="mid", color=STYLE["ink_2"], lw=0.9, label="true")
                a.step(t[:m], pr["y_hat"][:m], where="mid", color=CATEGORICAL[1], lw=0.9,
                       alpha=0.9, label="decoded (held-out)")
                levels = np.unique(pr["y"])
                if len(levels) <= 8:
                    a.set_yticks(levels)
            elif pr["task"] == "circular":
                a.plot(*_wrap_breaks(t[:m], pr["y"][:m]), color=STYLE["ink_2"], lw=0.9, label="true")
                a.plot(*_wrap_breaks(t[:m], pr["y_hat"][:m]), color=CATEGORICAL[1], lw=0.9,
                       label="decoded (held-out)")
                a.set_yticks([-np.pi, 0, np.pi]); a.set_yticklabels(["-π", "0", "π"])
            else:
                a.plot(t[:m], pr["y"][:m], color=STYLE["ink_2"], lw=0.9, label="true")
                a.plot(t[:m], pr["y_hat"][:m], color=CATEGORICAL[1], lw=0.9,
                       label="decoded (held-out)")
            a.set_ylabel(name)
            _clean(a, grid_axis="y")
            if i == 0:
                a.set_title(f"held-out predictions, first {int(excerpt_frac * 100)}% of frames",
                            loc="left")
            if i == n - 1:
                a.set_xlabel("time (s)" if times is not None else "frame")

            b = axes[i, 1]
            rows = tab[tab.target == name].sort_values(
                "features", key=lambda s: s.map(FEATURE_ORDER).fillna(len(FEATURE_ORDER)))
            feats = list(rows["features"])
            y = np.arange(len(feats))
            sc = rows["score"].to_numpy()
            b.barh(y, sc, height=STYLE["bar_height"],
                   color=[FEATURE_COLORS.get(f, STYLE["ink_3"]) for f in feats])
            right = 0.0
            for yi, (_, r) in zip(y, rows.iterrows()):
                fs = json.loads(r["fold_scores"]) if isinstance(r["fold_scores"], str) else []
                b.plot(fs, np.full(len(fs), yi), "o", ms=2.2, color=STYLE["ink"], alpha=0.55,
                       mec="none")
                b.plot([r["null_p95"]] * 2, [yi - 0.3, yi + 0.3], color=CATEGORICAL[7], lw=1.2)
                x_lab = max([r["score"], r["null_p95"]] + list(fs))
                right = max(right, x_lab)
                b.text(x_lab, yi, f"  {r['score']:.2f}  {_ptext(r['p_holm'])}", va="center",
                       fontsize=STYLE["font_size"] - 1, color=STYLE["ink_2"])
            b.axvline(rows["chance"].iloc[0], color=STYLE["ink_3"], lw=0.6)
            b.set_yticks(y); b.set_yticklabels([FEATURE_LABELS.get(f, f) for f in feats])
            b.invert_yaxis()
            b.set_xlabel(METRIC_LABELS.get(rows["metric"].iloc[0], rows["metric"].iloc[0]))
            lo = min(0.0, float(np.nanmin(np.r_[sc, rows["null_p95"].to_numpy()])) - 0.05)
            b.set_xlim(lo, max(1.0, right) + 0.55)
            b.set_xticks([t_ for t_ in (0, 0.5, 1.0) if t_ >= lo])
            _clean(b, grid_axis="x")
            if i == 0:
                b.set_title("held-out score", loc="left")
        handles = [Line2D([], [], color=STYLE["ink_2"], lw=1.0, label="true"),
                   Line2D([], [], color=CATEGORICAL[1], lw=1.0, label="decoded"),
                   Line2D([], [], color=CATEGORICAL[7], lw=1.2, label="null 95th pct"),
                   Line2D([], [], color=STYLE["ink"], marker="o", ls="", ms=2.5, label="folds")]
        fig.legend(handles=handles, loc="outside lower center", ncols=4)
    return fig


# --------------------------------------------------------------------------- #
# one-page dashboard                                                           #
# --------------------------------------------------------------------------- #

def dashboard(emb, behavior=None, circular=(), method=None):
    """One-page summary of a fitted ``Embedder``: projections (time / state /
    behaviour), K selection, state raster aligned with behaviour, occupancy,
    transitions, dwell times and decoding scores."""
    from .backbones import label as _label
    cached = [k[0] for k in emb._proj]
    method = method or ("umap" if "umap" in cached else "pca")
    P, idx = emb.projection(method)
    Z, states, times = emb.Z, emb.states, emb.times
    beh_name, beh = (next(iter(behavior.items())) if behavior else (None, None))
    circ = beh_name in (circular or ())
    pretty = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}.get(method, method.upper())
    fig = _fig(7.4, 7.6)
    with matplotlib.rc_context(RC):
        root = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.62, 1.0])
        top = root[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.9])
        mid = root[1].subgridspec(2, 1, height_ratios=[0.42, 0.58], hspace=0.05)
        bot = root[2].subgridspec(1, 4, width_ratios=[0.75, 1.05, 1.0, 1.1])

        # --- row 1: projections + K selection --------------------------------
        panels = [("time", "time"), ("state", "state")]
        if beh is not None:
            panels.append(("behavior", beh_name))
        for j, (col, lab) in enumerate(panels):
            ax = fig.add_subplot(top[0, j])
            v, cmap, norm, kind = _color_spec(len(Z), col, states, beh, circ)
            ax.scatter(P[:, 0], P[:, 1], c=np.asarray(v)[idx], cmap=cmap, norm=norm,
                       s=STYLE["point_size"] * 0.7, alpha=STYLE["point_alpha"], linewidths=0,
                       rasterized=True)
            ax.set_xticks([]); ax.set_yticks([]); _clean(ax, keep=())
            ax.set_aspect("equal", adjustable="box")
            ax.margins(0.04)
            ax.set_title(f"{pretty} by {lab}", loc="left")
            if kind in ("state", "discrete"):
                K_ = int(norm.boundaries[-1] + 0.5)
                names = [str(k) for k in range(K_)] if kind == "state" else \
                    [str(l) for l in np.unique(np.asarray(beh))]
                if K_ <= 10:
                    hs = [Line2D([], [], marker="o", ls="", ms=3.5, mfc=cmap(k), mec="none")
                          for k in range(K_)]
                    ax.legend(hs, names, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                              ncols=min(K_, 5), handletextpad=0.05, columnspacing=0.6,
                              fontsize=STYLE["font_size"] - 1.5, borderaxespad=0.1)
            else:
                _mini_cbar(ax, cmap, norm, kind)
        if beh is None:                                   # keep the grid balanced
            ax = fig.add_subplot(top[0, 2])
            sp = np.linalg.norm(np.diff(Z, axis=0, prepend=Z[:1]), axis=1)
            nrm = Normalize(*np.percentile(sp, [1, 99]))
            ax.scatter(P[:, 0], P[:, 1], c=sp[idx], cmap=SEQ_CMAP, norm=nrm,
                       s=STYLE["point_size"] * 0.7, alpha=STYLE["point_alpha"], linewidths=0,
                       rasterized=True)
            ax.set_xticks([]); ax.set_yticks([]); _clean(ax, keep=())
            ax.set_aspect("equal", adjustable="box"); ax.margins(0.04)
            ax.set_title(f"{pretty} by latent speed", loc="left")
            _mini_cbar(ax, SEQ_CMAP, nrm, "continuous")
        axk = fig.add_subplot(top[0, 3])
        if emb.silhouette_scores:
            ks = list(emb.silhouette_scores); sv = list(emb.silhouette_scores.values())
            axk.plot(ks, sv, "-o", color=CATEGORICAL[0], lw=STYLE["line_width"], ms=3.5,
                     mec=STYLE["surface"], mew=1.0)
            axk.plot([emb.n_states_used], [emb.silhouette_scores[emb.n_states_used]], "o",
                     ms=7, mfc="none", mec=STYLE["ink"], mew=0.8)
            axk.set_xticks(ks[::2] if len(ks) > 6 else ks)
            axk.set_xlabel("K (states)"); axk.set_ylabel("silhouette")
            axk.set_box_aspect(1)                   # square, like the projections
            axk.set_title(f"auto K = {emb.n_states_used}", loc="left")
            _clean(axk, grid_axis="y")
        else:
            axk.text(0.5, 0.5, f"K = {emb.n_states_used}\n(user-set)", ha="center", va="center",
                     transform=axk.transAxes, color=STYLE["ink_2"])
            axk.set_xticks([]); axk.set_yticks([]); _clean(axk, keep=())

        # --- row 2: raster + behaviour ---------------------------------------
        K = int(states.max()) + 1
        ar = fig.add_subplot(mid[0])
        _raster(ar, states, times, K)
        ar.set_ylabel("state", rotation=0, ha="right", va="center")
        ar.tick_params(labelbottom=False, length=0)
        ab = fig.add_subplot(mid[1], sharex=ar)
        if beh is not None:
            if circ:
                ab.plot(*_wrap_breaks(times, beh), color=STYLE["ink_2"], lw=0.6)
                ab.set_yticks([-np.pi, np.pi]); ab.set_yticklabels(["-π", "π"])
                _clean(ab, grid_axis="y")
                ab.set_ylabel(beh_name, rotation=0, ha="right", va="center")
            else:
                _behaviour_track(ab, beh, times, beh_name)
        else:
            ab.plot(times, np.linalg.norm(np.diff(Z, axis=0, prepend=Z[:1]), axis=1),
                    color=STYLE["ink_2"], lw=0.5)
            ab.set_ylabel("latent\nspeed", rotation=0, ha="right", va="center"); _clean(ab)
        ab.set_xlim(times[0], times[-1])
        ab.set_xlabel("time (s)")

        # --- row 3: dynamics + decoding --------------------------------------
        a_occ, a_tr, a_dw = (fig.add_subplot(bot[0, 0]), fig.add_subplot(bot[0, 1]),
                             fig.add_subplot(bot[0, 2]))
        state_dynamics_figure(states, emb.frame_dt(), fig=fig, axes=(a_occ, a_tr, a_dw))
        ad = fig.add_subplot(bot[0, 3])
        if emb.decoding is not None and len(emb.decoding.table):
            t = emb.decoding.table
            te = t[t.features == "embedding"].reset_index(drop=True)
            y = np.arange(len(te))
            h = 0.36
            ad.barh(y - h / 2, te["score"], height=h * 0.9, color=CATEGORICAL[0], label="embedding")
            if (t.features == "raw_pca").any():
                tr = t[t.features == "raw_pca"].set_index("target").reindex(te["target"])
                ad.barh(y + h / 2, tr["score"].to_numpy(), height=h * 0.9, color=STYLE["ink_3"],
                        label="raw PCA")
            for yi, (_, r) in zip(y, te.iterrows()):
                ad.plot([r["null_p95"]] * 2, [yi - 0.42, yi + 0.42], color=CATEGORICAL[7], lw=1.1)
            ad.set_yticks(y); ad.set_yticklabels(te["target"]); ad.invert_yaxis()
            lo = min(0.0, float(np.nanmin(t["null_p95"])) - 0.05)
            ad.set_xlim(lo, 1.02); ad.set_xticks([0, 0.5, 1])
            hd, lb = ad.get_legend_handles_labels()
            hd.append(Line2D([], [], color=CATEGORICAL[7], lw=1.1)); lb.append("null 95%")
            ad.legend(hd, lb, loc="upper left", bbox_to_anchor=(0.0, -0.12), ncols=2,
                      fontsize=STYLE["font_size"] - 1.5, handlelength=1.0, columnspacing=0.8)
            ad.set_title("decoding", loc="left")
            _clean(ad, grid_axis="x")
        else:
            h = emb.history or {}
            ad.plot(h.get("step", []), h.get("train_loss", []), color=CATEGORICAL[0], lw=1,
                    label="train")
            v = np.asarray(h.get("val_loss", []), float)
            if v.size and np.isfinite(v).any():
                ad.plot(h["step"], v, color=CATEGORICAL[1], lw=1, label="validation")
                ad.legend(loc="upper right", fontsize=STYLE["font_size"] - 1.5)
            ad.set_xlabel("step"); ad.set_title("training loss", loc="left")
            _clean(ad, grid_axis="y")
        fig.suptitle(f"{_label(emb.backbone)}  |  {emb.X_input.shape[0]} channels x "
                     f"{len(Z):,} frames  |  K={emb.n_states_used}  |  "
                     f"fit {emb.timings.get('total_s', 0):.1f} s ({emb.speed})",
                     fontsize=STYLE["title_size"])
    return fig


# --------------------------------------------------------------------------- #
# trajectory GIF (rendered frame by frame on Agg: thread-safe, no pyplot)     #
# --------------------------------------------------------------------------- #

def trajectory_gif(Z, save_path, method="umap", seed=0, n_frames=None, tail=None, fps=None,
                   title="trajectory", projection=None, states=None):
    """Animated trail through the 2-D projection (strided subset of frames)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from PIL import Image
    n_frames = n_frames or STYLE["gif_frames"]; tail = tail or STYLE["gif_tail"]
    fps = fps or STYLE["gif_fps"]
    P, idx = projection if projection is not None else project_subset(Z, method, seed)
    with matplotlib.rc_context(RC):
        fig = Figure(figsize=(3.4, 3.4))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.88])
        pad = 0.05 * (P.max(0) - P.min(0) + 1e-9)
        ax.set_xlim(P[:, 0].min() - pad[0], P[:, 0].max() + pad[0])
        ax.set_ylim(P[:, 1].min() - pad[1], P[:, 1].max() + pad[1])
        bg = ("#e3e2de" if states is None else
              [matplotlib.colors.to_rgba(c, 0.25) for c in np.asarray(state_colors(int(np.max(states)) + 1))[np.asarray(states)[idx]]])
        ax.scatter(P[:, 0], P[:, 1], s=1.5, c=bg, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([]); _clean(ax, keep=())
        ax.set_title(f"{title} ({method.upper()})", fontsize=STYLE["title_size"])
        lc = LineCollection([], cmap=SEQ_CMAP, linewidths=1.6, capstyle="round")
        ax.add_collection(lc)
        head = ax.scatter([], [], s=28, c=CATEGORICAL[7], zorder=5, edgecolor=STYLE["surface"], lw=1.0)
        T = len(P); step = max(T // n_frames, 1); ends = list(range(2, T, step))
        images = []
        for e in ends:
            s0 = max(0, e - tail); seg = P[s0:e].reshape(-1, 1, 2)
            lc.set_segments(np.concatenate([seg[:-1], seg[1:]], axis=1))
            lc.set_array(np.linspace(0, 1, max(len(seg) - 1, 1)))
            head.set_offsets(P[e - 1])
            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())
            images.append(Image.fromarray(buf[..., :3]).quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    if images:
        images[0].save(save_path, save_all=True, append_images=images[1:],
                       duration=int(1000 / fps), loop=0, optimize=False)
    return save_path


# --------------------------------------------------------------------------- #
# interactive HTML explorer (self-contained, no internet, no extra deps)      #
# --------------------------------------------------------------------------- #

_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEMBA explorer</title>
<style>
:root{--surface:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8984;--grid:#e6e5e1;--accent:#2a78d6;--hot:#e34948}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--panel:#222220;--ink:#ffffff;--ink2:#c3c2b7;--ink3:#8f8e86;--grid:#383835;--accent:#3987e5;--hot:#e66767}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#222220;--ink:#ffffff;--ink2:#c3c2b7;--ink3:#8f8e86;--grid:#383835;--accent:#3987e5;--hot:#e66767}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:14px 16px 6px}h1{font-size:16px;font-weight:600;margin:0}header p{margin:2px 0 0;color:var(--ink2)}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 16px}
.bar label{color:var(--ink2)}select,button{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--grid);border-radius:6px;padding:4px 8px}
button{cursor:pointer;min-width:64px}button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
button[hidden]{display:none}input[type=range]{flex:1;min-width:160px;accent-color:var(--accent)}
main{display:grid;grid-template-columns:minmax(0,1fr) 220px;gap:12px;padding:0 16px 16px}
@media (max-width:760px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:8px;position:relative}
canvas{width:100%;display:block;touch-action:none;outline:none;border-radius:6px}canvas:focus-visible{box-shadow:0 0 0 2px var(--accent)}
#tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--grid);border-radius:6px;padding:4px 7px;font-size:12px;display:none;white-space:nowrap}
#hint{margin:4px 2px 0;color:var(--ink3);font-size:12px}#hint[hidden]{display:none}
#legend div{display:flex;align-items:center;gap:6px;margin:2px 0;color:var(--ink2)}#legend i{width:10px;height:10px;border-radius:50%;display:inline-block}
#stats{color:var(--ink2);font-size:12px;margin-top:8px}#stats b{color:var(--ink);font-weight:600}
</style></head><body>
<header><h1 id="ttl"></h1><p id="sub"></p></header>
<div class="bar"><label for="col">Colour by</label><select id="col"></select>
<button id="view3d" aria-pressed="false" title="3-D view: drag to rotate, wheel to zoom, double-click to reset">3D</button>
<button id="play" aria-label="Play trajectory">Play</button>
<input id="slider" type="range" min="0" value="0" aria-label="Time"><span id="tlab"></span>
<button id="theme" aria-label="Toggle theme">Theme</button></div>
<main><div class="card"><canvas id="cv" height="560" tabindex="0" aria-label="Embedding map"></canvas><div id="tip"></div>
<p id="hint" hidden>drag to rotate &middot; wheel to zoom &middot; double-click or 0 to reset &middot; arrow keys rotate</p></div>
<div class="card"><div id="legend"></div><div id="stats"></div></div></main>
<script>
const D = __DATA__;
const CAT = ["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#4a3aa7","#e34948","#184f95","#a8461f","#0f6e4c","#9a6a00","#a24b6e","#2e2483"];
const RAMP = ["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"];
const $ = id => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d"), tip = $("tip");
const sel = $("col"), slider = $("slider"), b3 = $("view3d"), hint = $("hint");
$("ttl").textContent = D.title;
const modes = [["time","time"],["state","cluster state"]].concat(Object.keys(D.behavior).map(k=>["b:"+k,k]));
for (const [v,t] of modes){const o=document.createElement("option");o.value=v;o.textContent=t;sel.appendChild(o);}
// Two views with the same fields: the 2-D map (V2) and the optional 3-D projection (V3).
// When the 3-D subset differs from the 2-D one it carries its own time / state / behaviour.
function bvals(src){const r={};for(const k of Object.keys(D.behavior))r[k]=src?src[k]:D.behavior[k].v;return r;}
const V2={x:D.x,y:D.y,z:null,t:D.t,state:D.state,b:bvals(null),sub:D.subtitle};
const P3=D.p3||null;
const V3=P3?{x:P3.x,y:P3.y,z:P3.z,t:P3.same?D.t:P3.t,state:P3.same?D.state:P3.state,
 b:P3.same?V2.b:bvals(P3.beh),sub:P3.subtitle}:null;
let V=V2;
if(!V3)b3.hidden=true;
// 3-D camera: yaw / pitch in radians, zoom factor; the cloud is centred and scaled to ~unit radius
const CAM0={yaw:-0.6,pitch:0.35,zoom:1},DIST=3.2;let cam=Object.assign({},CAM0);
let C3=[0,0,0],R3=1;
if(V3){const n=V3.x.length;for(let i=0;i<n;i++){C3[0]+=V3.x[i];C3[1]+=V3.y[i];C3[2]+=V3.z[i];}
 C3=C3.map(c=>c/Math.max(1,n));const r=new Float64Array(n);
 for(let i=0;i<n;i++){const a=V3.x[i]-C3[0],b=V3.y[i]-C3[1],c=V3.z[i]-C3[2];r[i]=Math.sqrt(a*a+b*b+c*c);}
 r.sort();R3=Math.max(r[Math.min(n-1,Math.floor(0.98*n))]||0,1e-9);}   // robust radius: 98th percentile
const clampP=p=>Math.max(-1.55,Math.min(1.55,p)), clampZ=z=>Math.max(0.3,Math.min(8,z));
function lerp(a,b,t){return a+(b-a)*t}
function hex(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)]}
function ramp(t){t=Math.max(0,Math.min(1,t));const p=t*(RAMP.length-1),i=Math.min(Math.floor(p),RAMP.length-2),f=p-i;const a=hex(RAMP[i]),b=hex(RAMP[i+1]);return `rgb(${lerp(a[0],b[0],f)|0},${lerp(a[1],b[1],f)|0},${lerp(a[2],b[2],f)|0})`}
function cyc(v){const t=(v+Math.PI)/(2*Math.PI);const h=Math.round(t*360);return `hsl(${h},55%,55%)`}
function colorOf(i){const m=sel.value,n=V.x.length;
 if(m==="time") return ramp(i/Math.max(1,n-1));
 if(m==="state") return CAT[V.state[i]%CAT.length];
 const k=m.slice(2),b=D.behavior[k],v=V.b[k][i];
 if(b.kind==="discrete") return v>=0?CAT[v%CAT.length]:"#9a9993";
 if(b.kind==="circular") return cyc(v);
 return ramp((v-b.lo)/(b.hi-b.lo||1));}
let W=0,H=0,sx,sy;
function fit(){const r=cv.getBoundingClientRect();const dpr=window.devicePixelRatio||1;W=r.width;H=Math.min(560,Math.max(320,r.width*0.7));cv.style.height=H+"px";cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
 const xs=V2.x,ys=V2.y;let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;for(let i=0;i<xs.length;i++){x0=Math.min(x0,xs[i]);x1=Math.max(x1,xs[i]);y0=Math.min(y0,ys[i]);y1=Math.max(y1,ys[i]);}
 const pad=16,s=Math.min((W-2*pad)/(x1-x0||1),(H-2*pad)/(y1-y0||1));const cx=(x0+x1)/2,cy=(y0+y1)/2;sx=x=>W/2+(x-cx)*s;sy=y=>H/2-(y-cy)*s;}
// screen position (PX, PY), perspective factor (PF), depth (PZ) and painter's order (ORD) of the active view
let PX=new Float32Array(0),PY=PX,PF=PX,PZ=PX,ORD=new Uint32Array(0);
function rot(x,y,z){const cy=Math.cos(cam.yaw),sy_=Math.sin(cam.yaw),cp=Math.cos(cam.pitch),sp=Math.sin(cam.pitch);
 const x1=cy*x+sy_*z,z1=-sy_*x+cy*z;return [x1,cp*y-sp*z1,sp*y+cp*z1];}   // yaw about the vertical axis, then pitch
function project(){const n=V.x.length;
 if(PX.length!==n){PX=new Float32Array(n);PY=new Float32Array(n);PF=new Float32Array(n);PZ=new Float32Array(n);ORD=new Uint32Array(n);for(let i=0;i<n;i++)ORD[i]=i;}
 if(!V.z){for(let i=0;i<n;i++){PX[i]=sx(V.x[i]);PY[i]=sy(V.y[i]);PF[i]=1;PZ[i]=0;}return;}
 const cy=Math.cos(cam.yaw),sy_=Math.sin(cam.yaw),cp=Math.cos(cam.pitch),sp=Math.sin(cam.pitch);
 const s=Math.min(W,H)*0.42*cam.zoom;
 for(let i=0;i<n;i++){const x=(V.x[i]-C3[0])/R3,y=(V.y[i]-C3[1])/R3,z=(V.z[i]-C3[2])/R3;
  const x1=cy*x+sy_*z,z1=-sy_*x+cy*z,y2=cp*y-sp*z1,z2=Math.min(sp*y+cp*z1,DIST-0.6);
  const f=DIST/(DIST-z2);                       // simple perspective: nearer points spread and grow
  PX[i]=W/2+x1*s*f;PY[i]=H/2-y2*s*f;PF[i]=f;PZ[i]=z2;}
 ORD.sort((a,b)=>PZ[a]-PZ[b]);}                  // painter's algorithm: far points first
function gizmo(col){const L=18,ox=26,oy=H-24;ctx.save();ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1;
 ctx.font="10px system-ui,sans-serif";ctx.textAlign="center";ctx.textBaseline="middle";
 [[1,0,0],[0,1,0],[0,0,1]].forEach((a,k)=>{const r=rot(a[0],a[1],a[2]);ctx.beginPath();ctx.moveTo(ox,oy);
  ctx.lineTo(ox+L*r[0],oy-L*r[1]);ctx.stroke();ctx.fillText(String(k+1),ox+(L+7)*r[0],oy-(L+7)*r[1]);});ctx.restore();}
function draw(){ctx.clearRect(0,0,W,H);project();const n=V.x.length,cur=Math.min(+slider.value,n-1);
 const cs=getComputedStyle(document.documentElement),ink=cs.getPropertyValue("--ink"),hot=cs.getPropertyValue("--hot"),panel=cs.getPropertyValue("--panel");
 ctx.globalAlpha=0.55;
 if(V.z){for(let k=0;k<n;k++){const i=ORD[k],r=1.3*PF[i];ctx.fillStyle=colorOf(i);ctx.fillRect(PX[i]-r,PY[i]-r,2*r,2*r);}}
 else{for(let i=0;i<n;i++){ctx.fillStyle=colorOf(i);ctx.fillRect(PX[i]-1.3,PY[i]-1.3,2.6,2.6);}}
 ctx.globalAlpha=1;const tail=Math.max(5,Math.round(n/60));ctx.lineWidth=2;ctx.lineCap="round";ctx.lineJoin="round";ctx.strokeStyle=ink;
 for(let i=Math.max(1,cur-tail);i<=cur;i++){ctx.globalAlpha=(i-(cur-tail))/tail;ctx.beginPath();ctx.moveTo(PX[i-1],PY[i-1]);ctx.lineTo(PX[i],PY[i]);ctx.stroke();}
 ctx.globalAlpha=1;ctx.fillStyle=hot;ctx.strokeStyle=panel;ctx.lineWidth=2;
 ctx.beginPath();ctx.arc(PX[cur],PY[cur],5,0,2*Math.PI);ctx.fill();ctx.stroke();
 if(V.z)gizmo(cs.getPropertyValue("--ink3"));
 $("tlab").textContent=`t = ${V.t[cur].toFixed(1)} s  |  state ${V.state[cur]}`;legend();}
function legend(){const L=$("legend");L.innerHTML="";const m=sel.value;
 const add=(c,txt)=>{const d=document.createElement("div");const i=document.createElement("i");i.style.background=c;d.appendChild(i);d.appendChild(document.createTextNode(txt));L.appendChild(d);};
 if(m==="state"){for(let k=0;k<D.K;k++)add(CAT[k%CAT.length],`state ${k}  (${(100*D.occ[k]).toFixed(1)}%)`);}
 else if(m==="time"){add(RAMP[0],"start");add(RAMP[RAMP.length-1],"end");}
 else{const b=D.behavior[m.slice(2)];if(b.kind==="discrete"){b.levels.forEach((l,k)=>add(CAT[k%CAT.length],String(l)));}
  else if(b.kind==="circular"){add(cyc(-Math.PI),"-π");add(cyc(0),"0");}else{add(RAMP[0],b.lo.toPrecision(3));add(RAMP[RAMP.length-1],b.hi.toPrecision(3));}}
 $("stats").innerHTML=D.stats;}
// shareable views: explorer.html#color=state&t=0.45&view=3d (t = fraction of the recording);
// written back on deliberate changes only (never during playback: browsers throttle replaceState)
function share(){try{if(typeof history==="undefined"||!history.replaceState)return;const q=new URLSearchParams();
 q.set("color",sel.value);q.set("t",((+slider.value)/Math.max(1,V.x.length-1)).toFixed(3));if(V===V3)q.set("view","3d");
 history.replaceState(null,"","#"+q.toString());}catch(_){}}
function setView(want3,frac,quiet){const v3=!!(want3&&V3);
 if(frac===undefined)frac=(+slider.value)/Math.max(1,V.x.length-1);
 V=v3?V3:V2;slider.max=V.x.length-1;slider.value=Math.round(Math.max(0,Math.min(1,frac))*(V.x.length-1));
 b3.setAttribute("aria-pressed",v3?"true":"false");hint.hidden=!v3;cv.style.cursor=v3?"grab":"default";
 $("sub").textContent=V.sub;tip.style.display="none";fit();draw();if(!quiet)share();}
let playing=null;$("play").onclick=()=>{const btn=$("play");if(playing){clearInterval(playing);playing=null;btn.textContent="Play";share();return;}
 btn.textContent="Pause";playing=setInterval(()=>{const n=V.x.length;slider.value=(+slider.value+Math.max(1,Math.round(n/600)))%n;draw();},30);};
slider.oninput=draw;slider.onchange=share;sel.onchange=()=>{draw();share();};window.onresize=()=>{fit();draw();};
b3.onclick=()=>setView(V!==V3);
$("theme").onclick=()=>{const r=document.documentElement;const dark=r.dataset.theme?r.dataset.theme==="dark":matchMedia("(prefers-color-scheme: dark)").matches;r.dataset.theme=dark?"light":"dark";draw();};
function hover(e){const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=144,front=-1,fz=-Infinity;
 for(let i=0;i<PX.length;i++){const dx=PX[i]-mx,dy=PY[i]-my,d=dx*dx+dy*dy;if(d<bd){bd=d;best=i;}
  if(V.z&&d<16&&PZ[i]>fz){fz=PZ[i];front=i;}}           // 3-D: the front-most point under the cursor wins
 if(front>=0)best=front;
 if(best<0){tip.style.display="none";return;}let s=`t = ${V.t[best].toFixed(2)} s<br>state ${V.state[best]}`;
 for(const k of Object.keys(D.behavior)){const b=D.behavior[k],v=V.b[k][best];
  s+=`<br>${k}: ${b.kind==="discrete"?((v>=0&&v<b.levels.length)?b.levels[v]:"?"):(+v).toPrecision(3)}`;}
 tip.innerHTML=s;tip.style.display="block";tip.style.left=(mx+14)+"px";tip.style.top=(my+8)+"px";}
// pointer: hover in both views, drag rotates the 3-D view
let drag=null;
cv.onpointerdown=e=>{if(!V.z)return;drag={x:e.clientX,y:e.clientY};tip.style.display="none";cv.style.cursor="grabbing";
 try{if(e.pointerId!==undefined&&cv.setPointerCapture)cv.setPointerCapture(e.pointerId);}catch(_){}};
cv.onpointermove=e=>{if(drag){cam.yaw+=(e.clientX-drag.x)*0.01;cam.pitch=clampP(cam.pitch+(e.clientY-drag.y)*0.01);
  drag={x:e.clientX,y:e.clientY};draw();return;}hover(e);};
cv.onpointerup=cv.onpointercancel=()=>{if(drag){drag=null;cv.style.cursor=V.z?"grab":"default";}};
cv.onpointerleave=()=>{if(!drag)tip.style.display="none";};
cv.ondblclick=()=>{if(V.z){cam=Object.assign({},CAM0);draw();}};
cv.addEventListener("wheel",e=>{if(!V.z)return;e.preventDefault();cam.zoom=clampZ(cam.zoom*Math.exp(-e.deltaY*0.0015));draw();},{passive:false});
cv.onkeydown=e=>{if(!V.z)return;const k=e.key;let used=true;
 if(k==="ArrowLeft")cam.yaw-=0.08;else if(k==="ArrowRight")cam.yaw+=0.08;
 else if(k==="ArrowUp")cam.pitch=clampP(cam.pitch-0.08);else if(k==="ArrowDown")cam.pitch=clampP(cam.pitch+0.08);
 else if(k==="+"||k==="=")cam.zoom=clampZ(cam.zoom*1.15);else if(k==="-")cam.zoom=clampZ(cam.zoom/1.15);
 else if(k==="0")cam=Object.assign({},CAM0);else used=false;
 if(used){e.preventDefault();draw();}};
let start3=false,startT;
if(typeof location!=="undefined"&&location.hash){const Q=new URLSearchParams(location.hash.slice(1));
 if(Q.get("color")){const c=Q.get("color"),v=modes.some(m=>m[0]===c)?c:("b:"+c);if(modes.some(m=>m[0]===v))sel.value=v;}
 start3=Q.get("view")==="3d";
 if(Q.get("t")!==null&&isFinite(+Q.get("t")))startT=Math.max(0,Math.min(1,+Q.get("t")));}
fit();setView(start3,startT===undefined?0:startT,true);
</script></body></html>
"""


def interactive_html(emb, save_path, behavior=None, method="umap", max_behaviors=6, circular=(),
                     view3d=True, max_points_3d=6000):
    """Write a self-contained interactive explorer (hover, colour switch,
    trajectory playback, light/dark, optional 3-D view) for a fitted ``Embedder``.

    The 2-D map uses ``emb.projection(method)``. With ``view3d=True`` a
    3-component projection is added (``manifold.project3d`` with the same
    method when it is 'pca', 'umap' or 'tphate', otherwise PCA) and a "3D"
    toggle appears: drag to rotate, wheel to zoom, points depth-sorted. The
    3-D view reuses the 2-D frame subset when the two coincide (same strided
    frames); otherwise it carries its own time, state and behaviour values.
    ``max_points_3d`` caps the 3-D subset (each redraw sorts every point).
    URL-hash views: ``#color=state&t=0.45&view=3d``."""
    from .backbones import label as _label
    P, idx = emb.projection(method)
    st = state_statistics(emb.states)
    beh = {}
    specs = {}                                   # name -> (kind, levels, full series) for the 3-D subset
    for k, v_full in list((behavior or {}).items())[:max_behaviors]:
        v_full = np.asarray(v_full)
        v = v_full[idx]
        vals, _, norm, kind = _color_spec(len(v), "behavior", behavior=v, circular=k in circular)
        if kind == "circular":
            beh[k] = dict(kind="circular", v=np.nan_to_num(v.astype(float)).round(4).tolist())
            specs[k] = ("circular", None, v_full)
        elif kind == "discrete":
            levels = np.unique(v)
            beh[k] = dict(kind="discrete", v=vals.astype(int).tolist(),
                          levels=[str(x) for x in levels])
            specs[k] = ("discrete", levels, v_full)
        else:
            fin = np.where(np.isfinite(v.astype(float)), v.astype(float), np.nan)
            beh[k] = dict(kind="continuous", v=np.nan_to_num(fin).round(4).tolist(),
                          lo=float(norm.vmin), hi=float(norm.vmax))
            specs[k] = ("continuous", None, v_full)
    dec = ""
    if emb.decoding is not None:
        te = emb.decoding.table[emb.decoding.table.features == "embedding"]
        dec = "<br>".join(f"{r.target}: <b>{r.score:.3f}</b> ({r.metric}, p={r.p_value:.3g})"
                          for r in te.itertuples())
    m = emb.metrics() if emb.history else {}
    stats = (f"K = <b>{emb.n_states_used}</b><br>silhouette <b>{m.get('silhouette', float('nan')):.3f}</b>"
             f"<br>circularity <b>{m.get('circularity', float('nan')):.3f}</b>"
             f"<br>participation ratio <b>{m.get('participation_ratio', float('nan')):.2f}</b>"
             + (f"<br><br>decoding<br>{dec}" if dec else ""))
    pretty = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE", "tphate": "T-PHATE"}

    def _view3d():
        """3-D payload of the explorer, or None when it cannot be computed.

        The subset size requested from ``project3d`` equals the 2-D one (capped
        at ``max_points_3d``), so both use the same strided frames whenever
        ``analysis.subsample_index`` gives the same stride; when they differ
        the 3-D view gets its own time / state / behaviour arrays, mapped onto
        the 2-D colour scales (discrete levels, continuous range)."""
        from .manifold import project3d
        # view3d may name the 3-D method ("pca" is instant; "umap" / "tphate" cost
        # seconds); True reuses the 2-D method
        m3 = view3d if isinstance(view3d, str) else (
            method if method in ("pca", "umap", "tphate") else "pca")
        try:
            P3, idx3 = project3d(emb.Z, m3, max_points=int(min(len(idx), max_points_3d)),
                                 seed=emb.seed)
            P3, idx3 = np.asarray(P3, float), np.asarray(idx3)
        except Exception:  # noqa: BLE001  (e.g. a 2-D embedding: no third axis)
            return None
        if P3.ndim != 2 or P3.shape[1] < 3 or not np.isfinite(P3).all():
            return None
        out = dict(x=P3[:, 0].round(4).tolist(), y=P3[:, 1].round(4).tolist(),
                   z=P3[:, 2].round(4).tolist(), same=bool(np.array_equal(idx3, idx)),
                   subtitle=f"{pretty.get(m3, m3.upper())} 3-D of {len(idx3):,} / {len(emb.Z):,} "
                            f"frames, {emb.X_input.shape[0]} input channels, drag to rotate")
        if not out["same"]:
            out["t"] = np.asarray(emb.times)[idx3].round(3).tolist()
            out["state"] = np.asarray(emb.states)[idx3].astype(int).tolist()
            b3 = {}
            for k, (kind, levels, v_full) in specs.items():
                v3 = v_full[idx3]
                if kind == "discrete":
                    # index into the 2-D levels; frames with an unseen level get -1 (grey)
                    pos = np.clip(np.searchsorted(levels, v3), 0, len(levels) - 1)
                    b3[k] = np.where(levels[pos] == v3, pos, -1).astype(int).tolist()
                else:
                    b3[k] = np.nan_to_num(v3.astype(float)).round(4).tolist()
            out["beh"] = b3
        return out

    data = dict(title=f"{_label(emb.backbone)} embedding explorer",
                subtitle=f"{pretty.get(method, method.upper())} of {len(idx):,} / {len(emb.Z):,} "
                         f"frames, {emb.X_input.shape[0]} input channels, hover a point for details",
                x=P[:, 0].round(4).tolist(), y=P[:, 1].round(4).tolist(),
                t=np.asarray(emb.times)[idx].round(3).tolist(),
                state=np.asarray(emb.states)[idx].astype(int).tolist(),
                K=int(len(st["occupancy"])), occ=st["occupancy"].round(4).tolist(),
                behavior=beh, stats=stats, p3=_view3d() if view3d else None)
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write(_HTML.replace("__DATA__", json.dumps(data)))
    return save_path


# keep the v0.1 name importable
def _colors(n, color, states):
    v, cmap, norm, kind = _color_spec(n, color, states)
    return v, cmap
