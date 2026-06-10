"""Figures + animation: embedding panels (PCA/t-SNE/UMAP), state sequence raster,
trajectory GIF. Headless (Agg) by default; ``embedding_figure`` returns a Figure
so a GUI can display it."""

from __future__ import annotations

import numpy as np
import matplotlib
if matplotlib.get_backend().lower() not in ("tkagg", "qtagg", "qt5agg"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from .analysis import project


def _colors(n, color, states):
    if color is None or color == "time":
        return np.arange(n), "viridis"
    if color == "state" and states is not None:
        return np.asarray(states), "tab10"
    arr = np.asarray(color)
    return arr, ("tab10" if arr.dtype.kind in "iu" else "viridis")


def embedding_figure(Z, methods=("pca", "tsne", "umap"), color="time", states=None,
                     seed=0, title="embedding", fig=None):
    """Return a Figure with side-by-side 2-D projections (for GUI or saving)."""
    c, cmap = _colors(len(Z), color, states)
    if fig is None:
        fig = Figure(figsize=(4.6 * len(methods), 4.4))
    axes = fig.subplots(1, len(methods)) if len(methods) > 1 else [fig.subplots(1, 1)]
    for ax, m in zip(np.atleast_1d(axes), methods):
        P = project(Z, m, seed)
        ax.scatter(P[:, 0], P[:, 1], c=c, cmap=cmap, s=4, alpha=0.7, linewidths=0)
        ax.set_title(m.upper(), fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def embedding_panel(Z, save_path, **kw):
    fig = embedding_figure(Z, **kw)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def state_sequence(states, save_path, times=None, behavior=None, title="state sequence"):
    states = np.asarray(states)
    t = np.arange(len(states)) if times is None else np.asarray(times)
    K = int(states.max()) + 1 if states.size else 1
    cmap = ListedColormap([plt.get_cmap("tab10")(i % 10) for i in range(max(K, 1))])
    norm = BoundaryNorm(np.arange(-0.5, K + 0.5, 1), cmap.N)
    nrow = 1 if behavior is None else 2
    fig, axes = plt.subplots(nrow, 1, figsize=(13, 1.4 * nrow + 0.6), squeeze=False)
    ax = axes[0][0]
    ax.imshow(states[None, :], aspect="auto", cmap=cmap, norm=norm,
              extent=[t[0], t[-1], 0, 1], interpolation="nearest")
    ax.set_yticks([]); ax.set_ylabel("state", rotation=0, ha="right", va="center")
    if behavior is None:
        ax.set_xlabel("time")
    else:
        ax.set_xticks([])
        axb = axes[1][0]; beh = np.asarray(behavior)
        if beh.dtype.kind in "iu" and beh.max() > 1:
            bK = int(beh.max()) + 1
            bcm = ListedColormap([plt.get_cmap("tab20")(i % 20) for i in range(bK)])
            axb.imshow(beh[None, :], aspect="auto", cmap=bcm,
                       norm=BoundaryNorm(np.arange(-0.5, bK + 0.5, 1), bcm.N),
                       extent=[t[0], t[-1], 0, 1], interpolation="nearest")
        else:
            axb.fill_between(t, 0, beh, step="mid", color="#c1121f", lw=0)
            axb.set_xlim(t[0], t[-1]); axb.set_ylim(0, 1)
        axb.set_yticks([]); axb.set_ylabel("behaviour", rotation=0, ha="right", va="center")
        axb.set_xlabel("time")
        for sp in ("top", "right", "left"):
            axb.spines[sp].set_visible(False)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return save_path


def trajectory_gif(Z, save_path, method="umap", seed=0, n_frames=120, tail=45, fps=20,
                   title="trajectory"):
    P = project(Z, method, seed)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.set_xlim(P[:, 0].min() - 0.5, P[:, 0].max() + 0.5)
    ax.set_ylim(P[:, 1].min() - 0.5, P[:, 1].max() + 0.5)
    ax.scatter(P[:, 0], P[:, 1], s=2, c="#e0e0e0", alpha=0.5, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{title} ({method.upper()})")
    lc = LineCollection([], cmap="viridis", linewidths=2); ax.add_collection(lc)
    head = ax.scatter([], [], s=45, c="#c1121f", zorder=5, edgecolor="k", lw=0.5)
    T = len(P); step = max(T // n_frames, 1); ends = list(range(2, T, step))

    def update(i):
        e = ends[i]; s = max(0, e - tail); seg = P[s:e]; pts = seg.reshape(-1, 1, 2)
        lc.set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1))
        lc.set_array(np.linspace(0, 1, max(len(seg) - 1, 1))); head.set_offsets(P[e - 1])
        return lc, head
    FuncAnimation(fig, update, frames=len(ends), blit=False).save(
        save_path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return save_path
