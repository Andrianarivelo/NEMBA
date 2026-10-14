"""Figures for latent dynamics, topology, state validation and population rasters.

Every function returns a bare ``matplotlib.figure.Figure`` (no pyplot global
state, safe in GUI worker threads); save with ``viz.save_figure(fig, path)``.
The visual language is the one of ``viz``: fixed-order categorical palette
for identities (states, homology dimensions) with a legend, one-hue blue ramp
for magnitudes, a blue / grey / red diverging map only for signed
similarities, thin marks, hairline grids, text in neutral ink.

Figures
-------
persistence_figure    barcode + persistence diagram (ring / torus evidence)
flow_field_figure     mean latent velocity per cell of a 2-D projection
recurrence_figure     similarity of the latent state across time + lag profile
contingency_figure    discovered states vs annotated labels (ARI, NMI, ...)
consistency_figure    pairwise linear consistency (R^2) of embeddings
rastermap_figure      sorted population raster with state and behaviour tracks
"""

from __future__ import annotations

import warnings

import numpy as np
import matplotlib
import matplotlib.ticker  # noqa: F401  (log-axis formatter)
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
from scipy.spatial.distance import cdist

from . import manifold as _mf
from .analysis import subsample_index
from .viz import (BLUE_RAMP, CATEGORICAL, RC, STYLE, _behaviour_track, _clean, _fig,
                  _raster, _wrap_breaks, state_colors)

# --------------------------------------------------------------------------- #
# user-configurable style (edit here)                                          #
# --------------------------------------------------------------------------- #
DYN_STYLE = dict(
    raster_vmin=0.0, raster_vmax=2.0,     # colour range of z-scored activity
    raster_max_rows=1000,                 # more neurons -> averaged into super-neurons
    barcode_max_bars=(30, 30, 15),        # bars shown for H0, H1, H2 (longest first)
    cloud_color="#c9c8c3", cloud_alpha=0.35,   # faint point cloud under the flow field
    annotate_max_cells=100,               # write values in heatmap cells up to this many
    null_color=CATEGORICAL[7],            # null references (same red as decoding_figure)
)
# magnitude heatmaps start at the surface colour so that 0 reads as "nothing"
HEAT_CMAP = LinearSegmentedColormap.from_list("nemba_heat", ["#ffffff"] + BLUE_RAMP)
# arrows need contrast on white: darker half of the blue ramp
ARROW_CMAP = LinearSegmentedColormap.from_list("nemba_arrow", BLUE_RAMP[2:])
# signed similarity: blue (anti-similar) / neutral grey / red (similar)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "nemba_div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3aeab", "#e34948", "#a3302f"])
DIM_COLORS = CATEGORICAL[:3]              # H0, H1, H2 in fixed slot order


def _message_figure(text, w=4.0, h=1.6, title=None):
    """Small figure carrying only an explanatory message (missing dependency)."""
    fig = _fig(w, h)
    with matplotlib.rc_context(RC):
        ax = fig.subplots()
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
                color=STYLE["ink_2"], wrap=True)
        ax.set_xticks([]); ax.set_yticks([]); _clean(ax, keep=())
        if title:
            ax.set_title(title, loc="left")
    return fig


def _heat_text_color(v, vmax):
    """White text on dark cells, neutral ink elsewhere."""
    return "#ffffff" if v > 0.6 * vmax else STYLE["ink"]


def _time_axis(times, idx):
    """Time stamps of the strided frames and the matching axis label."""
    if times is not None:
        return np.asarray(times, float)[idx], "time (s)"
    return np.asarray(idx, float), "frame"


# --------------------------------------------------------------------------- #
# 1. topology                                                                  #
# --------------------------------------------------------------------------- #

def persistence_figure(Z, max_points=600, maxdim=1, seed=0, null=True, title="persistent homology"):
    """Barcode and persistence diagram of the latent cloud (ripser).

    All filtration values are divided by the diameter of the subsampled cloud
    (``manifold.persistence``, same normalisation as
    ``analysis.circularity``), so bars read as "fraction of the cloud size".
    One long H1 bar = ring (Chaudhuri et al., Nat Neurosci 2019, head
    direction); two long H1 bars + one H2 bar = torus (Gardner et al., Nature
    2022, grid cells). ``null=True`` also computes the longest H1 bar of a
    Gaussian cloud with the same mean and covariance and draws it as a band
    above the diagonal: features below it are indistinguishable from an
    unstructured cloud of the same shape. Long bars can also come from
    sampling gaps (unvisited parts of a manifold), so read them together with
    the trajectory."""
    res = _mf.persistence(Z, max_points=max_points, maxdim=maxdim, seed=seed)
    if not res["available"] or res["dgms"] is None:
        return _message_figure("ripser is not installed:\npip install ripser\n"
                               "(persistent homology unavailable)", title=title)
    diam = res["diameter"]
    dgms = [d / (diam + 1e-12) for d in res["dgms"]]          # normalised filtration
    null_h1 = float("nan")
    if null:
        Zs = np.asarray(Z, float)
        rng = np.random.default_rng(seed + 1)
        G = rng.multivariate_normal(Zs.mean(0), np.cov(Zs.T).reshape(Zs.shape[1], -1),
                                    size=min(len(Zs), max_points), method="eigh")
        null_h1 = _mf.persistence(G, max_points=max_points, maxdim=1, seed=seed)["h1_norm"]
    finite = np.concatenate([d[np.isfinite(d[:, 1]), 1] for d in dgms if len(d)])
    top = max(float(finite.max()) if finite.size else 0.1, 0.05) * 1.08
    fig = _fig(6.6, 2.75)
    with matplotlib.rc_context(RC):
        a_bar, a_dg = fig.subplots(1, 2, width_ratios=[1.3, 1])

        # --- barcode: longest bars of each dimension, H0 on top -------------
        y, yt, ytl = 0.0, [], []
        best = None                                   # (y, birth, death) of top H1 bar
        for dim, d in enumerate(dgms):
            if not len(d):
                continue
            life = np.where(np.isfinite(d[:, 1]), d[:, 1] - d[:, 0], np.inf)
            order = np.argsort(-life)[: DYN_STYLE["barcode_max_bars"][min(dim, 2)]]
            y0 = y
            for r, k in enumerate(order):
                b, e = d[k]
                inf = not np.isfinite(e)
                e = top if inf else e
                lw = 2.0 if (dim == 1 and r == 0) else 1.1
                a_bar.plot([b, e], [y, y], color=DIM_COLORS[dim % 3], lw=lw,
                           solid_capstyle="butt")
                if inf:                               # essential class: arrow head
                    a_bar.plot([e], [y], marker=">", ms=3.5, color=DIM_COLORS[dim % 3])
                if dim == 1 and r == 0:
                    best = (y, b, e)
                y += 1
            yt.append(0.5 * (y0 + y - 1))
            ytl.append(f"H{dim}\n{len(order)} of {len(d)}")
            y += 2.5                                  # gap between dimensions
        if best is not None:                          # length of the top H1 bar, at its end
            yb, b, e = best
            a_bar.text(e + 0.012 * top, yb, f"{e - b:.2f}", ha="left", va="center",
                       color=STYLE["ink"], fontsize=STYLE["font_size"] - 1)
        a_bar.set_yticks(yt); a_bar.set_yticklabels(ytl)
        a_bar.tick_params(axis="y", length=0)
        a_bar.set_ylim(y - 2.0, -2.0)                 # inverted: H0 first
        a_bar.set_xlim(0, top * 1.13)                 # room for the bar-length label
        a_bar.set_xlabel("filtration radius / diameter")
        _clean(a_bar, grid_axis="x", keep=("bottom",))
        a_bar.set_title("barcode (longest bars)", loc="left")

        # --- persistence diagram -------------------------------------------
        a_dg.plot([0, top], [0, top], color=STYLE["ink_3"], lw=0.6)
        if np.isfinite(null_h1):
            a_dg.fill_between([0, top], [0, top], [null_h1, top + null_h1],
                              color=DYN_STYLE["null_color"], alpha=0.10, lw=0)
            a_dg.plot([0, top], [null_h1, top + null_h1], color=DYN_STYLE["null_color"],
                      lw=0.8, ls="--")
        handles = []
        for dim, d in enumerate(dgms):
            if not len(d):
                continue
            fin = np.isfinite(d[:, 1])
            a_dg.scatter(d[fin, 0], d[fin, 1], s=9, color=DIM_COLORS[dim % 3], alpha=0.85,
                         linewidths=0, zorder=3)
            if (~fin).any():                          # essential classes on the top edge
                a_dg.scatter(d[~fin, 0], np.full((~fin).sum(), top), s=14, marker="^",
                             color=DIM_COLORS[dim % 3], linewidths=0, zorder=3, clip_on=False)
            handles.append(Line2D([], [], marker="o", ls="", ms=4, mfc=DIM_COLORS[dim % 3],
                                  mec="none", label=f"H{dim}"))
        if best is not None:
            a_dg.plot([best[1]], [best[2]], "o", ms=8, mfc="none", mec=STYLE["ink"], mew=0.8,
                      zorder=4)
        if np.isfinite(null_h1):
            handles.append(Line2D([], [], color=DYN_STYLE["null_color"], lw=0.8, ls="--",
                                  label="Gaussian null"))
        a_dg.set_xlim(0, top); a_dg.set_ylim(0, top * 1.04)
        a_dg.set_box_aspect(1)
        a_dg.set_xlabel("birth / diameter"); a_dg.set_ylabel("death / diameter")
        a_dg.legend(handles=handles, loc="lower right", handletextpad=0.3, borderaxespad=0.2)
        _clean(a_dg)
        a_dg.set_title("diagram (▲ = never dies)", loc="left")
        sub = (f"H1 max {res['h1_norm']:.2f}, 2nd {res['h1_second']:.2f}"
               + (f", H2 max {res['h2_norm']:.2f}" if maxdim >= 2 else "")
               + (f" | Gaussian null {null_h1:.2f}" if np.isfinite(null_h1) else "")
               + f" | n = {res['n_points']} frames")
        fig.suptitle(f"{title}: {sub}", fontsize=STYLE["title_size"])
    fig.nemba_stats = dict(res, null_h1=null_h1)
    return fig


# --------------------------------------------------------------------------- #
# 2. flow field                                                                #
# --------------------------------------------------------------------------- #

def flow_field(P, idx, dt=1.0, grid=20, min_count=5):
    """Mean velocity of a 2-D projected trajectory per grid cell.

    P [n, 2] are projected frames ``idx`` (strided: consecutive entries
    equally spaced in time). Velocity between consecutive subset frames =
    displacement / (dt * frame gap), assigned to the cell of the segment
    midpoint. Square cells, ``grid`` along the longer axis.

    Returns dict(cx, cy, vx, vy, speed, count, coherence, cell, stride) for
    cells with >= ``min_count`` segments. coherence = |mean v| / mean |v|
    (1: every passage moves the same way, ~0: crossing or diffusive motion;
    biased upward in sparse cells, ~1/sqrt(count) for random directions)."""
    P = np.asarray(P, float)[:, :2]
    idx = np.asarray(idx)
    gap = np.diff(idx).astype(float)
    stride = float(np.median(gap)) if gap.size else 1.0
    ok = gap == stride                                 # skip irregular jumps, if any
    V = np.diff(P, axis=0) / (dt * gap[:, None])
    M = 0.5 * (P[:-1] + P[1:])
    V, M = V[ok], M[ok]
    lo, hi = P.min(0), P.max(0)
    cell = float((hi - lo).max() / max(int(grid), 1)) or 1.0
    nx, ny = (np.maximum(np.ceil((hi - lo) / cell), 1)).astype(int)
    ix = np.clip(((M[:, 0] - lo[0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((M[:, 1] - lo[1]) / cell).astype(int), 0, ny - 1)
    flat = ix * ny + iy
    nc = nx * ny
    cnt = np.bincount(flat, minlength=nc)
    sp = np.linalg.norm(V, axis=1)
    sx = np.bincount(flat, V[:, 0], nc); sy = np.bincount(flat, V[:, 1], nc)
    ss = np.bincount(flat, sp, nc)
    show = cnt >= min_count
    c = np.maximum(cnt, 1)
    vx, vy, spd = sx / c, sy / c, ss / c
    gx, gy = np.divmod(np.arange(nc), ny)
    coh = np.hypot(vx, vy) / np.maximum(spd, 1e-12)
    return dict(cx=lo[0] + (gx[show] + 0.5) * cell, cy=lo[1] + (gy[show] + 0.5) * cell,
                vx=vx[show], vy=vy[show], speed=spd[show], count=cnt[show],
                coherence=coh[show], cell=cell, stride=stride)


def flow_field_figure(P, idx, dt=1.0, grid=20, min_count=5, states=None, title="latent flow",
                      axis_label="dim"):
    """Average displacement per grid cell of a 2-D projection, drawn as
    arrows over the faint point cloud; arrow colour = mean speed, arrow
    length = |mean velocity| (short arrows in fast cells = incoherent flow).

    ``states`` (per frame of the full recording, or per projected frame)
    colours the background cloud. Flow in PCA space is a linear view of the
    true latent velocity; in t-SNE / UMAP space distances are non-metric, so
    read directions, not speeds. Large strides between projected frames
    alias fast dynamics (check ``stride`` in the title)."""
    P = np.asarray(P, float)
    idx = np.asarray(idx)
    ff = flow_field(P, idx, dt, grid, min_count)
    unit = "s" if dt != 1.0 else "frame"
    fig = _fig(3.9, 3.3)
    with matplotlib.rc_context(RC):
        ax = fig.subplots()
        if states is not None:
            st = np.asarray(states).astype(int)
            st = st if len(st) == len(P) else st[idx]
            K = int(st.max()) + 1 if st.size else 1
            cols = np.asarray(state_colors(K))[st]
            ax.scatter(P[:, 0], P[:, 1], s=1.6, c=cols, alpha=0.22, linewidths=0, rasterized=True)
            hs = [Line2D([], [], marker="o", ls="", ms=4, mfc=c, mec="none")
                  for c in state_colors(K)]
            fig.legend(hs, [str(k) for k in range(K)], loc="outside right upper", title="state",
                       title_fontsize=STYLE["font_size"], handletextpad=0.2, labelspacing=0.3)
        else:
            ax.scatter(P[:, 0], P[:, 1], s=1.6, c=DYN_STYLE["cloud_color"],
                       alpha=DYN_STYLE["cloud_alpha"], linewidths=0, rasterized=True)
        if len(ff["cx"]):
            mag = np.hypot(ff["vx"], ff["vy"])
            # longest mean-velocity arrow spans ~90% of a cell
            scale = float(mag.max()) / (0.9 * ff["cell"]) if mag.max() > 0 else 1.0
            nrm = Normalize(*np.percentile(ff["speed"], [2, 98]))
            q = ax.quiver(ff["cx"], ff["cy"], ff["vx"], ff["vy"], ff["speed"], cmap=ARROW_CMAP,
                          norm=nrm, angles="xy", scale_units="xy", scale=scale, pivot="mid",
                          width=0.0055, headwidth=3.6, headlength=4.0, headaxislength=3.4,
                          zorder=3)
            cb = fig.colorbar(q, ax=ax, shrink=0.6, aspect=22, pad=0.02, location="bottom")
            cb.outline.set_visible(False); cb.ax.tick_params(length=2, width=0.5)
            cb.set_label(f"mean speed (units / {unit})")
            coh = float(np.average(ff["coherence"], weights=ff["count"]))
        else:
            coh = float("nan")
            ax.text(0.5, 0.5, f"no cell with >= {min_count} samples", ha="center",
                    va="center", transform=ax.transAxes, color=STYLE["ink_2"])
        ax.set_aspect("equal", adjustable="box")      # square cells stay square
        ax.margins(0.03)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"{axis_label} 1"); ax.set_ylabel(f"{axis_label} 2")
        _clean(ax)
        st_ = ff["stride"]
        ax.set_title(f"{title}\n{len(ff['cx'])} cells (>= {min_count} samples), stride "
                     f"{st_:g} frame{'' if st_ == 1 else 's'}, coherence {coh:.2f}", loc="left",
                     fontsize=STYLE["font_size"])
    fig.nemba_stats = dict(coherence=coh, n_cells=int(len(ff["cx"])), stride=ff["stride"])
    return fig


# --------------------------------------------------------------------------- #
# 3. recurrence                                                                #
# --------------------------------------------------------------------------- #

def recurrence_figure(Z, times=None, max_points=1500, metric="cosine", title="recurrence"):
    """Similarity of the latent state between every pair of (strided) frames,
    plus its mean as a function of lag.

    Bright off-diagonal stripes = the trajectory comes back to a previously
    visited state (recurrence; Marwan et al., Phys Rep 2007); a checkerboard
    = discrete states; a broad diagonal band = slow drift. ``metric`` is any
    scipy ``cdist`` metric: 'cosine' and 'correlation' are shown as signed
    similarity (1 - distance, diverging map), others as distance.

    The lag panel (log axis) marks the first lag where the mean similarity
    falls below 1/e of its lag-0 value; its resolution is the stride of the
    subset (``fig.nemba_stats['tau_1e']``)."""
    Z = np.asarray(Z, float)
    idx = subsample_index(len(Z), max_points)
    X = Z[idx]
    D = cdist(X, X, metric=metric)
    signed = metric in ("cosine", "correlation")
    S = 1.0 - D if signed else D
    t, tlab = _time_axis(times, idx)
    n = len(idx)
    # mean of each diagonal (lag in strided steps) via one weighted bincount
    i, j = np.triu_indices(n)
    lag_mean = np.bincount(j - i, S[i, j], n) / np.bincount(j - i, minlength=n)
    nl = max(2, n // 2)
    step = float(np.median(np.diff(t))) if n > 1 else 1.0
    lags = np.arange(nl) * step
    fig = _fig(6.6, 3.0)
    with matplotlib.rc_context(RC):
        a_m, a_l = fig.subplots(1, 2, width_ratios=[1.25, 1])
        if signed:
            im = a_m.imshow(S, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, origin="lower",
                            extent=[t[0], t[-1], t[0], t[-1]], interpolation="nearest")
        else:
            im = a_m.imshow(S, cmap=HEAT_CMAP.reversed(), vmin=0,
                            vmax=float(np.percentile(S, 99)), origin="lower",
                            extent=[t[0], t[-1], t[0], t[-1]], interpolation="nearest")
        im.set_rasterized(True)
        a_m.set_xlabel(tlab); a_m.set_ylabel(tlab)
        for sp in a_m.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=a_m, shrink=0.8, aspect=24, pad=0.02)
        cb.outline.set_visible(False); cb.ax.tick_params(length=2, width=0.5)
        cb.set_label(f"{metric} {'similarity' if signed else 'distance'}")
        a_m.set_title(f"{n:,} of {len(Z):,} frames", loc="left", fontsize=STYLE["font_size"])

        # log lag axis: the fast decorrelation and slow returns are both visible
        a_l.plot(lags[1:], lag_mean[1:nl], color=CATEGORICAL[0], lw=STYLE["line_width"])
        a_l.axhline(0, color=STYLE["ink_3"], lw=0.6)
        a_l.set_xscale("log")
        a_l.set_xlim(lags[1], lags[-1])
        a_l.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a_l.set_xlabel(f"lag ({'s' if times is not None else 'frames'}, log scale)")
        a_l.set_ylabel(f"mean {'similarity' if signed else 'distance'}")
        _clean(a_l, grid_axis="y")
        a_l.set_box_aspect(0.8)
        tau = float("nan")
        if signed and lag_mean[0] > 0:
            below = np.flatnonzero(lag_mean[:nl] < lag_mean[0] / np.e)
            if below.size and below[0] > 0:
                tau = float(lags[below[0]])
                a_l.axvline(tau, color=STYLE["ink_3"], lw=0.6, ls="--")
                a_l.text(tau, a_l.get_ylim()[1], f" 1/e at {tau:.3g}", va="top", ha="left",
                         color=STYLE["ink_2"], fontsize=STYLE["font_size"] - 1)
        a_l.set_title("mean over pairs at each lag", loc="left", fontsize=STYLE["font_size"])
        fig.suptitle(f"{title} ({metric})", fontsize=STYLE["title_size"])
    fig.nemba_stats = dict(tau_1e=tau, lag=lags, lag_profile=lag_mean[:nl])
    return fig


# --------------------------------------------------------------------------- #
# 4. states vs annotated labels                                                #
# --------------------------------------------------------------------------- #

def contingency_table(states, labels, n_perm=200, seed=0):
    """Cross-tabulation of discovered states vs annotated labels.

    Frames with a missing (NaN) label are dropped. States are reordered by
    Hungarian matching (``scipy.optimize.linear_sum_assignment`` on the
    counts), unmatched states follow, grouped by their dominant label.
    Scores (sklearn, labels = ground truth): ARI (chance-corrected
    agreement), NMI, homogeneity (each state holds one label) and
    completeness (each label sits in one state). Because both sequences are
    autocorrelated, the ARI p-value uses a circular-shift null (shifts of
    10-90% of the recording), not a frame shuffle.

    Returns dict(table: row-normalised DataFrame [state x label] in display
    order, counts: DataFrame, order, ari, ari_p, ari_null95, nmi,
    homogeneity, completeness)."""
    import pandas as pd
    from scipy.optimize import linear_sum_assignment
    from sklearn import metrics as skm
    s = np.asarray(states).astype(int)
    lab = np.asarray(labels)
    if len(s) != len(lab):
        raise ValueError(f"states ({len(s)}) and labels ({len(lab)}) lengths differ")
    if lab.dtype.kind == "f":
        ok = np.isfinite(lab)
        s, lab = s[ok], lab[ok]
    sv, si = np.unique(s, return_inverse=True)
    lv, li = np.unique(lab, return_inverse=True)
    C = np.zeros((len(sv), len(lv)))
    np.add.at(C, (si, li), 1.0)
    r, c = linear_sum_assignment(-C)
    match = dict(zip(r, c))
    order = sorted(match, key=lambda k: match[k])
    rest = [k for k in range(len(sv)) if k not in match]
    order += sorted(rest, key=lambda k: (int(np.argmax(C[k])), -C[k].sum()))
    R = C / np.maximum(C.sum(1, keepdims=True), 1)
    ari = float(skm.adjusted_rand_score(li, si))
    null = np.array([])
    if n_perm and len(s) > 20:
        rng = np.random.default_rng(seed)
        shifts = rng.integers(int(0.1 * len(s)), int(0.9 * len(s)), size=int(n_perm))
        null = np.array([skm.adjusted_rand_score(li, np.roll(si, k)) for k in shifts])
    p = float((1 + np.sum(null >= ari)) / (1 + len(null))) if null.size else float("nan")
    names = [str(v) for v in lv]
    rows = [str(sv[k]) for k in order]
    return dict(table=pd.DataFrame(R[order], index=rows, columns=names),
                counts=pd.DataFrame(C[order].astype(int), index=rows, columns=names),
                order=np.asarray([sv[k] for k in order]), ari=ari, ari_p=p,
                ari_null95=float(np.percentile(null, 95)) if null.size else float("nan"),
                nmi=float(skm.normalized_mutual_info_score(li, si)),
                homogeneity=float(skm.homogeneity_score(li, si)),
                completeness=float(skm.completeness_score(li, si)))


def _ptext(p):
    """Compact p-value label."""
    if not np.isfinite(p):
        return "p n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


def contingency_figure(states, labels, label_name="behaviour", n_perm=200, seed=0,
                       title="states vs labels"):
    """Row-normalised contingency table P(label | state) as a heatmap, states
    reordered by Hungarian matching; title reports ARI (with circular-shift
    p-value), NMI, homogeneity and completeness (see ``contingency_table``).
    The number of frames per state is written at the end of each row."""
    ct = contingency_table(states, labels, n_perm=n_perm, seed=seed)
    R, N = ct["table"].to_numpy(), ct["counts"].to_numpy()
    K, L = R.shape
    Kc = int(np.max(np.asarray(states).astype(int))) + 1
    cols = state_colors(Kc)
    fig = _fig(float(np.clip(1.9 + 0.48 * L, 3.6, 7.2)), float(np.clip(1.25 + 0.26 * K, 2.1, 6.0)))
    with matplotlib.rc_context(RC):
        ax = fig.subplots()
        im = ax.imshow(R, cmap=HEAT_CMAP, vmin=0, vmax=1, aspect="auto",
                       interpolation="nearest")
        ax.set_xticks(range(L)); ax.set_xticklabels(ct["table"].columns,
                                                    rotation=0 if L <= 6 else 45,
                                                    ha="center" if L <= 6 else "right")
        ax.set_yticks(range(K)); ax.set_yticklabels(ct["table"].index)
        ax.tick_params(length=0)
        ax.tick_params(axis="y", pad=11)               # room for the colour swatches
        # state identity swatch between label and heatmap: x fixed 5 pt left of the
        # axes edge (independent of the figure size), y in data units
        swatch = ax.get_yaxis_transform() + ScaledTranslation(-5.5 / 72, 0,
                                                              fig.dpi_scale_trans)
        for k, sid in enumerate(ct["order"]):
            ax.plot([0], [k], marker="s", ms=5.5, mec="none", color=cols[int(sid) % len(cols)],
                    transform=swatch, clip_on=False)
            ax.text(L - 0.5 + 0.08, k, f"n={N[k].sum():,}", va="center", ha="left",
                    color=STYLE["ink_3"], fontsize=STYLE["font_size"] - 1.5)
        if K * L <= DYN_STYLE["annotate_max_cells"]:
            for a in range(K):
                for b in range(L):
                    v = R[a, b]
                    if v >= 0.005:
                        ax.text(b, a, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center",
                                va="center", fontsize=STYLE["font_size"] - 1.5,
                                color=_heat_text_color(v, 1.0))
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(label_name); ax.set_ylabel("discovered state")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, aspect=22, pad=0.02)
        cb.outline.set_visible(False); cb.ax.tick_params(length=2, width=0.5)
        cb.set_label(f"P({label_name} | state)")
        ax.set_title(f"ARI {ct['ari']:.2f} ({_ptext(ct['ari_p'])}, shift null)   "
                     f"NMI {ct['nmi']:.2f}\nhomogeneity {ct['homogeneity']:.2f}   "
                     f"completeness {ct['completeness']:.2f}", loc="left",
                     fontsize=STYLE["font_size"], color=STYLE["ink_2"])
        fig.suptitle(title, fontsize=STYLE["title_size"])
    fig.nemba_stats = ct
    return fig


# --------------------------------------------------------------------------- #
# 5. consistency between embeddings                                            #
# --------------------------------------------------------------------------- #

def consistency_figure(matrix, title="embedding consistency"):
    """Heatmap of a square DataFrame of pairwise consistency R^2 (rows =
    source / predictor, columns = target), e.g. ``align.consistency_matrix``.
    Colour range 0..1 (negative R^2 shown as 0); values are written in the
    cells for matrices up to 8 x 8; the diagonal (NaN) is left blank."""
    M = matrix.to_numpy(dtype=float)
    names = [str(c) for c in matrix.columns]
    rows = [str(r) for r in matrix.index]
    n = len(names)
    off = M[~np.eye(*M.shape, dtype=bool)] if M.shape[0] == M.shape[1] else M.ravel()
    mean_off = float(np.nanmean(off)) if np.isfinite(off).any() else float("nan")
    side = float(np.clip(1.6 + 0.42 * n, 2.6, 6.5))
    fig = _fig(side + 0.9, side + 0.2)
    cmap = HEAT_CMAP.copy()
    cmap.set_bad("#f0efec")              # neutral grey diagonal: "not applicable", not 0
    with matplotlib.rc_context(RC):
        ax = fig.subplots()
        im = ax.imshow(np.ma.masked_invalid(np.clip(M, 0, 1)), cmap=cmap, vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_xticks(range(n)); ax.set_yticks(range(len(rows)))
        long = max(len(s) for s in names) > 5 or n > 6
        ax.set_xticklabels(names, rotation=45 if long else 0, ha="right" if long else "center")
        ax.set_yticklabels(rows)
        ax.tick_params(length=0)
        if n <= 8:
            for a in range(M.shape[0]):
                for b in range(M.shape[1]):
                    v = M[a, b]
                    if np.isfinite(v):
                        ax.text(b, a, f"{v:.2f}", ha="center", va="center",
                                fontsize=STYLE["font_size"] - 1,
                                color=_heat_text_color(np.clip(v, 0, 1), 1.0))
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("target (predicted)"); ax.set_ylabel("source (predictor)")
        cb = fig.colorbar(im, ax=ax, shrink=0.75, aspect=22, pad=0.03)
        cb.outline.set_visible(False); cb.ax.tick_params(length=2, width=0.5)
        cb.set_label("cross-validated R²")
        ax.set_title(f"{title}\nmean off-diagonal R² {mean_off:.2f}", loc="left")
    return fig


# --------------------------------------------------------------------------- #
# 6. population raster                                                         #
# --------------------------------------------------------------------------- #

def _zscore_rows(X):
    """Row-wise z-score (constant rows become 0)."""
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def _bin_cols(X, f):
    """Average groups of ``f`` adjacent columns (last group may be partial)."""
    if f <= 1:
        return X
    n = X.shape[1]
    nb = int(np.ceil(n / f))
    pad = nb * f - n
    Xp = np.pad(X, ((0, 0), (0, pad)), constant_values=np.nan)
    return np.nanmean(Xp.reshape(X.shape[0], nb, f), axis=2)


def sort_neurons(X, method="auto", seed=0):
    """Order neurons (rows of X [channels, time]) for a raster display.

    'rastermap' (or 'auto' when the package is installed): Rastermap
    (Stringer et al., Nat Neurosci 2025); with <= 200 neurons single neurons
    are sorted without clustering (settings of the package's single-neuron
    tutorial: n_clusters=None, locality=0.15, time_lag_window=15,
    grid_upsample=0). Fallback ('pca', or when rastermap is missing / fails):
    angle of each neuron's loading on the first two PCs of the z-scored
    activity, which orders ring-tuned populations by preferred phase but is
    only a coarse 1-D ordering for other structures.

    Returns (order [n_channels], method_used)."""
    X = np.asarray(X, float)
    N, T = X.shape
    if method in ("auto", "rastermap") and N >= 3:
        try:
            from rastermap import Rastermap
        except ImportError:
            Rastermap = None
        if Rastermap is not None:
            try:
                if N <= 200:
                    model = Rastermap(n_clusters=None, n_PCs=int(min(64, N - 1, T - 1)),
                                      locality=0.15, time_lag_window=15, grid_upsample=0,
                                      verbose=False, random_state=seed)
                else:
                    model = Rastermap(n_clusters=int(min(100, N // 2)),
                                      n_PCs=int(min(200, N - 1, T - 1)), verbose=False,
                                      random_state=seed)
                with warnings.catch_warnings():    # "<= 50 samples, skip clustering" notice
                    warnings.simplefilter("ignore", UserWarning)
                    model.fit(X.astype(np.float32))
                return np.asarray(model.isort, int), "rastermap"
            except Exception:  # noqa: BLE001  (degenerate input: use the fallback)
                pass
    Xz = _zscore_rows(X)
    U, S, _ = np.linalg.svd(Xz, full_matrices=False)
    if U.shape[1] < 2:
        return np.arange(N), "identity"
    ang = np.arctan2(U[:, 1] * S[1], U[:, 0] * S[0])
    return np.argsort(ang), "PC1-PC2 angle"


def rastermap_figure(X, times=None, states=None, behavior=None, behavior_name="behaviour",
                     max_cols=2000, circular=False, seed=0, title="population raster"):
    """Sorted, z-scored population activity with the state raster and a
    behaviour trace aligned below on a shared time axis.

    X [channels, time] (e.g. ``Embedder.X_input``). Neurons are sorted by
    ``sort_neurons`` (Rastermap when installed). Time is averaged into
    <= ``max_cols`` columns; more than ``DYN_STYLE['raster_max_rows']``
    neurons are averaged into super-neurons of adjacent sorted rows.
    The returned figure carries ``fig.nemba_order`` (row order, top to
    bottom) and ``fig.nemba_sort_method``."""
    X = np.asarray(X, float)
    N, T = X.shape
    order, how = sort_neurons(X, seed=seed)
    Xs = _zscore_rows(X)[order]
    rb = int(np.ceil(N / DYN_STYLE["raster_max_rows"]))
    if rb > 1:                                         # super-neurons
        Xs = _bin_cols(Xs.T, rb).T
    f = int(np.ceil(T / max(int(max_cols), 1)))
    Xb = _bin_cols(Xs, f)
    t = np.arange(T, dtype=float) if times is None else np.asarray(times, float)
    tlab = "frame" if times is None else "time (s)"
    tracks = [("heat", 3.0)]
    if states is not None:
        tracks.append(("state", 0.22))
    if behavior is not None:
        tracks.append(("beh", 0.6))
    fig = _fig(7.2, 0.7 + 0.62 * sum(h for _, h in tracks))
    with matplotlib.rc_context(RC):
        axes = fig.subplots(len(tracks), 1, sharex=True, squeeze=False,
                            height_ratios=[h for _, h in tracks])[:, 0]
        ah = axes[0]
        im = ah.imshow(Xb, aspect="auto", cmap=HEAT_CMAP, vmin=DYN_STYLE["raster_vmin"],
                       vmax=DYN_STYLE["raster_vmax"], interpolation="nearest",
                       extent=[t[0], t[-1], Xb.shape[0] - 0.5, -0.5])
        im.set_rasterized(True)
        ah.set_yticks([0, Xb.shape[0] - 1])
        ah.set_yticklabels(["1", f"{N}"])
        ah.set_ylabel(f"neurons\n({how})" + (f"\n{rb} per row" if rb > 1 else ""),
                      rotation=0, ha="right", va="center")
        _clean(ah, keep=("left",) if len(tracks) > 1 else ("left", "bottom"))
        if len(tracks) > 1:                            # time ticks only on the last track
            ah.tick_params(axis="x", labelbottom=False, length=0)
        cax = ah.inset_axes([1.012, 0.55, 0.012, 0.45])
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=STYLE["font_size"] - 1.5)
        cb.set_label("z", rotation=0, labelpad=6, va="center")
        r = 1
        if states is not None:
            s = np.asarray(states).astype(int)
            K = int(s.max()) + 1 if s.size else 1
            _raster(axes[r], s, t, K)
            axes[r].set_ylabel("state", rotation=0, ha="right", va="center")
            if r < len(tracks) - 1:
                axes[r].tick_params(labelbottom=False, length=0)
            handles = [Patch(color=c, label=f"{k}") for k, c in enumerate(state_colors(K))]
            fig.legend(handles=handles, loc="outside right center", title="state",
                       title_fontsize=STYLE["font_size"], handlelength=1.0, labelspacing=0.3)
            r += 1
        if behavior is not None:
            ab = axes[r]
            if circular:
                ab.plot(*_wrap_breaks(t, behavior), color=STYLE["ink_2"], lw=0.6)
                ab.set_yticks([-np.pi, np.pi]); ab.set_yticklabels(["-π", "π"])
                _clean(ab, grid_axis="y")
                ab.set_ylabel(behavior_name, rotation=0, ha="right", va="center")
            else:
                _behaviour_track(ab, behavior, t, behavior_name)
        axes[-1].set_xlim(t[0], t[-1])
        axes[-1].set_xlabel(tlab)
        ah.set_title(f"{N} neurons x {T:,} frames" + (f" ({f} frames per column)" if f > 1 else ""),
                     loc="left", fontsize=STYLE["font_size"])
        fig.suptitle(title, fontsize=STYLE["title_size"])
    fig.nemba_order = np.asarray(order)
    fig.nemba_sort_method = how
    return fig


__all__ = ["persistence_figure", "flow_field", "flow_field_figure", "recurrence_figure",
           "contingency_table", "contingency_figure", "consistency_figure", "sort_neurons",
           "rastermap_figure", "HEAT_CMAP", "DIVERGING_CMAP", "DYN_STYLE"]
