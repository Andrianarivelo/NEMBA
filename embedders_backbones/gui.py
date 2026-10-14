"""NEMBA desktop GUI: fit a backbone, explore the embedding, decode behaviour.

Usage
-----
    python main.py                          (no arguments opens this window)
    python -m embedders_backbones.gui

    Sidebar (left, top to bottom)
      Data       recording file (empty = synthetic demo), how to read it
                 ('auto' recognises spike times, COCO json, DeepLabCut CSV,
                 tables) and the data advisor (goal + "Analyse data")
      Behaviour  optional variables to decode and colour by (.csv / .npz / .npy),
                 decoder, PSID rows, co-smoothing score
      Model      backbone, speed preset, objective, likelihood, masking,
                 states and state model, embedding size, step budget
      Output     folder for figures (png/pdf/svg), CSV tables, GIF, HTML explorer
      Compare    comma list of backbones fitted side by side
    Run (Ctrl+R / F5) fits one backbone, Compare fits the list, Stop (Esc)
    cancels at the next validation check of the training loop. "Analyse
    data" profiles the recording and fills the Advisor tab; "Apply
    suggestions" copies the advice into the sidebar, "Run suggested" also runs.

    Tabs (right): Advisor (data profile + advice), Overview (dashboard),
    Embedding (2-D map, re-colourable), Decoding (held-out scores + table),
    Dynamics (topology, flow field, recurrence, raster, state contingency),
    Training (loss curves), Compare (grid or consistency matrix), Log.

Threading design
----------------
Tk is single threaded: only the main thread may touch widgets. All long work
(loading, training, projections, decoding, writing files and BUILDING the
matplotlib figures) runs in a daemon worker thread that talks to the GUI only
through a ``queue.Queue`` of ``(kind, payload)`` messages:

    ("log", str | (str, tag))          append text to the Log tab
    ("progress", dict)                 training step (determinate progress bar)
    ("phase", (text, frac | None))     stage text (None = indeterminate bar)
    ("call", fn)                       run fn() on the main thread (show results)
    ("error", (summary, traceback))    an exception escaped a worker
    ("finished", (kind, status, msg))  job ended: "done" | "stopped" | "error"

The main thread drains the queue every ``POLL_MS`` (60 ms) via ``root.after``.
Figures are bare ``matplotlib.figure.Figure`` objects (no pyplot), so they are
safe to build in the worker; only ``FigureCanvasTkAgg`` creation and drawing
happen on the main thread. Stop sets a ``threading.Event``; the progress
callback (fired at every validation check) and the log callback raise the
private ``_Cancelled`` exception when it is set, which unwinds the worker and
reports "stopped".

The last used field values are remembered in ``~/.nemba_gui.json``.
"""

from __future__ import annotations

import copy
import gc
import importlib
import importlib.util
import json
import os
import queue
import re
import sys
import textwrap
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import font as tkfont

import matplotlib
matplotlib.use("TkAgg")          # GUI module only: the package itself never imports pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine

from . import data as _data
from . import viz
from .core import Embedder
from .compare import compare_embedders
from .backbones import list_backbones, label, CPU_COST_HINT
from .training import SPEED_PRESETS, OBJECTIVES, MASK_MODES
from .decoding import infer_task
from .synthetic import make_dataset, demo_spikes

try:                                           # version string for the header
    from . import __version__ as _VERSION
except Exception:  # noqa: BLE001
    _VERSION = ""

# kept for backward compatibility (main.py of v0.1 imported it from here)
_demo_spikes = demo_spikes


# =========================================================================== #
# user-configurable appearance and defaults (edit here)                        #
# =========================================================================== #
PALETTE = dict(
    bg="#f6f6f4",            # window background
    panel="#ffffff",         # cards, tabs, tables
    text="#1f1f1d",          # primary text
    muted="#6b6a65",         # secondary text, hints
    accent="#2a78d6",        # primary action, selection
    accent_hover="#256abf",
    accent_press="#1f5aa3",
    accent_soft="#e6eefa",   # selected table rows, focus halo
    accent_off="#a9c6ec",    # disabled primary button
    border="#e3e2de",        # card and separator lines
    field="#d3d2cd",         # input field outline
    hover="#f1f0ed",         # hovered secondary controls
    disabled_bg="#f2f2f0",
    disabled_fg="#a6a5a0",
    heading="#f4f4f2",       # table headings
    stripe="#fafaf8",        # zebra rows
    ok="#1b9e6b", warn="#c98400", bad="#d1453b", idle="#c4c3be",
)
FONT_FAMILY = "Segoe UI"     # falls back to Tk's default family when missing
MONO_FAMILY = "Consolas"
BODY_PT, TITLE_PT, SMALL_PT, BRAND_PT, MONO_PT = 9, 10, 8, 15, 9

SIDEBAR_W = 340              # logical px (scaled with the screen DPI)
LABEL_COL_W = 93             # width of the label column (fits 'Embedding dim' + padding)
WINDOW_SIZE = (1380, 860)
MIN_SIZE = (1080, 680)
POLL_MS = 60                 # queue drain period of the main thread
MAX_LOG_LINES = 6000
# largest up-scaling of a figure above its native size (keeps fonts in proportion)
SCALE_CAP = dict(overview=1.35, embedding=1.45, decoding=1.6, training=1.9, compare=1.6,
                 advisor=1.45, dynamics=1.6)
EMBED_FIGSIZE = (6.2, 4.5)   # inches of the Embedding-tab scatter

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".nemba_gui.json")
INPUT_TYPES = ["auto", "spikes", "pose", "mask", "coco_mask", "fiber_photometry", "timeseries"]
TIME_AXES = ["1 - channels x time", "0 - time x channels"]
PROJECTIONS = ["umap", "pca", "tsne", "tphate"]
PRETTY = {"umap": "UMAP", "pca": "PCA", "tsne": "t-SNE", "tphate": "T-PHATE"}
SPEEDS = list(SPEED_PRESETS)
GOALS = ["auto", "decode", "states", "visualize"]          # advisor goals
LIKELIHOODS = ["gaussian", "poisson", "auto"]              # Embedder(likelihood=)
MASK_CHOICES = list(MASK_MODES)                            # Embedder(mask_mode=)
STATE_MODELS = ["auto", "kmeans", "hmm", "arhmm"]          # hmm / arhmm need .states
EMB_DIMS = ["auto", "2", "3", "4", "8", "12", "16", "32", "64"]
DECODERS = ["linear", "knn", "gbm", "logistic"]            # Embedder.decode(model=)
DEFAULT_COMPARE = "pca,tcn,tst,bilstm"
DEFAULTS = dict(
    file="", input_type="auto", time_axis=TIME_AXES[0], bin_size="0.2", fs="", dff=False,
    mask_size="64", mask_key="", beh_file="", circular="", n_perm="200", decoder="linear",
    psid=False, cosmooth=False, goal="auto",
    backbone="tcn", speed="fast", objective="masked", likelihood="gaussian", mask_mode="time",
    n_states="auto", state_model="auto", emb_dim="auto", max_steps="",
    smooth="4.0", outdir=os.path.join(os.getcwd(), "embedder_out"), gif=True, html=True,
    gif_method="umap", compare=DEFAULT_COMPARE,
)
# CPU cost hint -> (text, palette colour)
COST_TEXT = {"instant": ("instant, no training", "ok"), "fast": ("fast on CPU", "ok"),
             "medium": ("medium cost on CPU", "warn"),
             "slow": ("slow on CPU, prefer speed Fast", "bad"),
             "missing": ("requires cebra (pip install cebra)", "bad")}
DEMO_HINT = "Empty = synthetic demo with behaviour."
# which optional Data rows each input type needs (others are hidden)
DATA_ROWS = {"auto": ("bin",), "spikes": ("bin",), "pose": ("axis",), "timeseries": ("axis",),
             "mask": ("axis",), "coco_mask": ("mask", "mask_key", "fs"),
             "fiber_photometry": ("axis", "fs", "dff")}
# advisor fit method -> explicit input type (used by "Apply suggestions")
FIT_TO_INPUT = {"fit_spikes": "spikes", "fit_fiber": "fiber_photometry",
                "fit_coco_masks": "coco_mask", "fit_features": "timeseries"}
# Dynamics tab: how to read each figure (shown next to its statistics)
DYN_HELP = {
    "topology": "Persistent homology of the latent cloud (bars divided by its diameter). One "
                "long H1 bar well above the Gaussian null suggests a ring; two long H1 bars and "
                "one H2 bar a torus. Long bars can also come from unvisited gaps.",
    "flow": "Mean latent velocity per cell of the 2-D projection. Coherence near 1: every "
            "passage moves the same way; near 0: crossing or diffusive motion. In UMAP, t-SNE "
            "or T-PHATE space read directions, not speeds.",
    "recurrence": "Similarity of the latent state between all pairs of frames. Off-diagonal "
                  "stripes: returns to earlier states; checkerboard: discrete states; broad "
                  "diagonal band: slow drift. tau 1e is the lag where similarity drops to 1/e.",
    "rastermap": "Neurons sorted by activity similarity (Rastermap when installed, else the "
                 "PC1-PC2 angle), with the state raster and the first behaviour on the same "
                 "time axis.",
    "contingency": "P(label | state) after Hungarian matching. ARI is tested against a "
                   "circular-shift null (both sequences are autocorrelated); homogeneity: each "
                   "state holds one label; completeness: each label sits in one state.",
}

_FIT_RE = re.compile(r"^=+\s*fitting\s+(\S+)\s*=+$")


# =========================================================================== #
# small helpers                                                                #
# =========================================================================== #
class _Cancelled(Exception):
    """Raised inside a worker thread (from a library callback) after Stop."""


class _FormError(ValueError):
    """Invalid sidebar value; carries the widget that should get the focus."""

    def __init__(self, message, widget=None):
        """Store the user-facing ``message`` and the offending ``widget``."""
        super().__init__(message)
        self.widget = widget


@dataclass
class _RunResult:
    """Everything a finished single-backbone job hands to the main thread."""
    emb: Embedder
    behaviour: dict | None                   # aligned on emb.times (name -> array)
    circular: list
    outdir: str
    figures: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    view: tuple = ("umap", "time")           # (projection, colour key) of the embedding figure
    gif_method: str = "umap"                 # projection used by the flow field (as in the files)
    decoder: str = "linear"
    cosmooth: dict | None = None             # emb.cosmooth() result (bits / spike), if computed


@dataclass
class _CompareResult:
    """Everything a finished comparison job hands to the main thread."""
    cmp: object
    outdir: str
    figure: Figure
    view: tuple = ("umap", "state")
    consistency: object = None               # DataFrame (row predicts column) or None
    cons_figure: Figure | None = None


@dataclass
class _AdviceResult:
    """Everything a finished advisor job hands to the main thread."""
    rec: object                              # advisor.Recommendation
    figure: Figure                           # profile_figure(rec.profile)
    report: str                              # rec.report() (markdown)
    key: tuple                               # (file, input type) the advice belongs to
    name: str                                # data name for captions
    outdir: str


def _fmt(v):
    """Compact human formatting of a table cell (numbers, bools, missing)."""
    if v is None:
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if not np.isfinite(v):
            return "-"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.3f}"
    return str(v)


def _fmt_p(p):
    """Format a p-value ('<0.001' below the display resolution)."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _short_path(path, keep=2):
    """Keep only the last ``keep`` components of a long path (for captions)."""
    parts = Path(path).parts
    return str(path) if len(parts) <= keep + 1 else "..." + os.sep + os.sep.join(parts[-keep:])


def load_settings(path):
    """Read the remembered fields; any problem (missing, corrupted) -> {}."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def save_settings(path, values):
    """Write the fields atomically (temp file + replace); never raises."""
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(values, fh, indent=1)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass


def enable_dpi_awareness():
    """Make the process DPI aware on Windows (crisp text); must run before Tk()."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass


def _embedding_figure(emb, method, color_key, behaviour, circular):
    """Build the Embedding-tab scatter (worker-safe: no Tk involved).

    ``color_key`` is 'time', 'state' or 'beh:<name>' (a behaviour variable
    aligned on the embedding frames)."""
    color, beh, name, circ = "time", None, None, False
    if color_key == "state":
        color = "state"
    elif color_key.startswith("beh:") and behaviour and color_key[4:] in behaviour:
        name = color_key[4:]
        color, beh, circ = "behavior", np.asarray(behaviour[name]), name in (circular or ())
    proj = {method: emb.projection(method)}            # cached, computed here if missing
    fig = Figure(figsize=EMBED_FIGSIZE, facecolor=viz.STYLE["surface"])
    # Embedder.figure has no `circular` argument, so call the viz builder directly
    return viz.embedding_figure(emb.Z, methods=(method,), color=color, states=emb.states,
                                seed=emb.seed, projections=proj, times=emb.times,
                                behavior=beh, behavior_name=name, circular=circ,
                                title=f"{label(emb.backbone)} embedding", fig=fig)


FIGURE_FEATURES = ("embedding", "raw_pca")   # decoding rows viz.decoding_figure labels right
FEATURE_NAMES = {"embedding": "embedding", "raw_pca": "raw PCA", "psid": "PSID",
                 "bayes_counts": "Bayes (counts)"}


def _decoding_view(result):
    """Decoding result to draw with ``viz.decoding_figure``.

    ``decoding_figure`` labels every non-embedding row 'raw PCA', so reference
    rows such as PSID ('psid') or the Bayesian count decoder ('bayes_counts')
    would be mislabelled: the figure gets only ``FIGURE_FEATURES`` rows
    (shallow copy; the table in the GUI still lists every row)."""
    tab = getattr(result, "table", None)
    if tab is None or "features" not in tab or tab["features"].isin(FIGURE_FEATURES).all():
        return result
    view = copy.copy(result)
    view.table = tab[tab["features"].isin(FIGURE_FEATURES)].reset_index(drop=True)
    return view


def _dynamics_choices(emb, behaviour, circular):
    """Dynamics figures available for ``emb``: display name -> (kind, behaviour name).

    Mirrors ``Embedder.write_dynamics``: the raster needs >= 8 input channels,
    one contingency table per discrete (classification) behaviour."""
    out = {"Topology": ("topology", None), "Flow field": ("flow", None),
           "Recurrence": ("recurrence", None)}
    if emb.X_input is not None and emb.X_input.shape[0] >= 8:
        out["Rastermap"] = ("rastermap", None)
    for name, v in (behaviour or {}).items():
        if name not in (circular or ()) and infer_task(np.asarray(v)) == "classification":
            out[f"Contingency {name}"] = ("contingency", name)
    return out


def _dynamics_figure(emb, kind, beh_name, behaviour, circular, method):
    """Build one Dynamics figure (worker-safe) with the arguments of
    ``Embedder.write_dynamics``. Returns (figure, [(statistic, value)], caption)."""
    from . import viz_dynamics as VD
    circular = tuple(circular or ())
    if kind == "topology":
        fig = VD.persistence_figure(emb.Z, seed=emb.seed)
        cap = "Vietoris-Rips persistence (ripser) of a random subset of frames"
    elif kind == "flow":
        fig = VD.flow_field_figure(*emb.projection(method), dt=emb.frame_dt(), states=emb.states)
        cap = f"flow in {PRETTY.get(method, method)} space (projection of the run)"
    elif kind == "recurrence":
        fig = VD.recurrence_figure(emb.Z, times=emb.times)
        cap = "cosine similarity between strided frames"
    elif kind == "rastermap":
        first = next(iter((behaviour or {}).items()), (None, None))
        fig = VD.rastermap_figure(emb.X_input, times=emb.times, states=emb.states,
                                  behavior=first[1], behavior_name=first[0] or "behaviour",
                                  circular=first[0] in circular, seed=emb.seed)
        cap = f"{emb.X_input.shape[0]} input channels, sorted by {fig.nemba_sort_method}"
    elif kind == "contingency":
        fig = VD.contingency_figure(emb.states, behaviour[beh_name], label_name=beh_name,
                                    seed=emb.seed)
        cap = f"discovered states vs '{beh_name}'"
    else:
        raise ValueError(f"unknown dynamics figure {kind!r}")
    rows = _scalar_stats(getattr(fig, "nemba_stats", None))
    if kind == "rastermap":                     # the raster carries an order, not statistics
        rows = [("sort method", fig.nemba_sort_method),
                ("neurons", int(emb.X_input.shape[0])), ("frames", int(emb.X_input.shape[1]))]
    return fig, rows, cap


def _scalar_stats(stats):
    """Scalar entries of a ``fig.nemba_stats`` dict as (readable name, value) rows.

    Arrays, tables and lists are skipped (they are in the CSV files)."""
    rows = []
    for k, v in (stats or {}).items():
        if isinstance(v, (bool, np.bool_)) or (np.isscalar(v) and not isinstance(v, (str, bytes))):
            rows.append((str(k).replace("_", " "), v))
    return rows


class _Reporter:
    """Worker-side bridge turning library callbacks into queue messages.

    ``log`` / ``progress`` are handed to ``Embedder`` (or ``compare_embedders``);
    both raise ``_Cancelled`` once Stop was pressed. For a comparison the
    '=== fitting <bb> ===' log lines advance the backbone counter so that the
    progress bar shows the overall fraction across backbones."""

    def __init__(self, post, stop_event, backbones, post_fit_text):
        """``post(kind, payload)`` enqueues a message; ``backbones`` lists the job's models."""
        self.post = post
        self.stop = stop_event
        self.backbones = list(backbones)
        self.i = 0 if len(self.backbones) == 1 else -1
        self.post_fit_text = post_fit_text

    @property
    def prefix(self):
        """Short label of the backbone being fitted, e.g. 'tcn (2/4)'."""
        n = len(self.backbones)
        bb = self.backbones[max(self.i, 0)]
        return bb if n == 1 else f"{bb} ({max(self.i, 0) + 1}/{n})"

    def check(self):
        """Cancellation checkpoint: raise ``_Cancelled`` when Stop was pressed."""
        if self.stop.is_set():
            raise _Cancelled()

    def phase(self, text, frac=None):
        """Announce a stage (``frac`` None = indeterminate progress)."""
        self.post("phase", (text, frac))

    def log(self, msg):
        """Library log callback: forward the text, derive the stage, check Stop."""
        msg = str(msg)
        self.post("log", msg)
        s = msg.strip()
        m = _FIT_RE.match(s)
        if m and len(self.backbones) > 1:
            self.i = min(self.i + 1, len(self.backbones) - 1)
            self.phase(f"{self.prefix}: fitting", self.i / len(self.backbones))
        elif s.startswith("trained ") or "closed-form fit" in s:
            self.phase(f"{self.prefix}: {self.post_fit_text}")
        self.check()

    def progress(self, step, max_steps, train_loss, val_loss):
        """Library progress callback (each validation check): report, check Stop."""
        self.check()
        n = len(self.backbones)
        frac = (max(self.i, 0) + step / max(int(max_steps), 1)) / n
        self.post("progress", dict(prefix=self.prefix, step=int(step), max=int(max_steps),
                                   loss=float(train_loss), val=float(val_loss), frac=frac))


# =========================================================================== #
# reusable widgets                                                             #
# =========================================================================== #
class Tooltip:
    """Delayed hover tooltip for any widget (dark chip under the pointer)."""

    def __init__(self, widget, text, app, delay_ms=550):
        """Attach ``text`` to ``widget``; ``app`` provides fonts and scaling."""
        self.widget, self.text, self.app, self.delay = widget, text, app, delay_ms
        self._after = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        """Start the show timer when the pointer enters the widget."""
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        """Cancel a pending show timer."""
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None

    def _show(self):
        """Create the borderless tooltip window below the widget."""
        self._after = None
        if self._tip is not None or not self.widget.winfo_ismapped():
            return
        px = self.app.px
        x = self.widget.winfo_rootx() + px(4)
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + px(4)
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tk.Label(tip, text=self.text, justify="left", wraplength=px(300), bg="#2b2b29",
                 fg="#f4f4f2", font=self.app.f_small, padx=px(8), pady=px(5)).pack()
        self._tip = tip

    def _hide(self, _event=None):
        """Hide the tooltip and cancel any pending timer."""
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class ScrollableFrame(ttk.Frame):
    """Vertically scrollable container (Canvas + slim Scrollbar).

    Children go into ``self.inner``. The scrollbar column keeps its width
    when hidden, so toggling it never re-wraps the content (no layout loop)."""

    def __init__(self, master, app, style="App.TFrame"):
        """Build the canvas, the inner frame and the auto-hiding scrollbar."""
        super().__init__(master, style=style)
        self.app = app
        self.canvas = tk.Canvas(self, bg=PALETTE["bg"], highlightthickness=0, bd=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview,
                                 style="Side.Vertical.TScrollbar")
        self.inner = ttk.Frame(self.canvas, style=style)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, minsize=app.px(9))
        self.rowconfigure(0, weight=1)
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)

    def _on_inner(self, _event=None):
        """Content size changed: update the scroll region."""
        self.canvas.configure(scrollregion=(0, 0, self.inner.winfo_reqwidth(),
                                            self.inner.winfo_reqheight()))
        self._update_bar()

    def _on_canvas(self, event):
        """Viewport resized: stretch the content to the viewport width."""
        self.canvas.itemconfigure(self._win, width=event.width)
        self._update_bar()

    def overflowing(self):
        """True when the content is taller than the visible area."""
        return self.inner.winfo_reqheight() > self.canvas.winfo_height() + 1

    def _update_bar(self):
        """Show the scrollbar only when it is needed."""
        if self.overflowing():
            self.vsb.grid()
        else:
            self.vsb.grid_remove()
            self.canvas.yview_moveto(0)

    def scroll(self, units):
        """Scroll by ``units`` lines (no-op when everything is visible)."""
        if self.overflowing():
            self.canvas.yview_scroll(int(units), "units")

    def contains(self, widget):
        """True when ``widget`` lives inside this scroll area."""
        return widget is not None and str(widget).startswith(str(self.canvas))


class FigurePane(ttk.Frame):
    """A titled matplotlib area: header (title, caption, controls), a stage
    that shows one Figure at its native aspect ratio, and a navigation toolbar.

    ``show(fig)`` replaces the current figure: the old canvas widget and
    toolbar are destroyed and a new ``FigureCanvasTkAgg`` is created. Call it
    on the main thread only.

    Constrained layout ignores the WIDTH of axes titles, so long left-aligned
    titles can stick out of the figure (saved files hide this thanks to
    ``bbox_inches='tight'``). After each draw the pane measures the tight
    bounding box and, if needed, shrinks the layout rectangle and redraws."""

    def __init__(self, master, app, title, empty_text, scale_cap=1.5, min_h=None):
        """Build the empty pane with a centred placeholder message.

        ``min_h`` (logical px) is the stage height requested from the layout
        (used inside paned windows so the figure gets most of the space)."""
        super().__init__(master, style="Panel.TFrame")
        self.app, self.scale_cap = app, scale_cap
        self.canvas = None
        self.toolbar = None
        self.figure = None
        self._native_px = None
        self._after = None
        self._placed = None            # (w, h) of the canvas widget
        self._rounds = 0               # overflow corrections done at this size
        px = app.px

        head = ttk.Frame(self, style="Panel.TFrame")
        head.grid(row=0, column=0, sticky="ew", padx=px(16), pady=(px(12), px(6)))
        self.title_lbl = ttk.Label(head, text=title, style="PaneTitle.TLabel")
        self.title_lbl.pack(side="left")
        # packed before the caption: the controls keep their space, the caption is cut
        self.controls = ttk.Frame(head, style="Panel.TFrame")
        self.controls.pack(side="right")
        self.caption_lbl = ttk.Label(head, text="", style="Caption.TLabel")
        self.caption_lbl.pack(side="left", padx=(px(12), px(12)))

        self.stage = tk.Frame(self, bg=PALETTE["panel"], highlightthickness=0, bd=0)
        if min_h:
            self.stage.configure(height=px(min_h))
        self.stage.grid(row=1, column=0, sticky="nsew", padx=px(8))
        self.placeholder = ttk.Label(self.stage, text=empty_text, style="Empty.TLabel",
                                     justify="center", wraplength=px(420))
        self.placeholder.place(relx=0.5, rely=0.45, anchor="center")

        ttk.Frame(self, style="Line.TFrame", height=1).grid(row=2, column=0, sticky="ew")
        self.tbar_holder = tk.Frame(self, bg=PALETTE["panel"], height=px(34))
        self.tbar_holder.grid(row=3, column=0, sticky="ew", padx=px(8))
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.stage.bind("<Configure>", self._on_stage)

    # ----- public ----------------------------------------------------------- #
    def show(self, fig, caption=None):
        """Display ``fig`` (replacing the previous one) and its toolbar."""
        self._destroy_canvas()
        self.figure = fig
        # Cached figures (Dynamics, Compare toggle) can be shown again. Their Tk
        # canvas scaled the dpi by the device pixel ratio and resized the inches,
        # and re-pointing the canvas makes matplotlib take the SCALED dpi as the
        # new "original" one, so a second canvas would scale it twice (text 1.75x
        # too large on a 168-dpi screen). The native dpi and size are therefore
        # remembered on the first show and restored on every later one.
        dpi0 = getattr(fig, "_nemba_dpi0", None)
        if dpi0 is None:
            dpi0 = float(getattr(fig, "_original_dpi", fig.dpi))
            fig._nemba_dpi0 = dpi0
            fig._nemba_size0 = tuple(float(v) for v in fig.get_size_inches())
        else:
            fig._original_dpi = dpi0
            fig.set_dpi(dpi0)
            fig.set_size_inches(fig._nemba_size0, forward=False)
        size_in = np.asarray(fig._nemba_size0, float)
        self._native_px = size_in * dpi0 * self.app.dpr                # physical pixels
        self._placed, self._rounds = None, 0
        self.canvas = FigureCanvasTkAgg(fig, master=self.stage)
        self.canvas.mpl_connect("draw_event", self._on_draw)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg=PALETTE["panel"], highlightthickness=0, bd=0)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.tbar_holder, pack_toolbar=False)
        self.app.style_toolbar(self.toolbar)
        self.toolbar.pack(side="left", fill="x", expand=True)
        self.placeholder.place_forget()
        if caption is not None:
            self.caption_lbl.configure(text=caption)
        self._layout_now()
        self.canvas.draw_idle()

    def clear(self, text=None, caption=""):
        """Remove the figure and show a placeholder message instead."""
        self._destroy_canvas()
        if text:
            self.placeholder.configure(text=text)
        self.caption_lbl.configure(text=caption)
        self.placeholder.place(relx=0.5, rely=0.45, anchor="center")

    def destroy(self):
        """Cancel the pending debounced layout before the widget goes away
        (else Tk reports 'invalid command name ..._layout_now')."""
        if self._after is not None:
            try:
                self.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None
        super().destroy()

    # ----- internals -------------------------------------------------------- #
    def _destroy_canvas(self):
        """Destroy the current canvas widget and toolbar (frees the Figure).

        Figure, canvas and toolbar reference each other; the cycle holds Tk
        images and variables. It is broken here so that reference counting
        frees them on the main thread: if the cyclic garbage collector freed
        them later from a worker thread, their __del__ would call Tk from
        that thread ("main thread is not in main loop")."""
        canvas, toolbar = self.canvas, self.toolbar
        self.canvas = self.toolbar = self.figure = None
        if toolbar is not None:
            toolbar.destroy()
            # toolbar -> _buttons -> Button -> .master -> toolbar is a cycle holding
            # PhotoImages (icons), a Font and Tk variables: drop the toolbar's own
            # references so the whole tree is freed right now, on this thread
            buttons = list(getattr(toolbar, "_buttons", {}).values())
            toolbar.__dict__.clear()
            for b in buttons:
                b.__dict__.clear()
            del buttons
        if canvas is not None:
            canvas.get_tk_widget().destroy()
            canvas.toolbar = None
            fig = getattr(canvas, "figure", None)
            if fig is not None:
                FigureCanvasBase(fig)              # re-points fig.canvas away from Tk
        self.app.request_gc()

    def _on_stage(self, _event=None):
        """Debounced resize: re-render only once the user stops dragging."""
        if self._after is not None:
            self.after_cancel(self._after)
        self._after = self.after(90, self._layout_now)

    def _layout_now(self):
        """Fit the figure inside the stage, keeping its aspect ratio, centred."""
        self._after = None
        if self.canvas is None or self._native_px is None:
            return
        W, H = self.stage.winfo_width(), self.stage.winfo_height()
        if W < 20 or H < 20:
            return
        pad = self.app.px(6)
        w0, h0 = self._native_px
        s = min((W - pad) / w0, (H - pad) / h0, self.scale_cap)
        w, h = max(int(w0 * s), 50), max(int(h0 * s), 50)
        if (w, h) != self._placed:
            # new size: start again from the full layout rectangle
            self._placed, self._rounds = (w, h), 0
            eng = self.figure.get_layout_engine() if self.figure is not None else None
            if isinstance(eng, ConstrainedLayoutEngine):
                eng.set(rect=(0, 0, 1, 1))
        self.canvas.get_tk_widget().place(relx=0.5, rely=0.5, anchor="center", width=w, height=h)

    def _on_draw(self, event):
        """After a draw: if text sticks out of the figure, shrink the layout
        rectangle by the overflow (at most 3 corrections per size) and redraw."""
        fig = self.figure
        if fig is None or self._rounds >= 3 or event.canvas is not self.canvas:
            return
        eng = fig.get_layout_engine()
        if not isinstance(eng, ConstrainedLayoutEngine):
            return
        try:
            bb = fig.get_tightbbox(event.renderer)            # inches
        except Exception:  # noqa: BLE001
            return
        W, H = fig.get_size_inches()
        over = [max(0.0, -bb.x0), max(0.0, -bb.y0), max(0.0, bb.x1 - W), max(0.0, bb.y1 - H)]
        if max(over) < 0.01:
            return
        pad = 0.04                                            # inches of breathing room
        left, bottom, right, top = [(o + pad) / d if o > 0 else 0.0
                                    for o, d in zip(over, (W, H, W, H))]
        x, y, w, h = eng.get().get("rect", (0, 0, 1, 1))
        rect = (x + left, y + bottom, w - left - right, h - bottom - top)
        if rect[2] < 0.6 or rect[3] < 0.6:                    # never squash the content
            return
        eng.set(rect=rect)
        self._rounds += 1
        # redraw outside of this draw_event callback
        self.after_idle(lambda: self.canvas.draw() if self.canvas is not None else None)


class ReportView(ttk.Frame):
    """Read-only rich-text view of a markdown report (the advisor's advice).

    Renders what ``advisor.Recommendation.report()`` emits: '#' / '##'
    headings, '* ' bullets, '|' tables (monospace, every cell wrapped to the
    view width so nothing is clipped), fenced code blocks, inline **bold** and
    `code`, and an italic closing note. Bullets under a 'Warnings' heading use
    the warning colour. The text is re-flowed (debounced) when the width of
    the view changes, because table columns depend on it."""

    INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")

    def __init__(self, master, app, placeholder=""):
        """Build the text widget, its scrollbar and the formatting tags."""
        super().__init__(master, style="Panel.TFrame")
        self.app = app
        px, P = app.px, PALETTE
        self._md = ""
        self._cols = None                     # characters per line of the last rendering
        self._after = None
        self.text = tk.Text(self, font=app.f_body, bg=P["panel"], fg=P["text"], relief="flat",
                            bd=0, wrap="word", padx=px(18), pady=px(8), highlightthickness=0,
                            selectbackground=P["accent_soft"], selectforeground=P["text"],
                            insertbackground=P["panel"], cursor="arrow", spacing1=px(1),
                            spacing3=px(1), width=40, height=6, state="disabled")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.text.yview,
                            style="Slim.Vertical.TScrollbar")
        self.text.configure(yscrollcommand=App._autohide(vsb))
        self.text.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        fam = app.f_body.actual("family")
        self._fonts = dict(h1=tkfont.Font(self, family=fam, size=TITLE_PT + 3, weight="bold"),
                           italic=tkfont.Font(self, family=fam, size=BODY_PT, slant="italic"),
                           gap=tkfont.Font(self, family=fam, size=4))
        t, ind = self.text, px(6)
        t.tag_configure("h1", font=self._fonts["h1"], spacing1=px(2), spacing3=px(4))
        t.tag_configure("h2", font=app.f_title, foreground=P["accent"], spacing1=px(10),
                        spacing3=px(4))
        t.tag_configure("bullet", lmargin1=ind, lmargin2=px(20))
        t.tag_configure("para", lmargin1=ind, lmargin2=ind)
        t.tag_configure("note", font=self._fonts["italic"], foreground=P["muted"],
                        lmargin1=ind, lmargin2=ind, spacing1=px(6))
        t.tag_configure("code", font=app.f_mono, background=P["heading"], lmargin1=px(10),
                        lmargin2=px(22), rmargin=px(10), wrap="char")
        t.tag_configure("th", font=app.f_mono_bold, wrap="none", lmargin1=ind, lmargin2=ind)
        t.tag_configure("td", font=app.f_mono, wrap="none", lmargin1=ind, lmargin2=ind)
        t.tag_configure("rule", font=app.f_mono, foreground=P["field"], wrap="none",
                        lmargin1=ind)
        t.tag_configure("gap", font=self._fonts["gap"])
        t.tag_configure("placeholder", foreground=P["muted"], justify="center", spacing1=px(40))
        # colour and inline-font tags are created last: when several tags set the
        # same option, the most recently created tag wins
        t.tag_configure("warn", foreground=P["warn"])
        t.tag_configure("bold", font=app.f_bold)
        t.tag_configure("icode", font=app.f_mono, background=P["heading"])
        self.text.bind("<Configure>", self._on_resize)
        self.clear(placeholder)

    # ----- public ----------------------------------------------------------- #
    def set_markdown(self, md):
        """Show ``md`` (markdown text) from the top."""
        self._md = str(md or "")
        self._render()
        self.text.yview_moveto(0)

    def clear(self, text=""):
        """Show a muted placeholder message instead of a report."""
        self._md = ""
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("end", text, ("placeholder",))
        t.configure(state="disabled")

    def contents(self):
        """Plain text currently displayed (used by tests)."""
        return self.text.get("1.0", "end-1c")

    def destroy(self):
        """Cancel a pending re-flow before the widget goes away."""
        if self._after is not None:
            try:
                self.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None
        super().destroy()

    # ----- internals -------------------------------------------------------- #
    def _chars(self):
        """Monospace characters that fit on one line of the view."""
        w = self.text.winfo_width()
        if w <= 1:
            w = self.app.px(720)                  # not mapped yet: a typical width
        usable = w - 2 * int(float(self.text.cget("padx"))) - self.app.px(16)
        return max(40, int(usable // max(self.app.f_mono.measure("0"), 1)))

    def _on_resize(self, _event=None):
        """Debounced re-flow when the number of characters per line changes."""
        if self._after is not None:
            self.after_cancel(self._after)

        def _redo():
            """Re-render only if the line width (in characters) really changed."""
            self._after = None
            if self._md and self._chars() != self._cols:
                y = self.text.yview()[0]
                self._render()
                self.text.yview_moveto(y)
        self._after = self.after(120, _redo)

    def _render(self):
        """Parse the markdown line by line and insert it with tags."""
        t = self.text
        self._cols = self._chars()
        t.configure(state="normal")
        t.delete("1.0", "end")
        section, in_code, table = "", False, []
        for line in self._md.splitlines():
            s = line.rstrip()
            if s.startswith("```"):                       # fenced code block
                self._table(table)
                table = []
                in_code = not in_code
                continue
            if in_code:
                t.insert("end", s + "\n", ("code",))
                continue
            if s.lstrip().startswith("|"):                # table rows are collected first
                table.append(s)
                continue
            if table:
                self._table(table)
                table = []
            warn = section.lower().startswith("warning")
            if s.startswith("# "):
                t.insert("end", s[2:].strip() + "\n", ("h1",))
            elif s.startswith("## "):
                section = s[3:].strip()
                tags = ("h2", "warn") if section.lower().startswith("warning") else ("h2",)
                t.insert("end", section + "\n", tags)
            elif s.startswith(("* ", "- ")):
                base = ("bullet", "warn") if warn else ("bullet",)
                t.insert("end", "•  ", base)
                self._inline(s[2:], base)
                t.insert("end", "\n", base)
            elif len(s) > 2 and s.startswith("_") and s.endswith("_"):
                t.insert("end", s.strip("_") + "\n", ("note",))
            elif not s:
                t.insert("end", "\n", ("gap",))           # thin blank line
            else:
                self._inline(s, ("para",))
                t.insert("end", "\n", ("para",))
        self._table(table)
        t.configure(state="disabled")

    def _inline(self, s, base):
        """Insert ``s`` with **bold** and `code` spans on top of the ``base`` tags."""
        for part in self.INLINE.split(s):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**") and len(part) > 4:
                self.text.insert("end", part[2:-2], base + ("bold",))
            elif part.startswith("`") and part.endswith("`") and len(part) > 2:
                self.text.insert("end", part[1:-1], base + ("icode",))
            else:
                self.text.insert("end", part, base)

    def _table(self, raw):
        """Render collected '|' rows as an aligned monospace table.

        Column widths start at their natural width; while the table is wider
        than the view, the widest column loses one character (never below 8).
        Every cell is then wrapped to its column, so long evidence strings
        flow over several lines instead of being cut."""
        rows = []
        for line in raw:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue                                  # the |---|---| separator
            rows.append([re.sub(r"\*\*|`", "", c) for c in cells])
        if not rows:
            return
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        gap = 2
        avail = max(self._cols - gap * (ncol - 1), 8 * ncol)
        w = [max(1, max(len(r[j]) for r in rows)) for j in range(ncol)]
        while sum(w) > avail:
            j = max(range(ncol), key=lambda c: w[c])
            if w[j] <= 8:
                break
            w[j] -= 1
        wrapped = [[textwrap.wrap(c, w[j], break_long_words=True, break_on_hyphens=False) or [""]
                    for j, c in enumerate(r)] for r in rows]
        multi = any(len(c) > 1 for r in wrapped[1:] for c in r)
        for i, cells in enumerate(wrapped):
            tag = ("th",) if i == 0 else ("td",)
            for k in range(max(len(c) for c in cells)):
                text = (" " * gap).join((c[k] if k < len(c) else "").ljust(w[j])
                                        for j, c in enumerate(cells))
                self.text.insert("end", text.rstrip() + "\n", tag)
            if i == 0:
                self.text.insert("end", "─" * (sum(w) + gap * (ncol - 1)) + "\n", ("rule",))
            elif multi and i < len(wrapped) - 1:
                self.text.insert("end", "\n", ("gap",))   # separate wrapped rows
        self.text.insert("end", "\n", ("gap",))


# =========================================================================== #
# the application                                                              #
# =========================================================================== #
class App:
    """Main window: sidebar form, result tabs, status bar and job control."""

    # ----------------------------------------------------------------- setup --
    def __init__(self, root, settings_path=None):
        """Build the whole interface on ``root`` and start the queue poller.

        ``settings_path`` overrides where the remembered fields are stored
        (defaults to ``~/.nemba_gui.json``)."""
        self.root = root
        self.settings_path = settings_path or SETTINGS_PATH
        # 1 logical px = ui physical px (96 dpi reference); dpr as matplotlib computes it
        self.ui = max(1.0, float(root.winfo_fpixels("1i")) / 96.0)
        self.dpr = max(1.0, round(float(root.tk.call("tk", "scaling")) / (96 / 72), 2))

        # job state (main thread only, except the queue and the stop event)
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self.busy = False
        self._job_kind = None
        self._job_t0 = None
        self._indeterminate = False
        self._view_busy = {"embedding": False, "compare": False, "dynamics": False}
        self._view_pending = {"embedding": None, "compare": None, "dynamics": None}
        self.emb = None                 # last fitted Embedder
        self.beh = None                 # its behaviour, aligned on emb.times
        self.circular = []
        self.run_method = "umap"        # projection of the last run (flow field, files)
        self.cosmooth = None            # co-smoothing result of the last run (or None)
        self.cmp = None                 # last Comparison
        self.rec = None                 # last advisor Recommendation
        self._rec_key = None            # (file, input type) the recommendation belongs to
        self._dyn_map = {}              # Dynamics choice -> (kind, behaviour name)
        self._dyn_cache = {}            # Dynamics choice -> (figure, stats rows, caption)
        self._dyn_shown = None          # Dynamics choice on screen
        self._cmp_figs = {}             # Compare tab: "grid" / "consistency" -> Figure
        self._cmp_caption = ""          # caption of the comparison grid
        self._job_note = ""             # extra text for the "finished" status line
        self._states_ok_cache = None    # True once embedders_backbones.states imports
        self._states_error = ""
        self.last_outdir = None
        self.last_status = "idle"
        self.last_elapsed = 0.0
        self.error_count = 0
        self._fields = {}               # name -> widget (focus on validation errors)
        self._color_map = {}            # Embedding colour display -> key
        self._bb_display = {k: f"{k} - {label(k)}" for k in list_backbones()}
        self._bb_key = {v: k for k, v in self._bb_display.items()}

        root.title("NEMBA - neural embedding explorer")
        root.configure(bg=PALETTE["bg"])
        self._init_fonts()
        self._init_style()
        self._init_vars()
        self._apply_settings(load_settings(self.settings_path))
        self._build_layout()
        self._bind_events()
        self._place_window()

        self._on_data_change()
        self._on_backbone()
        self._on_speed()
        self._refresh_state_models()
        self._refresh_explorer_button()
        self._refresh_advice_buttons()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.report_callback_exception = self._report_tk_exception
        self._log(f"NEMBA {_VERSION}  |  {len(list_backbones())} backbones  |  "
                  "leave the data file empty and press Run for the synthetic demo", "head")
        self._poll_id = root.after(POLL_MS, self._poll)

    def px(self, n):
        """Logical pixels -> physical pixels for this screen's DPI."""
        return int(round(n * self.ui))

    def _init_fonts(self):
        """Create the named fonts and make Tk's defaults use the body font."""
        fams = set(tkfont.families(self.root))
        fam = FONT_FAMILY if FONT_FAMILY in fams else tkfont.nametofont("TkDefaultFont").actual("family")
        mono = MONO_FAMILY if MONO_FAMILY in fams else tkfont.nametofont("TkFixedFont").actual("family")
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkTooltipFont"):
            try:
                tkfont.nametofont(name).configure(family=fam, size=BODY_PT)
            except tk.TclError:
                pass
        self.f_body = tkfont.Font(self.root, family=fam, size=BODY_PT)
        self.f_bold = tkfont.Font(self.root, family=fam, size=BODY_PT, weight="bold")
        self.f_title = tkfont.Font(self.root, family=fam, size=TITLE_PT, weight="bold")
        self.f_small = tkfont.Font(self.root, family=fam, size=SMALL_PT)
        self.f_brand = tkfont.Font(self.root, family=fam, size=BRAND_PT, weight="bold")
        self.f_mono = tkfont.Font(self.root, family=mono, size=MONO_PT)
        self.f_mono_bold = tkfont.Font(self.root, family=mono, size=MONO_PT, weight="bold")

    def _init_style(self):
        """Configure the 'clam' theme into a flat, light, accent-blue look."""
        P, px = PALETTE, self.px
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure(".", background=P["panel"], foreground=P["text"], font=self.f_body,
                    bordercolor=P["border"], lightcolor=P["panel"], darkcolor=P["panel"],
                    troughcolor=P["bg"], focuscolor=P["accent"], selectbackground=P["accent"],
                    selectforeground="#ffffff", fieldbackground=P["panel"],
                    insertcolor=P["text"], arrowcolor=P["muted"])
        # frames and labels
        s.configure("App.TFrame", background=P["bg"])
        s.configure("Panel.TFrame", background=P["panel"])
        s.configure("Line.TFrame", background=P["border"])
        s.configure("TLabel", background=P["panel"], foreground=P["text"])
        s.configure("Field.TLabel", foreground=P["text"])
        s.configure("Hint.TLabel", foreground=P["muted"], font=self.f_small)
        s.configure("Dot.TLabel", font=self.f_small)
        s.configure("Brand.TLabel", background=P["bg"], foreground=P["text"], font=self.f_brand)
        s.configure("Sub.TLabel", background=P["bg"], foreground=P["muted"])
        s.configure("Version.TLabel", background=P["bg"], foreground=P["muted"], font=self.f_small)
        s.configure("PaneTitle.TLabel", font=self.f_title)
        s.configure("Caption.TLabel", foreground=P["muted"])
        s.configure("Empty.TLabel", foreground=P["muted"])
        s.configure("Status.TLabel", background=P["panel"], foreground=P["text"])
        s.configure("Elapsed.TLabel", background=P["panel"], foreground=P["muted"])
        # cards: a borderless white Labelframe whose title sits inside the box;
        # the 1 px outline comes from a wrapper frame (see _card)
        s.configure("Card.TLabelframe", background=P["panel"], bordercolor=P["panel"],
                    lightcolor=P["panel"], darkcolor=P["panel"], relief="flat",
                    borderwidth=0, padding=(px(12), px(2), px(12), px(9)),
                    labeloutside=True, labelmargins=(px(12), px(8), 0, px(2)))
        s.configure("Card.TLabelframe.Label", background=P["panel"], foreground=P["text"],
                    font=self.f_title)
        # buttons (no dashed focus ring: the border turns accent on focus instead)
        s.layout("TButton", [("Button.border", {"sticky": "nswe", "border": "1", "children": [
            ("Button.padding", {"sticky": "nswe", "children": [
                ("Button.label", {"sticky": "nswe"})]})]})])
        # relief 'raised' makes clam draw the outline; light/dark = background hides the bevel
        s.configure("TButton", background=P["panel"], foreground=P["text"], bordercolor=P["field"],
                    lightcolor=P["panel"], darkcolor=P["panel"], relief="raised",
                    padding=(px(10), px(5)), anchor="center")
        s.map("TButton",
              background=[("disabled", P["disabled_bg"]), ("pressed", P["border"]),
                          ("active", P["hover"])],
              lightcolor=[("disabled", P["disabled_bg"]), ("pressed", P["border"]),
                          ("active", P["hover"])],
              darkcolor=[("disabled", P["disabled_bg"]), ("pressed", P["border"]),
                         ("active", P["hover"])],
              foreground=[("disabled", P["disabled_fg"])],
              bordercolor=[("disabled", P["border"]), ("focus", P["accent"])])
        s.configure("Small.TButton", padding=(px(8), px(3)), width=-7)
        acc = [("disabled", P["accent_off"]), ("pressed", P["accent_press"]),
               ("active", P["accent_hover"])]
        s.configure("Accent.TButton", background=P["accent"], foreground="#ffffff",
                    bordercolor=P["accent"], lightcolor=P["accent"], darkcolor=P["accent"],
                    font=self.f_bold, padding=(px(12), px(7)))
        s.map("Accent.TButton", background=acc, lightcolor=acc, darkcolor=acc,
              bordercolor=acc + [("focus", P["accent_press"])],
              foreground=[("disabled", "#f2f6fc")])
        danger = [("disabled", P["disabled_bg"]), ("pressed", "#f7d9d5"), ("active", "#fdf0ee")]
        s.configure("Danger.TButton", foreground=P["bad"], bordercolor="#ecc4bf")
        s.map("Danger.TButton", background=danger, lightcolor=danger, darkcolor=danger,
              foreground=[("disabled", P["disabled_fg"])],
              bordercolor=[("disabled", P["border"]), ("focus", P["bad"])])
        # entries and comboboxes
        s.configure("TEntry", fieldbackground=P["panel"], bordercolor=P["field"],
                    lightcolor=P["panel"], darkcolor=P["panel"], padding=(px(6), px(3)))
        s.map("TEntry", bordercolor=[("focus", P["accent"])], lightcolor=[("focus", P["accent_soft"])],
              fieldbackground=[("disabled", P["disabled_bg"])],
              foreground=[("disabled", P["disabled_fg"])])
        s.configure("TCombobox", fieldbackground=P["panel"], background=P["panel"],
                    bordercolor=P["field"], lightcolor=P["panel"], darkcolor=P["panel"],
                    arrowcolor=P["muted"], arrowsize=px(11), padding=(px(6), px(3)))
        s.map("TCombobox",
              fieldbackground=[("disabled", P["disabled_bg"]), ("readonly", P["panel"])],
              background=[("disabled", P["disabled_bg"]), ("active", P["hover"])],
              bordercolor=[("focus", P["accent"]), ("active", "#bdbcb7")],
              lightcolor=[("focus", P["accent_soft"])],
              arrowcolor=[("disabled", P["disabled_fg"])],
              foreground=[("disabled", P["disabled_fg"])],
              selectbackground=[("readonly", P["panel"])],
              selectforeground=[("readonly", P["text"])])
        # the drop-down list of a combobox is a classic Tk listbox
        self.root.option_add("*TCombobox*Listbox.font", self.f_body)
        self.root.option_add("*TCombobox*Listbox.background", P["panel"])
        self.root.option_add("*TCombobox*Listbox.foreground", P["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", P["accent"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        # wide drop-down for the backbone list (field shows 'key - label')
        s.configure("Wide.TCombobox", postoffset=(0, 0, px(90), 0))
        # check buttons: accent square when ticked, no focus ring
        s.layout("TCheckbutton", [("Checkbutton.padding", {"sticky": "nswe", "children": [
            ("Checkbutton.indicator", {"side": "left", "sticky": ""}),
            ("Checkbutton.label", {"side": "left", "sticky": "nswe"})]})])
        s.configure("TCheckbutton", background=P["panel"], indicatorbackground=P["panel"],
                    indicatorforeground="#ffffff", upperbordercolor=P["field"],
                    lowerbordercolor=P["field"], indicatormargin=(0, 0, px(6), 0),
                    indicatorsize=px(12), padding=(0, px(2)))
        s.map("TCheckbutton",
              indicatorbackground=[("disabled", P["disabled_bg"]), ("selected", P["accent"]),
                                   ("active", P["hover"])],
              upperbordercolor=[("selected", P["accent"])],
              lowerbordercolor=[("selected", P["accent"])],
              foreground=[("disabled", P["disabled_fg"])],
              background=[("active", P["panel"])])
        # segmented speed chips (radio buttons drawn as buttons)
        s.layout("Segment.TRadiobutton", [("Button.border", {"sticky": "nswe", "border": "1", "children": [
            ("Button.padding", {"sticky": "nswe", "children": [
                ("Button.label", {"sticky": "nswe"})]})]})])
        s.configure("Segment.TRadiobutton", background=P["panel"], foreground=P["muted"],
                    bordercolor=P["field"], lightcolor=P["panel"], darkcolor=P["panel"],
                    relief="raised", padding=(px(4), px(3)), anchor="center")
        seg_bg = [("selected", "active", P["accent_hover"]), ("selected", P["accent"]),
                  ("active", P["hover"])]
        s.map("Segment.TRadiobutton", background=seg_bg, lightcolor=seg_bg, darkcolor=seg_bg,
              bordercolor=[("selected", P["accent"])],
              foreground=[("selected", "#ffffff"), ("active", P["text"])])
        # notebook
        s.configure("TNotebook", background=P["bg"], borderwidth=1, bordercolor=P["border"],
                    lightcolor=P["panel"], darkcolor=P["panel"], tabmargins=(0, 0, 0, 0))
        s.layout("TNotebook.Tab", [("Notebook.tab", {"sticky": "nswe", "children": [
            ("Notebook.padding", {"side": "top", "sticky": "nswe", "children": [
                ("Notebook.label", {"side": "top", "sticky": ""})]})]})])
        s.configure("TNotebook.Tab", background=P["bg"], foreground=P["muted"],
                    bordercolor=P["border"], lightcolor=P["bg"], darkcolor=P["bg"],
                    padding=(px(16), px(6)), font=self.f_body)
        s.map("TNotebook.Tab",
              background=[("selected", P["panel"]), ("active", P["hover"])],
              lightcolor=[("selected", P["panel"]), ("active", P["hover"])],
              foreground=[("selected", P["text"]), ("active", P["text"])],
              padding=[("selected", (px(16), px(6)))],
              expand=[("selected", (0, 0, 0, 0))])
        # tables
        s.layout("Treeview.Item", [("Treeitem.padding", {"sticky": "nswe", "children": [
            ("Treeitem.indicator", {"side": "left", "sticky": ""}),
            ("Treeitem.image", {"side": "left", "sticky": ""}),
            ("Treeitem.text", {"side": "left", "sticky": ""})]})])
        s.configure("Treeview", background=P["panel"], fieldbackground=P["panel"],
                    foreground=P["text"], bordercolor=P["border"], lightcolor=P["panel"],
                    darkcolor=P["panel"], rowheight=px(24), borderwidth=0, font=self.f_body)
        s.map("Treeview", background=[("selected", P["accent_soft"])],
              foreground=[("selected", P["text"])])
        s.configure("Treeview.Heading", background=P["heading"], foreground=P["muted"],
                    font=self.f_bold, relief="flat", bordercolor=P["border"],
                    lightcolor=P["heading"], darkcolor=P["heading"], padding=(px(6), px(5)))
        s.map("Treeview.Heading", background=[("active", P["hover"])],
              lightcolor=[("active", P["hover"])], darkcolor=[("active", P["hover"])])
        # progress bar and scrollbars
        s.configure("Accent.Horizontal.TProgressbar", troughcolor="#ebeae6", background=P["accent"],
                    bordercolor="#ebeae6", lightcolor=P["accent"], darkcolor=P["accent"],
                    thickness=px(6))
        for name, trough in (("Side", P["bg"]), ("Slim", P["panel"])):
            for orient, o in (("Vertical", "ns"), ("Horizontal", "we")):
                st = f"{name}.{orient}.TScrollbar"
                s.layout(st, [(f"{orient}.Scrollbar.trough", {"sticky": o, "children": [
                    (f"{orient}.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})])
                s.configure(st, troughcolor=trough, background="#d6d5d0", bordercolor=trough,
                            lightcolor="#d6d5d0", darkcolor="#d6d5d0", arrowsize=px(8),
                            gripcount=0, relief="flat")
                s.map(st, background=[("active", "#b9b8b3")],
                      lightcolor=[("active", "#b9b8b3")], darkcolor=[("active", "#b9b8b3")])
        s.configure("TPanedwindow", background=P["panel"])
        s.configure("Sash", sashthickness=px(8), gripcount=0, background=P["panel"],
                    lightcolor=P["panel"], bordercolor=P["panel"])

    def _init_vars(self):
        """Create the Tk variables behind every sidebar field (with defaults)."""
        D = DEFAULTS
        self.var_file = tk.StringVar(value=D["file"])
        self.var_type = tk.StringVar(value=D["input_type"])
        self.var_taxis = tk.StringVar(value=D["time_axis"])
        self.var_bin = tk.StringVar(value=D["bin_size"])
        self.var_fs = tk.StringVar(value=D["fs"])
        self.var_dff = tk.BooleanVar(value=D["dff"])
        self.var_mask_size = tk.StringVar(value=D["mask_size"])
        self.var_mask_key = tk.StringVar(value=D["mask_key"])
        self.var_goal = tk.StringVar(value=D["goal"])
        self.var_beh = tk.StringVar(value=D["beh_file"])
        self.var_circ = tk.StringVar(value=D["circular"])
        self.var_nperm = tk.StringVar(value=D["n_perm"])
        self.var_decoder = tk.StringVar(value=D["decoder"])
        self.var_psid = tk.BooleanVar(value=D["psid"])
        self.var_cosmooth = tk.BooleanVar(value=D["cosmooth"])
        self.var_backbone = tk.StringVar(value=self._bb_display.get(D["backbone"], ""))
        self.var_speed = tk.StringVar(value=D["speed"])
        self.var_objective = tk.StringVar(value=D["objective"])
        self.var_likelihood = tk.StringVar(value=D["likelihood"])
        self.var_mask_mode = tk.StringVar(value=D["mask_mode"])
        self.var_states = tk.StringVar(value=D["n_states"])
        self.var_state_model = tk.StringVar(value=D["state_model"])
        self.var_emb_dim = tk.StringVar(value=D["emb_dim"])
        self.var_max_steps = tk.StringVar(value=D["max_steps"])
        self.var_smooth = tk.StringVar(value=D["smooth"])
        self.var_out = tk.StringVar(value=D["outdir"])
        self.var_gif = tk.BooleanVar(value=D["gif"])
        self.var_html = tk.BooleanVar(value=D["html"])
        self.var_gif_method = tk.StringVar(value=D["gif_method"])
        self.var_compare = tk.StringVar(value=D["compare"])
        # result-view controls
        self.var_view_proj = tk.StringVar(value="umap")
        self.var_view_color = tk.StringVar(value="time")
        self.var_cmp_proj = tk.StringVar(value="umap")
        self.var_cmp_color = tk.StringVar(value="state")
        self.var_cmp_mode = tk.StringVar(value="grid")       # Compare tab: grid | consistency
        self.var_dyn_fig = tk.StringVar(value="Topology")    # Dynamics tab figure
        # status bar
        self.var_status = tk.StringVar(value="Ready")
        self.var_elapsed = tk.StringVar(value="")
        # name in the settings file -> variable (backbone handled separately)
        self._persist = {
            "file": self.var_file, "input_type": self.var_type, "time_axis": self.var_taxis,
            "bin_size": self.var_bin, "fs": self.var_fs, "dff": self.var_dff,
            "mask_size": self.var_mask_size, "mask_key": self.var_mask_key,
            "goal": self.var_goal,
            "beh_file": self.var_beh, "circular": self.var_circ, "n_perm": self.var_nperm,
            "decoder": self.var_decoder, "psid": self.var_psid, "cosmooth": self.var_cosmooth,
            "speed": self.var_speed, "objective": self.var_objective,
            "likelihood": self.var_likelihood, "mask_mode": self.var_mask_mode,
            "n_states": self.var_states, "state_model": self.var_state_model,
            "emb_dim": self.var_emb_dim,
            "max_steps": self.var_max_steps, "smooth": self.var_smooth, "outdir": self.var_out,
            "gif": self.var_gif, "html": self.var_html, "gif_method": self.var_gif_method,
            "compare": self.var_compare,
        }

    # ------------------------------------------------------------- settings --
    def _apply_settings(self, values):
        """Load remembered values, ignoring unknown keys and invalid choices.

        A remembered 'hmm' / 'arhmm' state model is kept only when the
        temporal state models can be imported (else the default stays)."""
        choices = {"input_type": INPUT_TYPES, "time_axis": TIME_AXES, "speed": SPEEDS,
                   "objective": list(OBJECTIVES), "gif_method": PROJECTIONS, "goal": GOALS,
                   "likelihood": LIKELIHOODS, "mask_mode": MASK_CHOICES,
                   "state_model": STATE_MODELS, "decoder": DECODERS}
        for key, var in self._persist.items():
            if key not in values:
                continue
            v = values[key]
            try:
                if isinstance(var, tk.BooleanVar):
                    if isinstance(v, bool):
                        var.set(v)
                elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
                    v = str(v)
                    if key in choices and v not in choices[key]:
                        continue
                    if key == "state_model" and v in ("hmm", "arhmm") and not self._states_ok():
                        continue
                    var.set(v)
            except (tk.TclError, ValueError):
                pass
        bb = values.get("backbone")
        if isinstance(bb, str) and bb in self._bb_display:
            self.var_backbone.set(self._bb_display[bb])

    def _settings_snapshot(self):
        """Current field values as a JSON-serialisable dict."""
        out = {}
        for key, var in self._persist.items():
            try:
                out[key] = var.get()
            except tk.TclError:
                pass
        out["backbone"] = self._bb_key.get(self.var_backbone.get(), DEFAULTS["backbone"])
        return out

    def _save_settings(self):
        """Persist the current fields (silently ignores I/O errors)."""
        save_settings(self.settings_path, self._settings_snapshot())

    # --------------------------------------------------------------- layout --
    def _place_window(self):
        """Size the window (scaled for DPI, clipped to the screen) and centre it."""
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w = min(self.px(WINDOW_SIZE[0]), sw - self.px(40))
        h = min(self.px(WINDOW_SIZE[1]), sh - self.px(90))
        x, y = max((sw - w) // 2, 0), max((sh - h) // 2 - self.px(20), 0)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(min(self.px(MIN_SIZE[0]), w), min(self.px(MIN_SIZE[1]), h))

    def _build_layout(self):
        """Root grid: [sidebar | 1 px line | main tabs] above the status bar."""
        body = ttk.Frame(self.root, style="App.TFrame")
        body.grid(row=0, column=0, sticky="nsew")
        ttk.Frame(self.root, style="Line.TFrame", height=1).grid(row=1, column=0, sticky="ew")
        self._build_statusbar(self.root).grid(row=2, column=0, sticky="ew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self._build_sidebar(body).grid(row=0, column=0, sticky="ns")
        ttk.Frame(body, style="Line.TFrame", width=1).grid(row=0, column=1, sticky="ns")
        self._build_main(body).grid(row=0, column=2, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(2, weight=1)

    # ----- sidebar ---------------------------------------------------------- #
    def _build_sidebar(self, parent):
        """Brand header, scrollable form cards and the fixed action footer."""
        px, P = self.px, PALETTE
        side = ttk.Frame(parent, style="App.TFrame", width=px(SIDEBAR_W))
        side.pack_propagate(False)

        head = ttk.Frame(side, style="App.TFrame")
        head.pack(side="top", fill="x", padx=px(16), pady=(px(12), px(8)))
        logo = tk.Canvas(head, width=px(26), height=px(26), bg=P["bg"], highlightthickness=0)
        self._draw_logo(logo, px(26))
        logo.pack(side="left", padx=(0, px(9)))
        ttk.Label(head, text="NEMBA", style="Brand.TLabel").pack(side="left")
        ttk.Label(head, text="neural embedding explorer", style="Sub.TLabel").pack(
            side="left", padx=(px(8), 0), pady=(px(5), 0))
        ttk.Label(head, text=f"v{_VERSION}" if _VERSION else "", style="Version.TLabel").pack(
            side="right", pady=(px(5), 0))

        foot = self._build_actions(side)
        foot.pack(side="bottom", fill="x", padx=px(14), pady=(px(10), px(14)))
        ttk.Frame(side, style="Line.TFrame", height=1).pack(side="bottom", fill="x")

        self.sidebar = ScrollableFrame(side, self)
        self.sidebar.pack(side="top", fill="both", expand=True)
        form = self.sidebar.inner
        self._build_data_card(form)
        self._build_behaviour_card(form)
        self._build_model_card(form)
        self._build_output_card(form)
        return side

    def _draw_logo(self, cv, size):
        """Tiny logo: a ring of dots shaded along the NEMBA blue ramp."""
        ramp = viz.BLUE_RAMP
        n, c, r, d = 12, size / 2, size * 0.36, max(size * 0.075, 1.5)
        for i in range(n):
            a = 2 * np.pi * i / n - np.pi / 2
            x, y = c + r * np.cos(a), c + r * np.sin(a)
            col = ramp[min(int(i / n * len(ramp)) + 1, len(ramp) - 1)]
            cv.create_oval(x - d, y - d, x + d, y + d, fill=col, outline="")
        cv.create_oval(c - d * 0.9, c - d * 0.9, c + d * 0.9, c + d * 0.9,
                       fill=PALETTE["accent"], outline="")

    def _card(self, parent, title):
        """A titled white section: ttk.Labelframe inside a 1 px outline frame.

        The Labelframe draws its title inside the white box (no border of its
        own); the wrapper's background shows through its 1 px padding as the
        card outline. Children use a 2-column grid (label | field)."""
        px = self.px
        outline = tk.Frame(parent, bg=PALETTE["border"], bd=0, highlightthickness=0)
        outline.pack(fill="x", padx=(px(14), px(6)), pady=(0, px(9)))
        card = ttk.Labelframe(outline, text=title, style="Card.TLabelframe")
        card.pack(fill="both", expand=True, padx=1, pady=1)
        card.columnconfigure(0, minsize=px(LABEL_COL_W))
        card.columnconfigure(1, weight=1)
        return card

    def _row(self, card, row, text, widget, button=None, tip=None, name=None):
        """Place 'label | field [| button]' on ``row`` of a card.

        Returns the widgets of the row (used to show / hide optional rows)."""
        px = self.px
        pad = px(2.5)
        lbl = ttk.Label(card, text=text, style="Field.TLabel")
        lbl.grid(row=row, column=0, sticky="w", padx=(0, px(8)), pady=pad)
        widget.grid(row=row, column=1, columnspan=1 if button else 2, sticky="ew", pady=pad)
        if button is not None:
            button.grid(row=row, column=2, sticky="ew", padx=(px(6), 0), pady=pad)
        if tip:
            Tooltip(lbl, tip, self)
            Tooltip(widget, tip, self)
        if name:
            self._fields[name] = widget
        return [w for w in (lbl, widget, button) if w is not None]

    def _hint(self, card, row, text="", **kw):
        """Small muted explanatory text under a field (field column)."""
        lbl = ttk.Label(card, text=text, style="Hint.TLabel", justify="left",
                        wraplength=self.px(SIDEBAR_W - LABEL_COL_W - 54), **kw)
        lbl.grid(row=row, column=1, columnspan=2, sticky="w", pady=(0, self.px(3)))
        return lbl

    def _entry(self, card, var, width=8):
        """A ttk.Entry bound to ``var``."""
        return ttk.Entry(card, textvariable=var, width=width)

    def _combo(self, card, var, values, readonly=True, width=8, style="TCombobox"):
        """A ttk.Combobox bound to ``var``."""
        return ttk.Combobox(card, textvariable=var, values=values, width=width, style=style,
                            state="readonly" if readonly else "normal")

    def _build_data_card(self, form):
        """Data section: file, input type and the options that type needs.

        Optional rows (time axis, bin size, fs, dF/F, mask raster / key) are
        shown only for the input types that use them (see ``DATA_ROWS``).
        The last row holds the advisor: goal selector and 'Analyse data'."""
        c = self._card(form, "Data")
        e = self._entry(c, self.var_file)
        b = ttk.Button(c, text="Browse", style="Small.TButton", command=self._browse_data)
        self._row(c, 0, "File", e, b, name="file",
                  tip="Recording (.npy / .npz / .csv / .pkl, or a COCO .json for masks). "
                      "Leave empty for the synthetic demo.")
        self.hint_file = self._hint(c, 1, DEMO_HINT)
        self.cb_type = self._combo(c, self.var_type, INPUT_TYPES)
        self._row(c, 2, "Input type", self.cb_type, name="input_type",
                  tip="auto = recognised from the file (spike-time arrays, COCO json, "
                      "DeepLabCut CSV, tables with a time column, matrices with the time axis "
                      "detected or taken from the advisor); spikes = list of spike-time arrays; "
                      "pose / mask / timeseries = matrix; fiber_photometry = signals (optional "
                      "dF/F); coco_mask = COCO json.")
        self.cb_taxis = self._combo(c, self.var_taxis, TIME_AXES)
        self.en_bin = self._entry(c, self.var_bin)
        self.en_fs = self._entry(c, self.var_fs)
        self.ck_dff = ttk.Checkbutton(c, text="compute dF/F", variable=self.var_dff)
        self.en_mask = self._entry(c, self.var_mask_size)
        self.en_mask_key = self._entry(c, self.var_mask_key)
        # optional rows: key -> (widgets of the row, the field widget to enable)
        self._data_rows = {
            "axis": (self._row(c, 3, "Time axis", self.cb_taxis, name="time_axis",
                               tip="Which array axis is time (1 = columns: the usual "
                                   "channels x time layout)."), self.cb_taxis),
            "bin": (self._row(c, 4, "Bin size (s)", self.en_bin, name="bin_size",
                              tip="Spike-count bin width in seconds (spike input only)."),
                    self.en_bin),
            "fs": (self._row(c, 5, "fs (Hz)", self.en_fs, name="fs",
                             tip="Sampling rate: needed for dF/F of fiber photometry, frame "
                                 "rate of COCO mask videos. Blank = unknown."), self.en_fs),
            "mask": (self._row(c, 7, "Mask raster", self.en_mask, name="mask_size",
                               tip="COCO masks are rasterised on an N x N grid (N*N channels, "
                                   "reduced by input PCA)."), self.en_mask),
            "mask_key": (self._row(c, 8, "Mask key", self.en_mask_key, name="mask_key",
                                   tip="COCO annotation key splitting masks into channels "
                                       "(e.g. track_id). Blank = one channel set."),
                         self.en_mask_key),
        }
        self.ck_dff.grid(row=6, column=1, columnspan=2, sticky="w", pady=self.px(2.5))
        Tooltip(self.ck_dff, "Convert raw fluorescence to dF/F (sliding baseline, needs fs).", self)
        self._data_rows["dff"] = ([self.ck_dff], self.ck_dff)

        # data advisor: goal + "Analyse data" share the field and button columns
        px = self.px
        box = ttk.Frame(c, style="Panel.TFrame")
        self.cb_goal = ttk.Combobox(box, textvariable=self.var_goal, values=GOALS, width=8,
                                    state="readonly")
        self.cb_goal.pack(side="left", fill="x", expand=True)
        self.btn_analyse = ttk.Button(box, text="Analyse data", style="Small.TButton",
                                      command=self.analyse)
        self.btn_analyse.pack(side="left", padx=(px(6), 0))
        self._row(c, 9, "Goal", box, name="goal",
                  tip="What the advisor optimises the suggestions for: auto (decided from the "
                      "data), decode (behaviour read-out), states (discrete states) or "
                      "visualize (geometry and trajectories).")
        self._fields["goal"] = self.cb_goal
        Tooltip(self.btn_analyse, "Profile the recording (value type, timescales, dimension, "
                                  "ring / cluster structure, behaviour) and suggest a pipeline "
                                  "(Advisor tab). Writes advice.md / .json and data_profile "
                                  "figures to the output folder.", self)
        self._hint(c, 10, "Suggests settings in the Advisor tab.")

    def _build_behaviour_card(self, form):
        """Behaviour section: optional file, circular names, null shuffles,
        decoder, PSID rows and the (label-free) co-smoothing score."""
        c = self._card(form, "Behaviour")
        e = self._entry(c, self.var_beh)
        b = ttk.Button(c, text="Browse", style="Small.TButton", command=self._browse_behaviour)
        self._row(c, 0, "File", e, b, name="beh_file",
                  tip="Variables to decode and colour by: .csv (header row), .npz (keys) or .npy. "
                      "A 'time' column resamples them onto the embedding frames.")
        self._hint(c, 1, "Optional; the demo brings its own.")
        self._row(c, 2, "Circular", self._entry(c, self.var_circ), name="circular",
                  tip="Comma list of angular variables in radians (decoded via sin / cos, "
                      "drawn with a cyclic colour map).")
        self._row(c, 3, "Shuffles", self._entry(c, self.var_nperm), name="n_perm",
                  tip="Circular-shift permutations of the decoding null distribution "
                      "(0 = no significance test).")
        self._row(c, 4, "Decoder", self._combo(c, self.var_decoder, DECODERS), name="decoder",
                  tip="linear = ridge regression / shrinkage LDA (default, 'explicitly "
                      "represented'); knn = local non-linear read-out; gbm = gradient-boosted "
                      "trees (strongest, slowest, null capped at 49 shifts); logistic = "
                      "logistic regression for discrete targets (ridge for continuous ones).")
        px = self.px
        # stacked: side by side the two labels would touch the edge of the card
        box = ttk.Frame(c, style="Panel.TFrame")
        box.grid(row=5, column=1, columnspan=2, sticky="w", pady=(px(2), px(1)))
        self.ck_psid = ttk.Checkbutton(box, text="PSID rows", variable=self.var_psid)
        self.ck_psid.pack(side="top", anchor="w")
        self.ck_cosmooth = ttk.Checkbutton(box, text="Co-smoothing score",
                                           variable=self.var_cosmooth)
        self.ck_cosmooth.pack(side="top", anchor="w", pady=(px(2), 0))
        if importlib.util.find_spec("PSID") is None:
            self.var_psid.set(False)
            self.ck_psid.state(["disabled"])
            Tooltip(self.ck_psid, "Unavailable: needs the PSID package (pip install PSID).", self)
        else:
            Tooltip(self.ck_psid, "Add cross-fitted PSID decoding rows (Sani et al. 2021: "
                                  "behaviourally relevant dynamics from the raw input) for "
                                  "continuous and circular targets. No permutation null.", self)
        Tooltip(self.ck_cosmooth, "After fitting, score the embedding by co-smoothing: bits per "
                                  "spike predicted for held-out neurons on held-out time (Neural "
                                  "Latents Benchmark). Label-free; needs raw counts (spikes or "
                                  "integer data); refits the backbone once.", self)

    def _build_model_card(self, form):
        """Model section: backbone (+ CPU hint), speed chips, objective,
        likelihood, masking, K, state model, embedding size, steps, smoothing."""
        px = self.px
        c = self._card(form, "Model")
        self.cb_backbone = self._combo(c, self.var_backbone, list(self._bb_display.values()),
                                       style="Wide.TCombobox")
        self._row(c, 0, "Backbone", self.cb_backbone, name="backbone",
                  tip="Sequence encoder. 'pca' is the no-training baseline: keep it as a "
                      "reference in comparisons.")
        hint = ttk.Frame(c, style="Panel.TFrame")
        hint.grid(row=1, column=1, columnspan=2, sticky="w", pady=(0, px(3)))
        self.lbl_cost_dot = ttk.Label(hint, text="●", style="Dot.TLabel")
        self.lbl_cost_dot.pack(side="left")
        self.lbl_cost = ttk.Label(hint, text="", style="Hint.TLabel",
                                  wraplength=px(SIDEBAR_W - LABEL_COL_W - 70))
        self.lbl_cost.pack(side="left", padx=(px(3), 0))

        seg = ttk.Frame(c, style="Panel.TFrame")
        for i, sp in enumerate(SPEEDS):
            rb = ttk.Radiobutton(seg, text=sp.capitalize(), value=sp, variable=self.var_speed,
                                 style="Segment.TRadiobutton", command=self._on_speed,
                                 takefocus=False)
            rb.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else px(4), 0))
            seg.columnconfigure(i, weight=1, uniform="seg")
        self._row(c, 2, "Speed", seg, name="speed",
                  tip="Training budget preset (steps, window, width, time cap). "
                      "Fast suits CPU laptops.")
        self.lbl_speed = self._hint(c, 3)
        self._row(c, 4, "Objective", self._combo(c, self.var_objective, list(OBJECTIVES)),
                  name="objective",
                  tip="masked = reconstruct hidden frames; contrastive = time-contrastive "
                      "InfoNCE; hybrid = both.")
        self._row(c, 5, "Likelihood", self._combo(c, self.var_likelihood, LIKELIHOODS),
                  name="likelihood",
                  tip="Reconstruction loss of the masked objective. gaussian = squared error "
                      "on the scaled input; poisson = Poisson log-likelihood on raw counts "
                      "(spikes or integer data only; disables input PCA); auto = poisson when "
                      "raw counts exist, else gaussian.")
        self._row(c, 6, "Masking", self._combo(c, self.var_mask_mode, MASK_CHOICES),
                  name="mask_mode",
                  tip="What the masked objective hides: time = spans of frames; neuron = "
                      "whole channels (learns population structure); both = alternate.")
        self._row(c, 7, "States", self._combo(c, self.var_states,
                                              ["auto"] + [str(k) for k in range(2, 11)],
                                              readonly=False),
                  name="n_states",
                  tip="Number of discrete states K. 'auto' picks K in 2-10 by silhouette "
                      "(or by held-out likelihood for hmm / arhmm).")
        self.cb_state_model = ttk.Combobox(c, textvariable=self.var_state_model,
                                           values=STATE_MODELS, width=8, state="readonly",
                                           postcommand=self._refresh_state_models)
        widgets = self._row(c, 8, "State model", self.cb_state_model, name="state_model")
        self._state_tips = [Tooltip(w, "", self) for w in widgets]
        self._row(c, 9, "Embedding dim", self._combo(c, self.var_emb_dim, EMB_DIMS,
                                                     readonly=False),
                  name="emb_dim",
                  tip="Size of the embedding. 'auto' = backbone default (32; 8 for gpfa and "
                      "cebra). The advisor suggests about twice the number of PCs that hold "
                      "90 % of the variance.")
        self._row(c, 10, "Max steps", self._entry(c, self.var_max_steps), name="max_steps",
                  tip="Override the preset step budget (blank = preset). Early stopping may "
                      "end training sooner.")
        self._row(c, 11, "Smoothing", self._entry(c, self.var_smooth), name="smooth",
                  tip="Gaussian smoothing of the embedding, sigma in frames (0 = off).")

    def _build_output_card(self, form):
        """Output section: folder, projection, GIF / HTML ticks, open buttons."""
        px = self.px
        c = self._card(form, "Output")
        e = self._entry(c, self.var_out)
        b = ttk.Button(c, text="Browse", style="Small.TButton", command=self._browse_out)
        self._row(c, 0, "Folder", e, b, name="outdir",
                  tip="Figures (png / pdf / svg), CSV tables, GIF and HTML explorer go here.")
        self._row(c, 1, "Projection", self._combo(c, self.var_gif_method, PROJECTIONS),
                  name="gif_method",
                  tip="Projection used by the trajectory GIF, the HTML explorer (2-D map and "
                      "its 3-D view; t-SNE falls back to PCA in 3-D) and the flow field. "
                      "tphate = time-aware T-PHATE (about 5-10 s for 3000 frames).")
        # stacked: with the sidebar scrollbar shown, side by side they would be clipped
        box = ttk.Frame(c, style="Panel.TFrame")
        box.grid(row=2, column=1, columnspan=2, sticky="w", pady=(px(2), px(2)))
        ttk.Checkbutton(box, text="Trajectory GIF", variable=self.var_gif).pack(side="top",
                                                                                anchor="w")
        ck_html = ttk.Checkbutton(box, text="HTML explorer (2-D / 3-D)", variable=self.var_html)
        ck_html.pack(side="top", anchor="w", pady=(px(2), 0))
        Tooltip(ck_html, "Self-contained interactive page: hover, colour switch, playback, "
                         "light / dark theme and a 3-D view (drag to rotate, wheel to zoom). "
                         "Views can be shared with the URL hash, e.g. "
                         "#color=state&t=0.45&view=3d.", self)
        row = ttk.Frame(c, style="Panel.TFrame")
        row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(px(6), 0))
        row.columnconfigure((0, 1), weight=1, uniform="open")
        self.btn_open = ttk.Button(row, text="Open folder", style="Small.TButton",
                                   command=self.open_output)
        self.btn_open.grid(row=0, column=0, sticky="ew", padx=(0, px(3)))
        self.btn_html = ttk.Button(row, text="Open explorer", style="Small.TButton",
                                   command=self.open_explorer)
        self.btn_html.grid(row=0, column=1, sticky="ew", padx=(px(3), 0))
        Tooltip(self.btn_open, "Open the last output folder in the file manager.", self)
        Tooltip(self.btn_html, "Open the latest interactive *_explorer.html in the browser.", self)

    def _build_actions(self, parent):
        """Fixed footer: Run + Stop, then the Compare list + Compare button."""
        px = self.px
        f = ttk.Frame(parent, style="App.TFrame")
        f.columnconfigure(0, weight=1)
        self.btn_run = ttk.Button(f, text="Run", style="Accent.TButton", command=self.run)
        self.btn_run.grid(row=0, column=0, sticky="ew")
        self.btn_stop = ttk.Button(f, text="Stop", style="Danger.TButton", command=self.stop,
                                   width=9)
        self.btn_stop.grid(row=0, column=1, sticky="nsew", padx=(px(6), 0))
        cmp_entry = ttk.Entry(f, textvariable=self.var_compare, width=8)
        cmp_entry.grid(row=1, column=0, sticky="nsew", pady=(px(8), 0))
        self._fields["compare"] = cmp_entry
        self.btn_cmp = ttk.Button(f, text="Compare", command=self.compare, width=9)
        self.btn_cmp.grid(row=1, column=1, sticky="nsew", padx=(px(6), 0), pady=(px(8), 0))
        self.btn_stop.state(["disabled"])
        Tooltip(self.btn_run, "Fit the selected backbone and write all outputs (Ctrl+R / F5).", self)
        Tooltip(cmp_entry, "Backbones to compare (comma list) on the same data; keep pca as "
                           "the no-training baseline. Outputs go to <folder>/compare.", self)
        Tooltip(self.btn_cmp, "Fit every listed backbone on the same data and compare them.", self)
        Tooltip(self.btn_stop, "Cancel the running job at the next checkpoint (Esc).", self)
        return f

    # ----- main area -------------------------------------------------------- #
    def _build_main(self, parent):
        """Notebook with the result tabs, in workflow order (advice first)."""
        px = self.px
        main = ttk.Frame(parent, style="App.TFrame", padding=(px(14), px(12), px(14), px(10)))
        self.nb = ttk.Notebook(main)
        self.nb.pack(fill="both", expand=True)
        self.panes = {}
        empty_run = ("No results yet.\nPress Run (Ctrl+R). With an empty data file the "
                     "synthetic demo is used.")
        self.tab_advisor = self._build_advisor_tab(self.nb)
        self.nb.add(self.tab_advisor, text="Advisor")
        self.panes["overview"] = FigurePane(self.nb, self, "Dashboard", empty_run,
                                            SCALE_CAP["overview"])
        self.nb.add(self.panes["overview"], text="Overview")
        self.panes["embedding"] = FigurePane(self.nb, self, "Embedding map", empty_run,
                                             SCALE_CAP["embedding"])
        self._build_embedding_controls(self.panes["embedding"].controls)
        self.nb.add(self.panes["embedding"], text="Embedding")
        self.tab_decoding = self._build_decoding_tab(self.nb)
        self.nb.add(self.tab_decoding, text="Decoding")
        self.tab_dynamics = self._build_dynamics_tab(self.nb)
        self.nb.add(self.tab_dynamics, text="Dynamics")
        self.panes["training"] = FigurePane(self.nb, self, "Training curves", empty_run,
                                            SCALE_CAP["training"])
        self.nb.add(self.panes["training"], text="Training")
        self.tab_compare = self._build_compare_tab(self.nb)
        self.nb.add(self.tab_compare, text="Compare")
        self.tab_log = self._build_log_tab(self.nb)
        self.nb.add(self.tab_log, text="Log")
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        return main

    def _build_advisor_tab(self, nb):
        """Advisor tab: data-profile figure (top) and the formatted advice (bottom)."""
        px = self.px
        pw = ttk.Panedwindow(nb, orient="vertical")
        self.panes["advisor"] = FigurePane(
            pw, self, "Data profile",
            "No analysis yet.\nChoose a goal in the Data card and press 'Analyse data' to "
            "profile the recording (empty file = synthetic demo) and get a suggested pipeline.",
            SCALE_CAP["advisor"], min_h=330)
        pw.add(self.panes["advisor"], weight=3)
        box = ttk.Frame(pw, style="Panel.TFrame")
        head = ttk.Frame(box, style="Panel.TFrame")
        head.pack(fill="x", padx=px(16), pady=(px(10), px(6)))
        ttk.Label(head, text="Recommendation", style="PaneTitle.TLabel").pack(side="left")
        self.btn_run_suggested = ttk.Button(head, text="Run suggested", style="Accent.TButton",
                                            command=self.run_suggested)
        self.btn_run_suggested.pack(side="right")
        self.btn_apply = ttk.Button(head, text="Apply suggestions", command=self.apply_suggestions)
        self.btn_apply.pack(side="right", padx=(0, px(8)))
        self.lbl_advice = ttk.Label(head, text="rule-based suggestions from measured properties",
                                    style="Caption.TLabel")
        self.lbl_advice.pack(side="left", padx=(px(12), px(12)))
        Tooltip(self.btn_apply, "Copy the advice into the sidebar: backbone, speed, objective, "
                                "likelihood, masking, states, state model, smoothing, embedding "
                                "size, decoder, shuffles, circular variables, Compare list (and "
                                "bin size / input type for a file). Changes are listed in the "
                                "Log.", self)
        Tooltip(self.btn_run_suggested, "Apply the suggestions, then Run. Also uses the advice "
                                        "that has no sidebar field (training window, input PCA, "
                                        "the demo's bin size).", self)
        ttk.Frame(box, style="Line.TFrame", height=1).pack(fill="x")
        self.report = ReportView(box, self, "The report appears here after 'Analyse data'.")
        self.report.pack(fill="both", expand=True)
        pw.add(box, weight=2)
        return pw

    def _build_dynamics_tab(self, nb):
        """Dynamics tab: selectable figure (top), its statistics and a reading guide."""
        px = self.px
        pw = ttk.Panedwindow(nb, orient="vertical")
        pane = FigurePane(pw, self, "Latent dynamics",
                          "No dynamics yet.\nRun a model: topology, flow field, recurrence, "
                          "raster and state contingency figures are drawn here on demand.",
                          SCALE_CAP["dynamics"], min_h=420)
        self.panes["dynamics"] = pane
        ttk.Label(pane.controls, text="Figure", style="Caption.TLabel").pack(side="left")
        self.cb_dyn = ttk.Combobox(pane.controls, textvariable=self.var_dyn_fig, values=[],
                                   width=20, state="readonly")
        self.cb_dyn.pack(side="left", padx=(px(6), 0))
        self.cb_dyn.bind("<<ComboboxSelected>>", self._on_dyn_change)
        self.cb_dyn.state(["disabled"])
        pw.add(pane, weight=4)
        box = ttk.Frame(pw, style="Panel.TFrame")
        left = ttk.Frame(box, style="Panel.TFrame")
        right = ttk.Frame(box, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(px(16), px(8)), pady=(px(10), px(12)))
        right.grid(row=0, column=1, sticky="nsew", padx=(px(8), px(16)), pady=(px(10), px(12)))
        box.columnconfigure(0, weight=1, uniform="dyn")      # table and guide share the width
        box.columnconfigure(1, weight=1, uniform="dyn")
        box.rowconfigure(0, weight=1)
        ttk.Label(left, text="Statistics", style="PaneTitle.TLabel").pack(anchor="w",
                                                                          pady=(0, px(6)))
        f, self.tree_dyn = self._make_tree(left, [("stat", "statistic", 120, "w"),
                                                  ("value", "value", 80, "e")], height=4)
        f.pack(fill="both", expand=True)
        ttk.Label(right, text="How to read it", style="PaneTitle.TLabel").pack(anchor="w",
                                                                               pady=(0, px(6)))
        self.lbl_dyn_help = ttk.Label(right, text="", style="Caption.TLabel", justify="left",
                                      wraplength=px(420))
        self.lbl_dyn_help.pack(anchor="nw", fill="x")
        # keep the reading guide wrapped to the width of its column
        right.bind("<Configure>", lambda e: self.lbl_dyn_help.configure(
            wraplength=max(e.width - px(4), px(160))))
        pw.add(box, weight=1)
        return pw

    def _view_controls(self, parent, var_proj, var_color, colors, callback):
        """'Projection [..]  Colour by [..]' pair used in figure headers."""
        px = self.px
        ttk.Label(parent, text="Projection", style="Caption.TLabel").pack(side="left")
        cb_p = ttk.Combobox(parent, textvariable=var_proj, values=PROJECTIONS, width=7,
                            state="readonly")
        cb_p.pack(side="left", padx=(px(6), px(14)))
        ttk.Label(parent, text="Colour by", style="Caption.TLabel").pack(side="left")
        cb_c = ttk.Combobox(parent, textvariable=var_color, values=colors, width=18,
                            state="readonly")
        cb_c.pack(side="left", padx=(px(6), 0))
        for cb in (cb_p, cb_c):
            cb.bind("<<ComboboxSelected>>", callback)
            cb.state(["disabled"])
        return cb_p, cb_c

    def _build_embedding_controls(self, parent):
        """Projection / colour selectors of the Embedding tab."""
        self.cb_view_proj, self.cb_view_color = self._view_controls(
            parent, self.var_view_proj, self.var_view_color, ["time", "model states"],
            self._on_view_change)

    def _build_decoding_tab(self, nb):
        """Decoding tab: figure (top) and the score table (bottom), resizable."""
        px = self.px
        pw = ttk.Panedwindow(nb, orient="vertical")
        self.panes["decoding"] = FigurePane(
            pw, self, "Held-out decoding",
            "No decoding yet.\nRun the demo, or add a behaviour file to decode it from the "
            "embedding (held-out, blocked cross-validation).", SCALE_CAP["decoding"],
            min_h=430)
        pw.add(self.panes["decoding"], weight=4)
        box = ttk.Frame(pw, style="Panel.TFrame")
        head = ttk.Frame(box, style="Panel.TFrame")
        head.pack(fill="x", padx=px(16), pady=(px(10), px(6)))
        ttk.Label(head, text="Scores", style="PaneTitle.TLabel").pack(side="left")
        ttk.Label(head, text="held-out score vs circular-shift null (95th percentile); "
                             "p Holm-corrected; grey rows are not significant",
                  style="Caption.TLabel").pack(side="left", padx=(px(12), 0))
        cols = [("target", "target", 120, "w"), ("features", "features", 110, "w"),
                ("metric", "metric", 160, "w"), ("score", "score", 90, "e"),
                ("null_p95", "null 95%", 90, "e"), ("p_holm", "p (Holm)", 90, "e")]
        frame, self.tree_decoding = self._make_tree(box, cols, height=6)
        frame.pack(fill="both", expand=True, padx=px(16), pady=(0, px(12)))
        pw.add(box, weight=1)
        return pw

    def _build_compare_tab(self, nb):
        """Compare tab: embedding grid (top), metric + decoding tables (bottom)."""
        px = self.px
        pw = ttk.Panedwindow(nb, orient="vertical")
        self.panes["compare"] = FigurePane(
            pw, self, "Backbone comparison",
            "No comparison yet.\nList backbones next to the Compare button and press it.",
            SCALE_CAP["compare"], min_h=400)
        # segmented toggle: embedding grid | consistency matrix (packed before the selectors)
        seg = ttk.Frame(self.panes["compare"].controls, style="Panel.TFrame")
        seg.pack(side="left", padx=(0, px(16)))
        self.rb_cmp_mode = {}
        for i, (val, text) in enumerate((("grid", "Embeddings"), ("consistency", "Consistency"))):
            rb = ttk.Radiobutton(seg, text=text, value=val, variable=self.var_cmp_mode,
                                 style="Segment.TRadiobutton", command=self._on_cmp_mode,
                                 takefocus=False, width=11)
            rb.grid(row=0, column=i, padx=(0 if i == 0 else px(3), 0))
            rb.state(["disabled"])
            self.rb_cmp_mode[val] = rb
        Tooltip(self.rb_cmp_mode["consistency"], "Pairwise linear consistency: cross-validated "
                                                 "R² of a linear map from one embedding "
                                                 "(row) to another (column), as in CEBRA.", self)
        self.cb_cmp_proj, self.cb_cmp_color = self._view_controls(
            self.panes["compare"].controls, self.var_cmp_proj, self.var_cmp_color,
            ["state", "time"], self._on_cmp_view_change)
        pw.add(self.panes["compare"], weight=3)
        box = ttk.Frame(pw, style="Panel.TFrame")
        left = ttk.Frame(box, style="Panel.TFrame")
        right = ttk.Frame(box, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(px(16), px(8)), pady=(px(10), px(12)))
        right.grid(row=0, column=1, sticky="nsew", padx=(px(8), px(16)), pady=(px(10), px(12)))
        box.columnconfigure(0, weight=3, uniform="cmp")
        box.columnconfigure(1, weight=2, uniform="cmp")
        box.rowconfigure(0, weight=1)
        ttk.Label(left, text="Metrics", style="PaneTitle.TLabel").pack(anchor="w", pady=(0, px(6)))
        f1, self.tree_cmp_metrics = self._make_tree(left, [("backbone", "backbone", 110, "w")],
                                                     height=4, xscroll=True)
        f1.pack(fill="both", expand=True)
        ttk.Label(right, text="Held-out decoding (embedding)", style="PaneTitle.TLabel").pack(
            anchor="w", pady=(0, px(6)))
        f2, self.tree_cmp_decoding = self._make_tree(right, [("backbone", "backbone", 110, "w")],
                                                      height=4, xscroll=True)
        f2.pack(fill="both", expand=True)
        pw.add(box, weight=1)
        return pw

    def _build_log_tab(self, nb):
        """Log tab: monospace read-only text with coloured tags."""
        px, P = self.px, PALETTE
        tab = ttk.Frame(nb, style="Panel.TFrame")
        head = ttk.Frame(tab, style="Panel.TFrame")
        head.pack(fill="x", padx=px(16), pady=(px(12), px(6)))
        ttk.Label(head, text="Log", style="PaneTitle.TLabel").pack(side="left")
        ttk.Label(head, text="training, decoding and file output messages",
                  style="Caption.TLabel").pack(side="left", padx=(px(12), 0))
        ttk.Button(head, text="Clear", style="Small.TButton", command=self._clear_log).pack(
            side="right")
        ttk.Frame(tab, style="Line.TFrame", height=1).pack(fill="x")
        body = ttk.Frame(tab, style="Panel.TFrame")
        body.pack(fill="both", expand=True)
        self.log_text = tk.Text(body, font=self.f_mono, bg=P["panel"], fg=P["text"],
                                relief="flat", bd=0, wrap="word", padx=px(16), pady=px(10),
                                highlightthickness=0, selectbackground=P["accent_soft"],
                                selectforeground=P["text"], insertbackground=P["text"],
                                spacing1=px(1), spacing3=px(1), state="disabled")
        vsb = ttk.Scrollbar(body, orient="vertical", command=self.log_text.yview,
                            style="Slim.Vertical.TScrollbar")
        self.log_text.configure(yscrollcommand=self._autohide(vsb))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self.log_text.tag_configure("err", foreground=P["bad"])
        self.log_text.tag_configure("warn", foreground=P["warn"])
        self.log_text.tag_configure("head", foreground=P["accent"], font=self.f_mono_bold)
        self.log_text.tag_configure("ok", foreground=P["ok"])
        self.log_text.tag_configure("muted", foreground=P["muted"])
        return tab

    def _make_tree(self, parent, columns, height=6, xscroll=False):
        """Striped, sortable Treeview with slim scrollbars.

        ``columns`` = [(id, heading, width, anchor)]. Returns (frame, tree)."""
        px = self.px
        frame = ttk.Frame(parent, style="Panel.TFrame")
        border = tk.Frame(frame, bg=PALETTE["border"], bd=0, highlightthickness=0)
        border.grid(row=0, column=0, sticky="nsew")
        tree = ttk.Treeview(border, columns=[c[0] for c in columns], show="headings",
                            height=height, selectmode="browse")
        tree.pack(fill="both", expand=True, padx=1, pady=1)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview,
                            style="Slim.Vertical.TScrollbar")
        tree.configure(yscrollcommand=self._autohide(vsb))
        vsb.grid(row=0, column=1, sticky="ns", padx=(px(2), 0))
        if xscroll:
            hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview,
                                style="Slim.Horizontal.TScrollbar")
            tree.configure(xscrollcommand=self._autohide(hsb))
            hsb.grid(row=1, column=0, sticky="ew", pady=(px(2), 0))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        tree.tag_configure("odd", background=PALETTE["stripe"])
        tree.tag_configure("nonsig", foreground=PALETTE["disabled_fg"])
        self._set_tree_columns(tree, columns)
        return frame, tree

    @staticmethod
    def _autohide(sb):
        """Scroll command that hides ``sb`` while everything is visible."""
        def _set(lo, hi):
            """Forward the scroll fractions; show the bar only when needed."""
            if float(lo) <= 0.0 and float(hi) >= 1.0:
                sb.grid_remove()
            else:
                sb.grid()
            sb.set(lo, hi)
        return _set

    def _set_tree_columns(self, tree, columns):
        """(Re)define the columns of ``tree`` with sortable headings."""
        tree.configure(columns=[c[0] for c in columns])
        for cid, head, width, anchor in columns:
            tree.heading(cid, text=head, anchor=anchor,
                         command=lambda c=cid: self._sort_tree(tree, c, False))
            tree.column(cid, width=self.px(width), minwidth=self.px(50), anchor=anchor,
                        stretch=True)

    def _fill_tree(self, tree, rows, tags=None):
        """Replace the rows of ``tree`` (``rows`` = list of value tuples)."""
        tree.delete(*tree.get_children(""))
        for i, vals in enumerate(rows):
            t = list(tags[i]) if tags else []
            if i % 2:
                t.append("odd")
            tree.insert("", "end", values=[str(v) for v in vals], tags=t)

    def _sort_tree(self, tree, col, reverse):
        """Sort rows by ``col`` (numeric when possible); click again to reverse."""
        def key(item):
            """Numbers first (as floats), then text (case-insensitive)."""
            v = item[0].replace(",", "").lstrip("<")
            try:
                return (0, float(v), "")
            except ValueError:
                return (1, 0.0, item[0].lower())
        rows = sorted(((tree.set(k, col), k) for k in tree.get_children("")), key=key,
                      reverse=reverse)
        for i, (_, k) in enumerate(rows):
            tree.move(k, "", i)
            t = [x for x in tree.item(k, "tags") if x != "odd"] + (["odd"] if i % 2 else [])
            tree.item(k, tags=t)
        tree.heading(col, command=lambda: self._sort_tree(tree, col, not reverse))

    def _fill_dataframe(self, tree, df, first_width=110):
        """Show a pandas DataFrame in ``tree`` (columns rebuilt from the frame)."""
        cols = []
        for j, c in enumerate(df.columns):
            numeric = df[c].dtype.kind in "fiub"
            width = first_width if j == 0 else max(70, min(130, 8 * len(str(c)) + 16))
            cols.append((str(c), str(c).replace("_", " "), width, "e" if numeric else "w"))
        tree.delete(*tree.get_children(""))
        self._set_tree_columns(tree, cols)
        rows = [[_fmt(v) for v in r] for r in df.itertuples(index=False, name=None)]
        self._fill_tree(tree, rows)

    # ----- status bar ------------------------------------------------------- #
    def _build_statusbar(self, parent):
        """Bottom bar: state dot, status text, progress bar, elapsed time."""
        px, P = self.px, PALETTE
        bar = ttk.Frame(parent, style="Panel.TFrame", padding=(px(16), px(7), px(16), px(7)))
        d = px(10)
        self.status_dot = tk.Canvas(bar, width=d, height=d, bg=P["panel"], highlightthickness=0)
        self._dot = self.status_dot.create_oval(1, 1, d - 1, d - 1, fill=P["idle"], outline="")
        self.status_dot.pack(side="left", padx=(0, px(10)))
        self.lbl_status = ttk.Label(bar, textvariable=self.var_status, style="Status.TLabel")
        self.lbl_status.pack(side="left", fill="x", expand=True)
        ttk.Label(bar, textvariable=self.var_elapsed, style="Elapsed.TLabel", width=12,
                  anchor="e").pack(side="right", padx=(px(12), 0))
        self.progress = ttk.Progressbar(bar, style="Accent.Horizontal.TProgressbar",
                                        mode="determinate", length=px(260), maximum=100.0)
        self.progress.pack(side="right")
        return bar

    def _set_dot(self, which):
        """Colour the status dot: idle | run | ok | warn | bad."""
        col = {"idle": PALETTE["idle"], "run": PALETTE["accent"], "ok": PALETTE["ok"],
               "warn": PALETTE["warn"], "bad": PALETTE["bad"]}[which]
        self.status_dot.itemconfigure(self._dot, fill=col)

    def _set_status(self, text, dot=None, error=False):
        """Update the status text (red for errors) and optionally the dot."""
        self.var_status.set(text)
        self.lbl_status.configure(foreground=PALETTE["bad"] if error else PALETTE["text"])
        if dot:
            self._set_dot(dot)

    def _progress_determinate(self, value):
        """Switch the bar to determinate mode and set it to ``value`` (0-100)."""
        if self._indeterminate:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self._indeterminate = False
        self.progress["value"] = max(0.0, min(100.0, value))

    def _progress_indeterminate(self):
        """Switch the bar to the 'working' animation."""
        if not self._indeterminate:
            self.progress.configure(mode="indeterminate")
            self.progress.start(14)
            self._indeterminate = True

    # --------------------------------------------------------------- events --
    def _bind_events(self):
        """Keyboard shortcuts, mouse wheel routing and field-change traces."""
        r = self.root
        r.bind("<Control-r>", lambda e: self.run())
        r.bind("<F5>", lambda e: self.run())
        r.bind("<Escape>", lambda e: self.stop())
        r.bind("<Control-o>", lambda e: self._browse_data())
        r.bind_all("<MouseWheel>", self._on_wheel, add="+")
        # a combobox must not change its value under the wheel while the sidebar scrolls
        r.bind_class("TCombobox", "<MouseWheel>", lambda e: None)
        self.var_file.trace_add("write", lambda *a: self._on_data_change())
        self.var_type.trace_add("write", lambda *a: self._on_data_change())
        self.var_backbone.trace_add("write", lambda *a: self._on_backbone())
        self.var_max_steps.trace_add("write", lambda *a: self._on_speed())
        self.var_out.trace_add("write", lambda *a: self._refresh_explorer_button())

    def _on_wheel(self, event):
        """Scroll the sidebar when the pointer is over it (Windows wheel units)."""
        try:
            w = self.root.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):
            return
        if self.sidebar.contains(w):
            self.sidebar.scroll(-event.delta / 120 * 2)

    def _on_data_change(self):
        """Enable only the fields that matter for the current input type."""
        path = self.var_file.get().strip()
        demo = not path
        typ = self.var_type.get()
        if typ == "mask" and path.lower().endswith(".json"):
            typ = "coco_mask"                     # a mask .json is read as COCO
        wanted = DATA_ROWS.get(typ, ())
        for key, (widgets, field_w) in self._data_rows.items():
            for w in widgets:
                if key in wanted:
                    w.grid()                       # grid() restores the remembered options
                else:
                    w.grid_remove()
            # the demo ignores these options: show them greyed out
            field_w.state(["disabled"] if demo else ["!disabled"])
        if demo:
            self.hint_file.configure(text=DEMO_HINT)
        else:
            p = self.var_file.get().strip()
            ok = os.path.isfile(p)
            text = f"{os.path.getsize(p) / 1e6:.1f} MB" if ok else "file not found"
            if ok and self._advice_matches():
                # the advisor already identified this file: say what it is
                from .advisor import MODALITIES
                mod = self.rec.profile.modality
                text += f"  ·  {MODALITIES.get(mod, mod)}"
            self.hint_file.configure(text=text)
        if hasattr(self, "btn_apply"):
            self._refresh_advice_buttons()

    def _on_backbone(self):
        """Update the CPU-cost hint under the backbone selector ('cebra' also
        needs its package: the hint says so when it is not installed)."""
        key = self._bb_key.get(self.var_backbone.get())
        if key is None:
            return
        cost = CPU_COST_HINT.get(key, "fast")
        if key == "cebra" and importlib.util.find_spec("cebra") is None:
            cost = "missing"
        text, col = COST_TEXT.get(cost, COST_TEXT["fast"])
        self.lbl_cost.configure(text=text)
        self.lbl_cost_dot.configure(foreground=PALETTE[col])

    def _states_ok(self):
        """True when the temporal state models (hmm / arhmm) can be imported.

        ``embedders_backbones.states`` may still be under construction: a
        failed import is retried on the next call (e.g. when the State model
        drop-down opens), a success is remembered."""
        if self._states_ok_cache:
            return True
        try:
            mod = importlib.import_module(".states", __package__ or "embedders_backbones")
            ok = hasattr(mod, "fit_states")
            self._states_error = "" if ok else "states.fit_states is missing"
        except Exception as exc:  # noqa: BLE001
            ok, self._states_error = False, f"{type(exc).__name__}: {exc}"
        self._states_ok_cache = ok or None
        return ok

    def _refresh_state_models(self):
        """Offer hmm / arhmm only when importable; explain why in the tooltip."""
        ok = self._states_ok()
        base = ("Discrete states from the embedding. auto = backbone default (K-means; DEC "
                "for dtc; CRF for enc_crf); kmeans = K-means; hmm = sticky Gaussian HMM; "
                "arhmm = autoregressive HMM (pose-like dynamics). hmm / arhmm pick K by "
                "held-out likelihood when States is 'auto'.")
        if ok:
            self.cb_state_model.configure(values=STATE_MODELS)
            tip = base
        else:
            self.cb_state_model.configure(values=["auto", "kmeans"])
            if self.var_state_model.get() not in ("auto", "kmeans"):
                self.var_state_model.set("auto")
            tip = (base + "\n\nhmm / arhmm are unavailable: the temporal state models could "
                   f"not be imported ({self._states_error}).")
        for t in getattr(self, "_state_tips", []):
            t.text = tip

    def _advice_matches(self):
        """True when the last recommendation belongs to the current data fields."""
        return (self.rec is not None and
                self._rec_key == (self.var_file.get().strip(), self.var_type.get()))

    def _refresh_advice_buttons(self):
        """Enable Apply / Run suggested only for advice on the current data."""
        ok = self._advice_matches() and not self.busy
        for b in (self.btn_apply, self.btn_run_suggested):
            b.state(["!disabled"] if ok else ["disabled"])
        if self.rec is not None and not self._advice_matches():
            self.lbl_advice.configure(text="the data fields changed: press 'Analyse data' again",
                                      foreground=PALETTE["warn"])
        elif self.rec is not None:
            self.lbl_advice.configure(text=self._advice_caption(), foreground=PALETTE["muted"])

    def _on_speed(self):
        """Describe the selected speed preset (and a max-steps override)."""
        p = SPEED_PRESETS.get(self.var_speed.get())
        if p is None:
            return
        cap = (f"{p['time_budget_s']:.0f} s cap" if p.get("time_budget_s") else "no time cap")
        steps = self.var_max_steps.get().strip()
        st = f"{steps} steps" if steps else f"≤ {p['max_steps']} steps"
        self.lbl_speed.configure(text=f"{st}  ·  window {p['win_len']}  ·  {cap}")

    def set_backbone(self, key):
        """Select backbone ``key`` (e.g. 'pca') in the Model card."""
        if key not in self._bb_display:
            raise ValueError(f"unknown backbone {key!r}")
        self.var_backbone.set(self._bb_display[key])

    def _browse_data(self):
        """Pick the recording file."""
        p = filedialog.askopenfilename(
            parent=self.root, title="Recording",
            filetypes=[("data", "*.npy *.npz *.csv *.json *.pkl *.pickle"), ("all", "*.*")])
        if p:
            self.var_file.set(p)

    def _browse_behaviour(self):
        """Pick the behaviour file."""
        p = filedialog.askopenfilename(parent=self.root, title="Behaviour variables",
                                       filetypes=[("behaviour", "*.csv *.npz *.npy"),
                                                  ("all", "*.*")])
        if p:
            self.var_beh.set(p)

    def _browse_out(self):
        """Pick the output folder."""
        p = filedialog.askdirectory(parent=self.root, title="Output folder",
                                    initialdir=self.var_out.get() or os.getcwd())
        if p:
            self.var_out.set(p)

    # ------------------------------------------------------- form handling --
    def _collect_config(self):
        """Read and validate every field ON THE MAIN THREAD.

        Returns a plain dict (workers never read Tk variables). Raises
        ``_FormError`` with a readable message on invalid input."""
        F = self._fields

        def number(var, name, cast, lo=None, blank=False, what="a number"):
            """Parse one numeric field (``blank`` -> None allowed) with a lower bound."""
            s = var.get().strip()
            if not s:
                if blank:
                    return None
                raise _FormError(f"{name}: value required", F.get(name))
            try:
                v = cast(s)
            except ValueError:
                raise _FormError(f"{name}: '{s}' is not {what}", F.get(name)) from None
            if lo is not None and v < lo:
                raise _FormError(f"{name}: must be >= {lo}", F.get(name))
            return v

        cfg = {}
        cfg["file"] = self.var_file.get().strip()
        if cfg["file"] and not os.path.exists(cfg["file"]):
            raise _FormError(f"data file not found: {cfg['file']}", F.get("file"))
        cfg["input_type"] = self.var_type.get()
        cfg["time_axis"] = int(self.var_taxis.get().strip()[:1] or 1)
        cfg["bin_size"] = number(self.var_bin, "bin_size", float, 1e-6)
        cfg["fs"] = number(self.var_fs, "fs", float, 1e-6, blank=True)
        cfg["dff"] = bool(self.var_dff.get())
        cfg["mask_size"] = number(self.var_mask_size, "mask_size", int, 4, what="an integer")
        cfg["mask_key"] = self.var_mask_key.get().strip() or None
        cfg["beh_file"] = self.var_beh.get().strip()
        if cfg["beh_file"] and not os.path.isfile(cfg["beh_file"]):
            raise _FormError(f"behaviour file not found: {cfg['beh_file']}", F.get("beh_file"))
        cfg["circular"] = [c.strip() for c in self.var_circ.get().split(",") if c.strip()]
        cfg["n_perm"] = number(self.var_nperm, "n_perm", int, 0, what="an integer")
        cfg["decoder"] = self.var_decoder.get() if self.var_decoder.get() in DECODERS else "linear"
        cfg["psid"] = bool(self.var_psid.get())
        cfg["cosmooth"] = bool(self.var_cosmooth.get())
        cfg["goal"] = self.var_goal.get() if self.var_goal.get() in GOALS else "auto"
        bb = self._bb_key.get(self.var_backbone.get())
        if bb is None:
            raise _FormError("choose a backbone", F.get("backbone"))
        cfg["backbone"] = bb
        cfg["speed"] = self.var_speed.get()
        cfg["objective"] = self.var_objective.get()
        cfg["likelihood"] = self.var_likelihood.get()
        cfg["mask_mode"] = self.var_mask_mode.get()
        cfg["state_model"] = self.var_state_model.get()
        if cfg["state_model"] in ("hmm", "arhmm") and not self._states_ok():
            raise _FormError(f"state model {cfg['state_model']} is unavailable "
                             f"({self._states_error})", F.get("state_model"))
        ed = self.var_emb_dim.get().strip().lower()
        if ed in ("", "auto"):
            cfg["emb_dim"] = None
        else:
            cfg["emb_dim"] = number(self.var_emb_dim, "emb_dim", int, 1,
                                    what="'auto' or an integer")
        ns = self.var_states.get().strip().lower()
        if ns in ("", "auto"):
            cfg["n_states"] = "auto"
        else:
            cfg["n_states"] = number(self.var_states, "n_states", int, 2, what="'auto' or an integer")
        cfg["max_steps"] = number(self.var_max_steps, "max_steps", int, 1, blank=True,
                                  what="an integer")
        cfg["smooth"] = number(self.var_smooth, "smooth", float, 0.0)
        out = self.var_out.get().strip()
        if not out:
            raise _FormError("choose an output folder", F.get("outdir"))
        cfg["outdir"] = os.path.abspath(out)
        cfg["gif"], cfg["html"] = bool(self.var_gif.get()), bool(self.var_html.get())
        cfg["gif_method"] = self.var_gif_method.get()
        cfg["view_method"] = self.var_view_proj.get() or "umap"
        bbs = [b.strip() for b in self.var_compare.get().split(",") if b.strip()]
        unknown = [b for b in bbs if b not in self._bb_display]
        if unknown:
            raise _FormError(f"unknown backbone(s) in Compare: {', '.join(unknown)}",
                             F.get("compare"))
        cfg["compare"] = bbs
        cfg["cmp_method"] = self.var_cmp_proj.get() or "umap"
        cfg["cmp_color"] = self.var_cmp_color.get() or "state"
        cfg["extra"] = {}               # Embedder kwargs without a sidebar field (Run suggested)
        cfg["demo_bin"] = None          # bin size of the synthetic demo (None = its native 0.1 s)
        # the advisor's fit method / time axis for this file ('auto' input), if analysed
        cfg["advice"] = None
        if self._advice_matches():
            cfg["advice"] = dict(fit_method=self.rec.fit_method,
                                 fit_kwargs=dict(self.rec.fit_kwargs),
                                 time_axis=int(self.rec.profile.time_axis))
        return cfg

    def _form_error(self, err):
        """Report an invalid field: red status, log line, focus the widget."""
        self._set_status(f"Check the form: {err}", dot="bad", error=True)
        self._log(f"form error: {err}", "err")
        self.root.bell()
        if err.widget is not None:
            try:
                err.widget.focus_set()
            except tk.TclError:
                pass

    # ------------------------------------------------------------ job control --
    def run(self, extra=None, demo_bin=None):
        """Start a single-backbone job (Run button).

        ``extra`` are additional ``Embedder`` kwargs without a sidebar field
        (e.g. win_len, input_pca from the advisor); ``demo_bin`` overrides
        the synthetic demo's bin size. Both are used by ``run_suggested``."""
        if self.busy:
            return
        try:
            cfg = self._collect_config()
        except _FormError as err:
            self._form_error(err)
            return
        cfg["extra"] = dict(extra or {})
        cfg["demo_bin"] = demo_bin
        self._begin_job("run", self._run_worker, cfg,
                        f"{label(cfg['backbone'])} on "
                        f"{os.path.basename(cfg['file']) or 'synthetic demo'}")

    def analyse(self):
        """Start the data advisor on the current data (Analyse data button)."""
        if self.busy:
            return
        try:
            cfg = self._collect_config()
        except _FormError as err:
            self._form_error(err)
            return
        self._begin_job("advise", self._advise_worker, cfg,
                        f"data advisor on {os.path.basename(cfg['file']) or 'synthetic demo'} "
                        f"(goal {cfg['goal']})")

    def apply_suggestions(self, _event=None):
        """Copy the last recommendation into the sidebar fields.

        Every changed field is listed in the Log ('old -> new'); the status
        bar shows a one-line summary. Returns True when the advice applied."""
        if self.busy:
            return False
        if not self._advice_matches():
            self._set_status("No advice for the current data: press 'Analyse data' first.",
                             dot="warn")
            return False
        rec = self.rec
        kw = rec.embedder_kwargs
        changes = []

        def put(var, value, name):
            """Set ``var`` to ``value`` and remember the change for the log."""
            old = var.get()
            if str(old) != str(value):
                var.set(value)
                changes.append(f"{name}: {old if old != '' else '(blank)'} -> {value}")

        bb = kw.get("backbone")
        if bb in self._bb_display:
            old = self._bb_key.get(self.var_backbone.get())
            if old != bb:
                self.var_backbone.set(self._bb_display[bb])
                changes.append(f"backbone: {old} -> {bb}")
        if kw.get("speed") in SPEEDS:
            put(self.var_speed, kw["speed"], "speed")
        if kw.get("objective") in OBJECTIVES:
            put(self.var_objective, kw["objective"], "objective")
        if kw.get("likelihood") in LIKELIHOODS:
            put(self.var_likelihood, kw["likelihood"], "likelihood")
        if kw.get("mask_mode") in MASK_CHOICES:
            put(self.var_mask_mode, kw["mask_mode"], "masking")
        sm = kw.get("state_model", "auto")
        if sm in ("hmm", "arhmm") and not self._states_ok():
            self._log(f"advisor suggests the {sm} state model, which is unavailable "
                      f"({self._states_error}): keeping 'auto'", "warn")
            sm = "auto"
        if sm in STATE_MODELS:
            put(self.var_state_model, sm, "state model")
        ns = kw.get("n_states")
        put(self.var_states, "auto" if ns in (None, "auto") else str(int(ns)), "states")
        if kw.get("smooth_sigma") is not None:
            put(self.var_smooth, f"{float(kw['smooth_sigma']):g}", "smoothing")
        ed = kw.get("embedding_dim")
        put(self.var_emb_dim, "auto" if ed is None else str(int(ed)), "embedding dim")
        dec = rec.decoding or {}
        if dec.get("model") in DECODERS:
            put(self.var_decoder, dec["model"], "decoder")
        if dec.get("n_perm") is not None:
            put(self.var_nperm, str(int(dec["n_perm"])), "shuffles")
        if dec.get("circular") is not None:
            put(self.var_circ, ",".join(dec["circular"]), "circular")
        cmp_list = [b for b in rec.compare if b in self._bb_display]
        if cmp_list:
            put(self.var_compare, ",".join(cmp_list), "compare")
        if self.var_file.get().strip():
            # file input: bin size and, for an explicit input type, how to read the file
            if kw.get("bin_size") is not None:
                put(self.var_bin, f"{float(kw['bin_size']):g}", "bin size")
            if self.var_type.get() != "auto":
                typ = FIT_TO_INPUT.get(rec.fit_method)
                if typ == "timeseries" and rec.profile.modality == "pose":
                    typ = "pose"
                fk = rec.fit_kwargs or {}
                if "time_axis" in fk:
                    put(self.var_taxis, TIME_AXES[0] if int(fk["time_axis"]) == 1
                        else TIME_AXES[1], "time axis")
                if fk.get("fs"):
                    put(self.var_fs, f"{float(fk['fs']):g}", "fs")
                if "dff" in fk:
                    put(self.var_dff, bool(fk["dff"]), "dF/F")
                if typ and typ != self.var_type.get():
                    put(self.var_type, typ, "input type")
                    self._rec_key = (self.var_file.get().strip(), typ)   # still the same data
        self._on_speed()
        self._on_backbone()
        self._on_data_change()
        for c in changes:
            self._log(f"  advisor applied  {c}", "ok")
        extra = self._advice_extra()
        note = ("" if not extra else "  |  Run suggested also uses " +
                ", ".join(f"{k} {v}" for k, v in extra.items()))
        self._set_status(f"Advisor suggestions applied: {len(changes)} field"
                         f"{'' if len(changes) == 1 else 's'} changed{note}", dot="ok")
        self._log(f"advisor suggestions applied ({len(changes)} changes)", "head")
        return True

    def _advice_extra(self):
        """Advice that has no sidebar field: win_len, input_pca and, for the
        synthetic demo, the suggested bin size (passed by ``run_suggested``)."""
        if self.rec is None:
            return {}
        kw = self.rec.embedder_kwargs
        # input_pca 'auto' is the Embedder default: not worth mentioning or passing
        out = {k: kw[k] for k in ("win_len", "input_pca")
               if kw.get(k) is not None and not (k == "input_pca" and kw[k] == "auto")}
        if not self.var_file.get().strip() and kw.get("bin_size") is not None:
            out["bin_size"] = kw["bin_size"]
        return out

    def run_suggested(self, _event=None):
        """Apply the suggestions, then run with the advice's extra settings."""
        if self.busy or not self.apply_suggestions():
            return
        extra = self._advice_extra()
        demo_bin = extra.pop("bin_size", None)
        self.run(extra=extra, demo_bin=demo_bin)

    def compare(self):
        """Start a comparison job over the Compare list (Compare button)."""
        if self.busy:
            return
        try:
            cfg = self._collect_config()
        except _FormError as err:
            self._form_error(err)
            return
        if not cfg["compare"]:
            self._form_error(_FormError("list at least one backbone to compare",
                                        self._fields.get("compare")))
            return
        self._begin_job("compare", self._compare_worker, cfg,
                        f"compare {', '.join(cfg['compare'])}")

    def stop(self):
        """Request cancellation; the worker stops at its next checkpoint."""
        if not self.busy or self._stop.is_set():
            return
        self._stop.set()
        self.btn_stop.state(["disabled"])
        self._set_status("Stopping at the next checkpoint...", dot="warn")
        self._log("stop requested", "warn")

    def _begin_job(self, kind, target, cfg, title):
        """Lock the UI, reset progress and launch ``target(cfg)`` in a thread."""
        self._save_settings()
        self.busy = True
        self._job_kind = kind
        self._job_note = ""
        self._stop.clear()
        for b in (self.btn_run, self.btn_cmp, self.btn_analyse, self.btn_apply,
                  self.btn_run_suggested):
            b.state(["disabled"])
        self.btn_stop.state(["!disabled"])
        self._progress_determinate(0)
        self._job_t0 = time.perf_counter()
        self.var_elapsed.set("0.0 s")
        self._set_status(f"Starting: {title}", dot="run")
        self._log(f"\n== {time.strftime('%H:%M:%S')}  {title} ==", "head")
        threading.Thread(target=self._job_main, args=(kind, target, cfg), daemon=True,
                         name=f"nemba-{kind}").start()

    def _job_main(self, kind, target, cfg):
        """Worker entry: run the job, translate its outcome into messages."""
        status, msg = "done", ""
        try:
            target(cfg)
        except _Cancelled:
            status = "stopped"
        except Exception as exc:  # noqa: BLE001
            status, msg = "error", f"{type(exc).__name__}: {exc}"
            self._post("error", (msg, traceback.format_exc()))
        finally:
            self._post("finished", (kind, status, msg))

    def _post(self, kind, payload=None):
        """Thread-safe: enqueue a message for the main thread."""
        self._queue.put((kind, payload))

    def _finish_job(self, kind, status, msg):
        """Main thread: unlock the UI and summarise the job outcome."""
        el = time.perf_counter() - (self._job_t0 or time.perf_counter())
        self.busy = False
        self.last_status, self.last_elapsed = status, el
        for b in (self.btn_run, self.btn_cmp, self.btn_analyse):
            b.state(["!disabled"])
        self.btn_stop.state(["disabled"])
        self._refresh_advice_buttons()
        self.var_elapsed.set(f"{el:.1f} s")
        what = {"compare": "Comparison", "advise": "Analysis"}.get(kind, "Run")
        if status == "done":
            self._progress_determinate(100)
            where = "advice and profile in" if kind == "advise" else "outputs in"
            self._set_status(f"{what} finished in {el:.1f} s{self._job_note}  |  {where} "
                             f"{_short_path(self.last_outdir or '', 3)}", dot="ok")
            self._log(f"{what.lower()} finished in {el:.1f} s", "ok")
        elif status == "stopped":
            self._progress_determinate(0)
            self._set_status(f"{what} stopped by user after {el:.1f} s", dot="warn")
            self._log(f"{what.lower()} stopped after {el:.1f} s", "warn")
        else:
            self._progress_determinate(0)
            self._set_status(f"{what} failed: {msg}  (details in the Log tab)", dot="bad",
                             error=True)
            self.nb.select(self.tab_log)
        self._summary = (self.var_status.get(), status)
        self._refresh_explorer_button()

    # ---------------------------------------------------------- queue drain --
    def _poll(self):
        """Main thread: drain the worker queue (batched), then reschedule."""
        try:
            lines = []
            for _ in range(800):
                try:
                    kind, payload = self._queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "log":
                    lines.append(payload)
                    continue
                if lines:
                    self._append_log(lines)
                    lines = []
                self._dispatch(kind, payload)
            if lines:
                self._append_log(lines)
            if self.busy and self._job_t0 is not None:
                self.var_elapsed.set(f"{time.perf_counter() - self._job_t0:.1f} s")
        except Exception:  # noqa: BLE001 - the poller must never die
            self._count_error(traceback.format_exc())
        finally:
            self._poll_id = self.root.after(POLL_MS, self._poll)

    def _dispatch(self, kind, payload):
        """Handle one non-log message from a worker."""
        if kind == "progress":
            p = payload
            self._progress_determinate(100.0 * p["frac"])
            val = f"  val {p['val']:.3f}" if np.isfinite(p["val"]) else ""
            self._set_status(f"{p['prefix']}  training step {p['step']}/{p['max']}  "
                             f"loss {p['loss']:.3f}{val}", dot="run")
        elif kind == "phase":
            text, frac = payload
            if frac is None:
                self._progress_indeterminate()
            else:
                self._progress_determinate(100.0 * frac)
            if not self._stop.is_set():
                self._set_status(text, dot="run")
        elif kind == "call":
            try:
                payload()
            except Exception:  # noqa: BLE001
                self._count_error(traceback.format_exc())
        elif kind == "error":
            summary, tb = payload
            self.error_count += 1
            self._append_log([(f"ERROR {summary}\n{tb.rstrip()}", "err")])
        elif kind == "finished":
            self._finish_job(*payload)

    def _count_error(self, tb):
        """Record an exception raised on the main thread (shown in the Log)."""
        self.error_count += 1
        self._append_log([(tb.rstrip(), "err")])

    def _report_tk_exception(self, exc, val, tb):
        """Tk callback errors go to the Log tab instead of stderr."""
        self._count_error("".join(traceback.format_exception(exc, val, tb)))

    # ------------------------------------------------------------------ log --
    def _log(self, text, tag=None):
        """Main thread: append one message to the Log tab."""
        self._append_log([(text, tag) if tag else text])

    @staticmethod
    def _classify(line):
        """Pick a colour tag for a library log line."""
        s = line.strip()
        if s.startswith(("ERROR", "Traceback")):
            return "err"
        if "WARNING" in s or s.startswith("warning"):
            return "warn"
        if s.startswith("=="):
            return "head"
        return None

    def _append_log(self, items):
        """Insert log items (str or (str, tag)); keep the view pinned to the end."""
        t = self.log_text
        at_end = t.yview()[1] > 0.995
        t.configure(state="normal")
        for it in items:
            text, tag = it if isinstance(it, tuple) else (it, None)
            for line in str(text).split("\n"):
                t.insert("end", line + "\n", (tag or self._classify(line)) or ())
        n = int(t.index("end-1c").split(".")[0])
        if n > MAX_LOG_LINES:
            t.delete("1.0", f"{n - MAX_LOG_LINES}.0")
        t.configure(state="disabled")
        if at_end:
            t.see("end")

    def _clear_log(self):
        """Empty the Log tab."""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def log_contents(self):
        """Full text of the Log tab (used by tests)."""
        return self.log_text.get("1.0", "end-1c")

    # ============================================================ WORKERS ==== #
    # Everything below until "main-thread result display" runs in a worker
    # thread: no Tk calls, only self._post(...) and the reporter callbacks.
    def _load_data(self, cfg, rep):
        """Worker: read the recording + behaviour (or build the demo).

        Returns a dict with the payload and how to fit it: ``kind`` is
        'spikes' | 'coco_mask' | 'fiber_photometry' | 'features' (any other
        matrix), plus ``time_axis``, ``times``, ``fs``, ``dff`` and
        ``bin_size``. Input type 'auto' reads the file with
        ``data.load_auto``; a matrix then follows the advisor's fit method
        and time axis when this file was analysed, else the time axis is
        detected (lag-1 autocorrelation, ``advisor.detect_time_axis``)."""
        out = dict(behavior=None, behavior_times=None, circular=list(cfg["circular"]),
                   time_axis=cfg["time_axis"], times=None, fs=cfg["fs"], dff=cfg["dff"],
                   bin_size=cfg["bin_size"])
        if not cfg["file"]:
            rep.phase("simulating the synthetic demo recording")
            ds = make_dataset(seed=0)
            out.update(kind="spikes", payload=ds["spike_times"], name="synthetic demo",
                       bin_size=float(cfg.get("demo_bin") or ds["dt"]))
            if not cfg["beh_file"]:
                out["behavior"], out["behavior_times"] = ds["behavior"], ds["times"]
                if "phase" not in out["circular"]:
                    out["circular"].append("phase")
        else:
            path, typ = cfg["file"], cfg["input_type"]
            rep.phase(f"loading {os.path.basename(path)}")
            out["name"] = os.path.basename(path)
            if typ == "auto":
                self._load_auto(path, cfg, out, rep)
            elif typ == "spikes":
                out.update(kind="spikes", payload=_data.load_spikes(path))
            elif typ == "coco_mask" or (typ == "mask" and path.lower().endswith(".json")):
                out.update(kind="coco_mask", payload=path)
            elif typ == "fiber_photometry":
                out.update(kind="fiber_photometry", payload=_data.load_array(path))
            else:
                out.update(kind="features", payload=_data.load_array(path))
        if cfg["beh_file"]:
            out["behavior"], out["behavior_times"] = _data.load_behavior(cfg["beh_file"])
        rep.check()
        return out

    @staticmethod
    def _load_auto(path, cfg, out, rep):
        """Worker: fill ``out`` for input type 'auto' (see ``_load_data``)."""
        payload, hints = _data.load_auto(path)
        kind = hints.get("kind")
        out["fs"] = out["fs"] or hints.get("fs")
        adv = cfg.get("advice")
        if kind == "spike_times":
            out.update(kind="spikes", payload=payload)
            how = f"{len(payload)} spike trains, {cfg['bin_size']:g} s bins"
        elif kind == "coco":
            out.update(kind="coco_mask", payload=payload)
            how = "COCO masks"
        elif kind in ("pose", "table"):
            # rows of a table / DeepLabCut file are time
            out.update(kind="features", payload=payload, time_axis=0, times=hints.get("times"))
            how = f"{kind} with {np.shape(payload)[1]} columns, time along rows"
        elif adv is not None and adv["fit_method"] == "fit_fiber":
            fk = adv["fit_kwargs"]
            out.update(kind="fiber_photometry", payload=payload,
                       time_axis=int(fk.get("time_axis", 1)), dff=bool(fk.get("dff", False)),
                       fs=fk.get("fs") or out["fs"])
            how = f"fiber photometry as advised (dF/F {'on' if out['dff'] else 'off'})"
        else:
            if adv is not None:
                axis, src = int(adv["time_axis"]), "advisor"
            else:
                from .advisor import detect_time_axis
                axis, conf = detect_time_axis(np.asarray(payload, float))
                src = f"detected, {conf} confidence"
            out.update(kind="features", payload=payload, time_axis=int(axis),
                       times=hints.get("times"))
            how = f"matrix {np.shape(payload)}, time axis {axis} ({src})"
        rep.post("log", (f"auto input: {how}", "muted"))

    @staticmethod
    def _fit(emb, data, cfg):
        """Worker: dispatch to the right ``Embedder.fit_*`` for the input kind."""
        kind, p = data["kind"], data["payload"]
        if kind == "spikes":
            emb.fit_spikes(p)
        elif kind == "coco_mask":
            s = cfg["mask_size"]
            emb.fit_coco_masks(p, resize=(s, s), channel_key=cfg["mask_key"], fps=data["fs"])
        elif kind == "fiber_photometry":
            emb.fit_fiber(p, time_axis=data["time_axis"], dff=data["dff"], fs=data["fs"])
        else:
            times = data.get("times")
            if times is not None and len(times) != np.shape(p)[data["time_axis"]]:
                times = None                          # time column does not match: frame index
            emb.fit_features(p, time_axis=data["time_axis"], times=times)

    @staticmethod
    def _embedder_kwargs(cfg):
        """Embedder settings shared by Run and Compare (from the sidebar)."""
        return dict(n_states=cfg["n_states"], speed=cfg["speed"], objective=cfg["objective"],
                    max_steps=cfg["max_steps"], smooth_sigma=cfg["smooth"],
                    likelihood=cfg["likelihood"], mask_mode=cfg["mask_mode"],
                    state_model=cfg["state_model"], embedding_dim=cfg["emb_dim"])

    def _run_worker(self, cfg):
        """Worker: fit one backbone, write outputs, build every figure."""
        bb = cfg["backbone"]
        rep = _Reporter(self._post, self._stop, [bb], "embedding + automatic state selection")
        data = self._load_data(cfg, rep)
        kw = self._embedder_kwargs(cfg)
        kw.update(cfg.get("extra") or {})
        if cfg.get("extra"):
            rep.post("log", ("advisor extras: " + ", ".join(f"{k}={v}" for k, v in
                                                             cfg["extra"].items()), "muted"))
        emb = Embedder(bb, bin_size=data["bin_size"], log=rep.log, progress=rep.progress, **kw)
        rep.phase(f"{bb}: preparing input and model")
        self._fit(emb, data, cfg)

        cos = None
        if cfg["cosmooth"]:
            cos = self._cosmooth(emb, rep)
        # projections are cached by the Embedder: computing them here (one at
        # a time) gives stage feedback and a Stop checkpoint between them
        order = ["pca", "tsne", "umap"]
        n_st = len(order) + 2
        for i, m in enumerate(order, 1):
            rep.check()
            rep.phase(f"{bb}: {PRETTY[m]} projection  ({i}/{n_st})")
            emb.projection(m)
        rep.check()
        beh, bt, circ = data["behavior"], data["behavior_times"], data["circular"]
        aligned = emb._align(beh, bt, None, circ) if beh is not None else None
        if aligned:
            # decode here (decoder choice, PSID rows); run() then reuses emb.decoding
            rep.phase(f"{bb}: decoding behaviour ({cfg['decoder']}"
                      + (", PSID" if cfg["psid"] else "") + ")")
            emb.decode(aligned, circular=circ, model=cfg["decoder"], n_perm=cfg["n_perm"],
                       psid=cfg["psid"])
            rep.check()
        rep.phase(f"{bb}: writing figures, tables, dynamics, GIF and explorer  ({n_st - 1}/{n_st})")
        met = emb.run(cfg["outdir"], behavior=aligned, behavior_times=None, animate=cfg["gif"],
                      gif_method=cfg["gif_method"], html=cfg["html"], circular=circ,
                      n_perm=cfg["n_perm"], decode=False)
        if emb.decoding is not None:
            emb.decoding.to_csv(os.path.join(cfg["outdir"], f"{bb}_decoding.csv"))
        rep.check()
        rep.phase(f"{bb}: building the views  ({n_st}/{n_st})")
        color = f"beh:{next(iter(aligned))}" if aligned else "time"
        method = cfg["view_method"] if cfg["view_method"] in PROJECTIONS else "umap"
        figs = dict(
            overview=viz.dashboard(emb, behavior=aligned, circular=circ),
            embedding=_embedding_figure(emb, method, color, aligned, circ),
            decoding=(viz.decoding_figure(_decoding_view(emb.decoding), emb.times)
                      if emb.decoding is not None else None),
            training=viz.training_figure(emb.history, title=f"{label(bb)} training"),
        )
        self._log_metrics(rep, met)
        res = _RunResult(emb=emb, behaviour=aligned, circular=circ, outdir=cfg["outdir"],
                         figures=figs, metrics=met, view=(method, color),
                         gif_method=cfg["gif_method"], decoder=cfg["decoder"], cosmooth=cos)
        self._post("call", lambda: self._show_run(res))

    @staticmethod
    def _cosmooth(emb, rep):
        """Worker: co-smoothing score of ``emb`` (bits / spike), or None.

        Skipped gracefully when the input has no raw counts (continuous
        data) or when the score cannot be computed (logged as a warning)."""
        if emb.counts is None:
            rep.post("log", ("co-smoothing skipped: it needs raw counts (spike times or "
                             "non-negative integer data)", "warn"))
            return None
        rep.phase(f"{emb.backbone}: co-smoothing (refit on held-in neurons)")
        old, rep.post_fit_text = rep.post_fit_text, "co-smoothing"
        try:
            cos = emb.cosmooth()
        except _Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            rep.post("log", (f"co-smoothing failed: {type(exc).__name__}: {exc}", "warn"))
            return None
        finally:
            rep.post_fit_text = old
        rep.post("log", (f"co-smoothing: {float(cos['bps']):.4f} bits/spike on "
                         f"{cos.get('n_heldout', '?')} held-out neurons, "
                         f"{cos.get('n_test_frames', '?')} test frames", "ok"))
        return cos

    @staticmethod
    def _log_metrics(rep, met):
        """Worker: one compact line with the headline metrics."""
        keys = ("n_states", "silhouette", "cluster_structure", "circularity",
                "participation_ratio", "cosmooth_bps", "train_steps", "train_seconds",
                "total_seconds")
        parts = [f"{k}={_fmt(met[k])}" for k in keys if k in met and
                 not (isinstance(met[k], float) and not np.isfinite(met[k]))]
        rep.post("log", ("metrics: " + "  ".join(parts), "ok"))

    def _compare_worker(self, cfg):
        """Worker: fit every listed backbone, write the comparison, build the
        embedding grid and the consistency matrix."""
        from .viz_dynamics import consistency_figure
        bbs = cfg["compare"]
        rep = _Reporter(self._post, self._stop, bbs, "embedding, auto K and decoding")
        data = self._load_data(cfg, rep)
        s = cfg["mask_size"]
        inputs = {"spikes": "spikes", "coco_mask": "coco_mask",
                  "fiber_photometry": "fiber_photometry"}
        cmp = compare_embedders(
            data["payload"], input=inputs.get(data["kind"], "timeseries"), backbones=bbs,
            time_axis=data["time_axis"], fs=data["fs"], dff=data["dff"],
            behavior=data["behavior"], behavior_times=data["behavior_times"],
            circular=data["circular"], log=rep.log, mask_resize=(s, s),
            mask_channel_key=cfg["mask_key"], n_perm=cfg["n_perm"], decode_model=cfg["decoder"],
            bin_size=data["bin_size"], progress=rep.progress, **self._embedder_kwargs(cfg))
        rep.check()
        outdir = os.path.join(cfg["outdir"], "compare")
        rep.phase("writing comparison figures and tables (UMAP of every backbone)")
        cmp.run(outdir)
        rep.check()
        rep.phase("building the comparison grid and the consistency matrix")
        method = cfg["cmp_method"] if cfg["cmp_method"] in PROJECTIONS else "umap"
        color = cfg["cmp_color"] if cfg["cmp_color"] in ("state", "time") else "state"
        fig = viz.compare_embeddings_figure(cmp.embedders, method=method, color=color)
        cm = cfig = None
        if len(cmp.embedders) >= 2:
            try:
                cm = cmp.consistency()
                cfig = consistency_figure(cm)
            except Exception as exc:  # noqa: BLE001  (optional view)
                rep.post("log", (f"consistency matrix skipped: {type(exc).__name__}: {exc}",
                                 "warn"))
        rep.post("log", ("\n" + cmp.metrics.round(3).to_string(index=False), None))
        if cm is not None:
            rep.post("log", ("consistency R² (row predicts column):\n"
                             + cm.round(3).to_string(), None))
        best = cmp.best("silhouette")
        rep.post("log", (f"best by silhouette: {best}", "ok"))
        if cmp.decoding is not None:
            for tgt in cmp.decoding["target"].unique():
                rep.post("log", (f"best held-out decoding of {tgt}: "
                                 f"{cmp.best('decoding:' + str(tgt))}", "ok"))
        res = _CompareResult(cmp=cmp, outdir=outdir, figure=fig, view=(method, color),
                             consistency=cm, cons_figure=cfig)
        self._post("call", lambda: self._show_compare(res))

    def _advise_worker(self, cfg):
        """Worker: profile the data, build the advice, its figure and files."""
        from .advisor import advise, profile_figure
        rep = _Reporter(self._post, self._stop, ["advisor"], "")
        beh = bt = columns = times = None
        fs, time_axis = cfg["fs"], "auto"
        if not cfg["file"]:
            rep.phase("simulating the synthetic demo recording")
            ds = make_dataset(seed=0)
            payload, name = ds["spike_times"], "synthetic demo"
            if not cfg["beh_file"]:
                beh, bt = ds["behavior"], ds["times"]
        else:
            path, typ, name = cfg["file"], cfg["input_type"], os.path.basename(cfg["file"])
            rep.phase(f"loading {name}")
            if typ == "auto":
                payload, hints = _data.load_auto(path)
                columns, times = hints.get("columns"), hints.get("times")
                fs = fs or hints.get("fs")
            elif typ == "spikes":
                payload = _data.load_spikes(path)
            elif typ == "coco_mask" or (typ == "mask" and path.lower().endswith(".json")):
                payload = path
            else:
                payload, time_axis = _data.load_array(path), cfg["time_axis"]
        if cfg["beh_file"]:
            beh, bt = _data.load_behavior(cfg["beh_file"])
        rep.check()
        rep.phase("profiling: value type, timescales, dimensionality, ring and cluster "
                  "structure" + (", behaviour" if beh is not None else ""))
        rec = advise(payload, goal=cfg["goal"], fs=fs, times=times, time_axis=time_axis,
                     columns=columns, behavior=beh, behavior_times=bt)
        rep.check()
        rep.phase("drawing the profile and writing advice files")
        fig = profile_figure(rec.profile)
        outdir = cfg["outdir"]
        os.makedirs(outdir, exist_ok=True)
        rec.save(os.path.join(outdir, "advice"))
        viz.save_figure(profile_figure(rec.profile), os.path.join(outdir, "data_profile"))
        kw = rec.embedder_kwargs
        rep.post("log", (f"advisor: {rec.profile.modality}, {rec.profile.n_channels} channels "
                         f"x {rec.profile.n_frames:,} frames -> {kw.get('backbone')} via "
                         f"{rec.fit_method} (goal {rec.goal}); profiled in "
                         f"{rec.profile.seconds:.1f} s", "ok"))
        for w in rec.warnings:
            rep.post("log", (f"advisor warning: {w}", "warn"))
        rep.post("log", (f"wrote advice.md, advice.json and data_profile.png/pdf/svg to "
                         f"{outdir}", "muted"))
        res = _AdviceResult(rec=rec, figure=fig, report=rec.report(),
                            key=(cfg["file"], cfg["input_type"]), name=name, outdir=outdir)
        self._post("call", lambda: self._show_advice(res))

    # ------------------------------------------- view re-render (workers) ---- #
    def _view_worker(self, emb, req, beh, circ):
        """Worker: recompute (if needed) a projection and build the scatter."""
        try:
            fig = _embedding_figure(emb, req[0], req[1], beh, circ)
            self._post("call", lambda: self._show_view(emb, req, fig))
        except Exception as exc:  # noqa: BLE001
            msg, tb = f"{type(exc).__name__}: {exc}", traceback.format_exc()
            self._post("error", (f"embedding view: {msg}", tb))
            self._post("call", lambda: self._view_done("embedding"))

    def _cmp_view_worker(self, cmp, req):
        """Worker: build the comparison grid for another projection / colour."""
        try:
            fig = viz.compare_embeddings_figure(cmp.embedders, method=req[0], color=req[1])
            self._post("call", lambda: self._show_cmp_view(cmp, req, fig))
        except Exception as exc:  # noqa: BLE001
            msg, tb = f"{type(exc).__name__}: {exc}", traceback.format_exc()
            self._post("error", (f"compare view: {msg}", tb))
            self._post("call", lambda: self._view_done("compare"))

    # ===================================== main-thread result display ===== #
    def _show_run(self, res):
        """Main thread: put a finished run's figures and tables on screen."""
        emb, m = res.emb, res.metrics
        self.emb, self.beh, self.circular = emb, res.behaviour, list(res.circular)
        self.run_method = res.gif_method if res.gif_method in PROJECTIONS else "umap"
        self.cosmooth = res.cosmooth
        self.last_outdir = res.outdir
        P = self.panes
        struct = m.get("cluster_structure", "undefined")
        cap = (f"silhouette {_fmt(m.get('silhouette'))} ("
               + ("no" if struct == "none" else struct) + " cluster structure)")
        if emb.decoding is not None and len(emb.decoding.table):
            te = emb.decoding.table[emb.decoding.table.features == "embedding"]
            cap += "  |  decoding " + ", ".join(f"{r.target} {r.score:.2f}"
                                                 for r in te.itertuples())
        if res.cosmooth is not None:
            bps = float(res.cosmooth["bps"])
            cap += f"  |  co-smoothing {bps:.3f} bits/spike"
            self._job_note = f"  |  co-smoothing {bps:.3f} bits/spike"
        P["overview"].show(res.figures["overview"], caption=cap)
        self._set_color_choices(res.behaviour, res.view[1])
        self.var_view_proj.set(res.view[0])
        P["embedding"].show(res.figures["embedding"], caption=self._embed_caption(emb, *res.view))
        if res.figures["decoding"] is not None:
            tab = emb.decoding.table
            emb_rows = tab[tab["features"] == "embedding"]
            extra = tab.loc[~tab["features"].isin(FIGURE_FEATURES), "features"]
            names = sorted({FEATURE_NAMES.get(f, str(f)) for f in extra})
            P["decoding"].show(res.figures["decoding"], caption=(
                f"{tab['target'].nunique()} targets  |  {res.decoder} decoder  |  blocked "
                f"{int(emb_rows['n_folds'].max())}-fold CV  |  "
                f"{int(emb_rows['n_perm'].max())} circular-shift shuffles"
                + (f"  |  {', '.join(names)} rows in the table only" if names else "")))
            self._fill_decoding(tab)
        else:
            P["decoding"].clear("No behaviour was given, so nothing was decoded.\n"
                                "Add a behaviour file in the sidebar and run again.")
            self.tree_decoding.delete(*self.tree_decoding.get_children(""))
        h = emb.history or {}
        cap = (f"{h.get('steps_done', 0)} steps  |  best step {h.get('best_step', 0)}  |  "
               f"{h.get('seconds', 0.0):.1f} s" + ("  |  early stop" if h.get("stopped_early") else "")
               if h.get("step") else "closed-form fit, no gradient training")
        P["training"].show(res.figures["training"], caption=cap)
        for cb in (self.cb_view_proj, self.cb_view_color):
            cb.state(["!disabled"])
        self._reset_dynamics()
        self.nb.select(P["overview"])

    def _fill_decoding(self, tab):
        """Fill the Decoding table (grey rows: Holm p >= 0.05; PSID rows have
        no permutation null, so they are never greyed and show p as 'n/a')."""
        rows, tags = [], []
        for r in tab.itertuples(index=False):
            psid = r.features == "psid"
            rows.append((r.target, FEATURE_NAMES.get(r.features, str(r.features)), r.metric,
                         _fmt(r.score), _fmt(r.null_p95), "n/a" if psid else _fmt_p(r.p_holm)))
            ok = psid or (np.isfinite(r.p_holm) and r.p_holm < 0.05)
            tags.append([] if ok else ["nonsig"])
        self._fill_tree(self.tree_decoding, rows, tags)

    def _set_color_choices(self, behaviour, current):
        """Colour-by options: time, model states, then each behaviour variable."""
        self._color_map = {"time": "time", "model states": "state"}
        for name in (behaviour or {}):
            self._color_map[f"{name} (behaviour)"] = f"beh:{name}"
        self.cb_view_color.configure(values=list(self._color_map))
        inv = {v: k for k, v in self._color_map.items()}
        self.var_view_color.set(inv.get(current, "time"))

    def _embed_caption(self, emb, method, color_key):
        """Caption of the Embedding tab, e.g. '3,000 of 6,000 frames shown'."""
        try:
            n = len(emb.projection(method)[1])          # cached at this point
        except Exception:  # noqa: BLE001
            n = len(emb.Z)
        # projection and colour are already shown by the controls next to the caption
        return f"{n:,} of {len(emb.Z):,} frames shown"

    def _on_view_change(self, _event=None):
        """Projection / colour changed: rebuild the scatter in a worker."""
        if self.emb is None:
            return
        method = self.var_view_proj.get()
        req = (method, self._color_map.get(self.var_view_color.get(), "time"))
        if self._view_busy["embedding"]:
            self._view_pending["embedding"] = req
            return
        self._view_busy["embedding"] = True
        cached = (method, None) in getattr(self.emb, "_proj", {})
        if not cached and not self.busy:
            self._set_status(f"Computing the {PRETTY.get(method, method)} projection...",
                             dot="run")
            self._progress_indeterminate()
        threading.Thread(target=self._view_worker, args=(self.emb, req, self.beh, self.circular),
                         daemon=True, name="nemba-view").start()

    def _show_view(self, emb, req, fig):
        """Main thread: display a re-rendered scatter (if still relevant)."""
        if emb is self.emb:
            self.panes["embedding"].show(fig, caption=self._embed_caption(emb, *req))
        self._view_done("embedding")

    def _on_cmp_view_change(self, _event=None):
        """Compare projection / colour changed: rebuild the grid in a worker."""
        if self.cmp is None:
            return
        req = (self.var_cmp_proj.get(), self.var_cmp_color.get())
        if self._view_busy["compare"]:
            self._view_pending["compare"] = req
            return
        self._view_busy["compare"] = True
        if not self.busy:
            self._set_status(f"Building the {PRETTY.get(req[0], req[0])} comparison grid...",
                             dot="run")
            self._progress_indeterminate()
        threading.Thread(target=self._cmp_view_worker, args=(self.cmp, req), daemon=True,
                         name="nemba-cmp-view").start()

    def _show_cmp_view(self, cmp, req, fig):
        """Main thread: display a re-rendered comparison grid (kept for the
        Embeddings / Consistency toggle; shown only in grid mode)."""
        if cmp is self.cmp:
            self._cmp_figs["grid"] = fig
            if self.var_cmp_mode.get() == "grid":
                self.panes["compare"].show(fig)
        self._view_done("compare")

    def _view_done(self, which):
        """A view worker ended: restore the status and serve a pending request."""
        self._view_busy[which] = False
        if not self.busy:
            # restore the last job summary (the view change is visible on screen)
            self._progress_determinate(100 if self.last_status == "done" else 0)
            text, status = getattr(self, "_summary", ("Ready", "idle"))
            dot = {"done": "ok", "stopped": "warn", "error": "bad"}.get(status, "idle")
            self._set_status(text, dot=dot, error=status == "error")
        pending, self._view_pending[which] = self._view_pending[which], None
        if pending is not None:
            {"embedding": self._on_view_change, "compare": self._on_cmp_view_change,
             "dynamics": self._on_dyn_change}[which]()

    def _show_compare(self, res):
        """Main thread: put a finished comparison on the Compare tab."""
        cmp = res.cmp
        self.cmp = cmp
        self.last_outdir = res.outdir
        self._cmp_figs = {"grid": res.figure, "consistency": res.cons_figure}
        self._cmp_caption = f"{len(cmp.embedders)} backbones  |  {_short_path(res.outdir, 2)}"
        self.var_cmp_mode.set("grid")
        self.rb_cmp_mode["grid"].state(["!disabled"])
        self.rb_cmp_mode["consistency"].state(
            ["!disabled"] if res.cons_figure is not None else ["disabled"])
        self.panes["compare"].show(res.figure, caption=self._cmp_caption)
        self._fill_dataframe(self.tree_cmp_metrics, cmp.metrics)
        piv = cmp.decoding_table()
        if piv is not None and len(piv):
            self._fill_dataframe(self.tree_cmp_decoding, piv.reset_index())
        else:
            self._set_tree_columns(self.tree_cmp_decoding, [("backbone", "backbone", 110, "w"),
                                                            ("note", "note", 220, "w")])
            self._fill_tree(self.tree_cmp_decoding, [("-", "no behaviour given")])
        self.var_cmp_proj.set(res.view[0])
        self.var_cmp_color.set(res.view[1])
        for cb in (self.cb_cmp_proj, self.cb_cmp_color):
            cb.state(["!disabled"])
        self.nb.select(self.tab_compare)

    def _on_cmp_mode(self):
        """Embeddings / Consistency toggle of the Compare tab (cached figures)."""
        mode = self.var_cmp_mode.get()
        fig = self._cmp_figs.get(mode)
        if self.cmp is None or fig is None:
            return
        pane = self.panes["compare"]
        if mode == "consistency":
            pane.show(fig, caption="cross-validated R² of a linear map, row predicts "
                                   "column  |  " + _short_path(self.last_outdir or "", 2))
        else:
            pane.show(fig, caption=self._cmp_caption)
        # projection and colour only apply to the embedding grid
        for cb in (self.cb_cmp_proj, self.cb_cmp_color):
            cb.state(["!disabled"] if mode == "grid" else ["disabled"])

    # ------------------------------------------------------------ dynamics -- #
    def _reset_dynamics(self):
        """New run: forget cached Dynamics figures and list the available ones."""
        self._dyn_cache, self._dyn_shown = {}, None
        self._dyn_map = _dynamics_choices(self.emb, self.beh, self.circular)
        names = list(self._dyn_map)
        self.cb_dyn.configure(values=names)
        if self.var_dyn_fig.get() not in names:
            self.var_dyn_fig.set(names[0])
        self.cb_dyn.state(["!disabled"])
        self.panes["dynamics"].clear("Rendering on demand: open this tab or pick a figure.")
        self.tree_dyn.delete(*self.tree_dyn.get_children(""))
        self.lbl_dyn_help.configure(text="")
        if self._current_tab() is self.tab_dynamics:
            self._on_dyn_change()

    def _current_tab(self):
        """Widget of the selected notebook tab (None if unavailable)."""
        try:
            return self.nb.nametowidget(self.nb.select())
        except (KeyError, tk.TclError):
            return None

    def _on_tab_changed(self, _event=None):
        """Render the selected Dynamics figure when its tab is opened."""
        if (self._current_tab() is self.tab_dynamics and self.emb is not None
                and self._dyn_shown != self.var_dyn_fig.get()):
            self._on_dyn_change()

    def _on_dyn_change(self, _event=None):
        """Dynamics figure selected: show it from the cache or build it in a worker."""
        if self.emb is None:
            return
        name = self.var_dyn_fig.get()
        if name not in self._dyn_map:
            return
        if name in self._dyn_cache:
            self._show_dyn(self.emb, name, *self._dyn_cache[name], cached=True)
            return
        if self._view_busy["dynamics"]:
            self._view_pending["dynamics"] = name
            return
        self._view_busy["dynamics"] = True
        self.panes["dynamics"].clear(f"Computing '{name}'...")
        if not self.busy:
            self._set_status(f"Computing the {name.lower()} figure...", dot="run")
            self._progress_indeterminate()
        kind, beh_name = self._dyn_map[name]
        threading.Thread(target=self._dyn_worker,
                         args=(self.emb, name, kind, beh_name, self.beh, list(self.circular),
                               self.run_method),
                         daemon=True, name="nemba-dynamics").start()

    def _dyn_worker(self, emb, name, kind, beh_name, beh, circ, method):
        """Worker: build one Dynamics figure and its statistics."""
        try:
            fig, rows, cap = _dynamics_figure(emb, kind, beh_name, beh, circ, method)
            self._post("call", lambda: self._show_dyn(emb, name, fig, rows, cap))
        except Exception as exc:  # noqa: BLE001
            msg, tb = f"{type(exc).__name__}: {exc}", traceback.format_exc()
            self._post("error", (f"dynamics view ({name}): {msg}", tb))

            def _failed():
                """Main thread: explain the failure in the pane, release the view slot."""
                if emb is self.emb:
                    self.panes["dynamics"].clear(f"'{name}' could not be drawn:\n{msg}")
                self._view_done("dynamics")
            self._post("call", _failed)

    def _show_dyn(self, emb, name, fig, rows, cap, cached=False):
        """Main thread: display a Dynamics figure, its statistics and reading guide."""
        if emb is self.emb:
            self._dyn_cache[name] = (fig, rows, cap)
            if self.var_dyn_fig.get() == name:
                self.panes["dynamics"].show(fig, caption=cap)
                self._dyn_shown = name
                self._fill_tree(self.tree_dyn, [(k, _fmt(v)) for k, v in rows])
                kind = self._dyn_map.get(name, ("", None))[0]
                self.lbl_dyn_help.configure(text=DYN_HELP.get(kind, ""))
        if not cached:
            self._view_done("dynamics")

    # ------------------------------------------------------------- advisor -- #
    def _advice_caption(self):
        """One-line summary of the recommendation shown above the report."""
        rec = self.rec
        kw = rec.embedder_kwargs
        extra = self._advice_extra()
        parts = [f"goal {rec.goal}", f"{kw.get('backbone')} ({kw.get('speed')})",
                 f"compare {', '.join(rec.compare)}"]
        if rec.warnings:
            parts.append(f"{len(rec.warnings)} warning{'s' if len(rec.warnings) > 1 else ''}")
        # (what "Run suggested" adds beyond the sidebar is written to the log)
        return "  |  ".join(parts)

    def _show_advice(self, res):
        """Main thread: show the profile figure and the formatted advice."""
        from .advisor import MODALITIES
        self.rec, self._rec_key = res.rec, res.key
        self.last_outdir = res.outdir
        p = res.rec.profile
        self.panes["advisor"].show(res.figure, caption=(
            f"{res.name}  |  {MODALITIES.get(p.modality, p.modality)}  |  {p.n_channels} ch x "
            f"{p.n_frames:,} frames  |  profiled in {p.seconds:.1f} s"))
        self.report.set_markdown(res.report)
        self._refresh_advice_buttons()
        self._on_data_change()                       # the File hint names the modality
        self.nb.select(self.tab_advisor)

    # ------------------------------------------------------------- utilities --
    def style_toolbar(self, tb):
        """Flatten matplotlib's Tk toolbar so it matches the panel style."""
        P = PALETTE
        try:
            tb.configure(bg=P["panel"], bd=0, highlightthickness=0)
        except tk.TclError:
            pass
        for w in tb.winfo_children():
            try:
                w.configure(bg=P["panel"])
            except tk.TclError:
                pass
            if isinstance(w, (tk.Button, tk.Checkbutton)):
                try:
                    w.configure(relief="flat", overrelief="flat", bd=0, highlightthickness=0,
                                activebackground=P["hover"], cursor="hand2", padx=self.px(4),
                                pady=self.px(3))
                    if isinstance(w, tk.Checkbutton):
                        w.configure(offrelief="flat", selectcolor=P["accent_soft"])
                except tk.TclError:
                    pass
                w.bind("<Enter>", lambda e: e.widget.configure(bg=PALETTE["hover"]), add="+")
                w.bind("<Leave>", lambda e: e.widget.configure(bg=PALETTE["panel"]), add="+")
            elif isinstance(w, tk.Label):
                try:
                    w.configure(fg=P["muted"], font=self.f_small)
                except tk.TclError:
                    pass
            elif isinstance(w, tk.Frame):           # separators between button groups
                try:
                    w.configure(bg=P["border"])
                except tk.TclError:
                    pass

    def _latest_explorer(self):
        """Newest '*_explorer.html' in the last output folder or the chosen one."""
        found = []
        for d in {self.last_outdir, self.var_out.get().strip()}:
            if d and os.path.isdir(d):
                found += list(Path(d).glob("*_explorer.html"))
        return max(found, key=lambda p: p.stat().st_mtime) if found else None

    def _refresh_explorer_button(self):
        """Enable 'Open explorer' only when an explorer file exists."""
        try:
            has = self._latest_explorer() is not None
        except OSError:
            has = False
        self.btn_html.state(["!disabled"] if has else ["disabled"])

    def open_output(self):
        """Open the last output folder in the file manager."""
        path = self.last_outdir or self.var_out.get().strip()
        if not path or not os.path.isdir(path):
            self._set_status("The output folder does not exist yet: run a model first.",
                             dot="warn")
            return
        try:
            os.startfile(path)                       # Windows
        except AttributeError:
            webbrowser.open(Path(path).resolve().as_uri())
        except OSError as exc:
            self._set_status(f"Could not open the folder: {exc}", dot="bad", error=True)

    def open_explorer(self):
        """Open the newest interactive HTML explorer in the web browser."""
        p = self._latest_explorer()
        if p is None:
            self._set_status("No explorer yet: run with 'HTML explorer' ticked.", dot="warn")
            return
        webbrowser.open(p.resolve().as_uri())

    def request_gc(self):
        """Schedule one garbage collection on the main thread (coalesced).

        Belt and braces for ``FigurePane._destroy_canvas``: any remaining
        cycle holding Tk objects is collected here, on the main thread, before
        a busy worker thread can trigger the collector."""
        if getattr(self, "_gc_pending", False):
            return
        self._gc_pending = True

        def _collect():
            """Run the collector (idle callback on the main thread)."""
            self._gc_pending = False
            gc.collect()
        self.root.after_idle(_collect)

    def _on_close(self):
        """Save the fields, cancel any job and close the window."""
        self._save_settings()
        self._stop.set()
        try:
            self.root.after_cancel(self._poll_id)
        except (tk.TclError, AttributeError):
            pass
        self.root.destroy()


def launch():
    """Open the NEMBA window (blocks until it is closed)."""
    enable_dpi_awareness()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
