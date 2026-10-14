"""NEMBA desktop GUI: fit a backbone, explore the embedding, decode behaviour.

Usage
-----
    python main.py                          (no arguments opens this window)
    python -m embedders_backbones.gui

    Sidebar (left, top to bottom)
      Data       recording file (empty = synthetic demo) and how to read it
      Behaviour  optional variables to decode and colour by (.csv / .npz / .npy)
      Model      backbone, speed preset, objective, states, step budget
      Output     folder for figures (png/pdf/svg), CSV tables, GIF, HTML explorer
      Compare    comma list of backbones fitted side by side
    Run (Ctrl+R / F5) fits one backbone, Compare fits the list, Stop (Esc)
    cancels at the next validation check of the training loop.

    Tabs (right): Overview (dashboard), Embedding (2-D map, re-colourable),
    Decoding (held-out scores + table), Training (loss curves), Compare, Log.

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

import gc
import json
import os
import queue
import re
import sys
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
from .training import SPEED_PRESETS, OBJECTIVES
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
LABEL_COL_W = 84             # width of the label column in the sidebar cards
WINDOW_SIZE = (1380, 860)
MIN_SIZE = (1080, 680)
POLL_MS = 60                 # queue drain period of the main thread
MAX_LOG_LINES = 6000
# largest up-scaling of a figure above its native size (keeps fonts in proportion)
SCALE_CAP = dict(overview=1.35, embedding=1.45, decoding=1.6, training=1.9, compare=1.6)
EMBED_FIGSIZE = (6.2, 4.5)   # inches of the Embedding-tab scatter

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".nemba_gui.json")
INPUT_TYPES = ["spikes", "pose", "mask", "coco_mask", "fiber_photometry", "timeseries"]
TIME_AXES = ["1 - channels x time", "0 - time x channels"]
PROJECTIONS = ["umap", "pca", "tsne"]
PRETTY = {"umap": "UMAP", "pca": "PCA", "tsne": "t-SNE"}
SPEEDS = list(SPEED_PRESETS)
DEFAULT_COMPARE = "pca,tcn,tst,bilstm"
DEFAULTS = dict(
    file="", input_type="spikes", time_axis=TIME_AXES[0], bin_size="0.2", fs="", dff=False,
    mask_size="64", mask_key="", beh_file="", circular="", n_perm="200",
    backbone="tcn", speed="fast", objective="masked", n_states="auto", max_steps="",
    smooth="4.0", outdir=os.path.join(os.getcwd(), "embedder_out"), gif=True, html=True,
    gif_method="umap", compare=DEFAULT_COMPARE,
)
# CPU cost hint -> (text, palette colour)
COST_TEXT = {"instant": ("instant, no training", "ok"), "fast": ("fast on CPU", "ok"),
             "medium": ("medium cost on CPU", "warn"),
             "slow": ("slow on CPU, prefer speed Fast", "bad")}
DEMO_HINT = "Empty = synthetic demo with behaviour."
# which optional Data rows each input type needs (others are hidden)
DATA_ROWS = {"spikes": ("bin",), "pose": ("axis",), "timeseries": ("axis",),
             "mask": ("axis",), "coco_mask": ("mask", "mask_key", "fs"),
             "fiber_photometry": ("axis", "fs", "dff")}

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


@dataclass
class _CompareResult:
    """Everything a finished comparison job hands to the main thread."""
    cmp: object
    outdir: str
    figure: Figure
    view: tuple = ("umap", "state")


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
        size_in = np.asarray(fig.get_size_inches(), float)
        self._native_px = size_in * float(fig.dpi) * self.app.dpr      # physical pixels
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
        self._view_busy = {"embedding": False, "compare": False}
        self._view_pending = {"embedding": None, "compare": None}
        self.emb = None                 # last fitted Embedder
        self.beh = None                 # its behaviour, aligned on emb.times
        self.circular = []
        self.cmp = None                 # last Comparison
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
        self._refresh_explorer_button()
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
        self.var_beh = tk.StringVar(value=D["beh_file"])
        self.var_circ = tk.StringVar(value=D["circular"])
        self.var_nperm = tk.StringVar(value=D["n_perm"])
        self.var_backbone = tk.StringVar(value=self._bb_display.get(D["backbone"], ""))
        self.var_speed = tk.StringVar(value=D["speed"])
        self.var_objective = tk.StringVar(value=D["objective"])
        self.var_states = tk.StringVar(value=D["n_states"])
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
        # status bar
        self.var_status = tk.StringVar(value="Ready")
        self.var_elapsed = tk.StringVar(value="")
        # name in the settings file -> variable (backbone handled separately)
        self._persist = {
            "file": self.var_file, "input_type": self.var_type, "time_axis": self.var_taxis,
            "bin_size": self.var_bin, "fs": self.var_fs, "dff": self.var_dff,
            "mask_size": self.var_mask_size, "mask_key": self.var_mask_key,
            "beh_file": self.var_beh, "circular": self.var_circ, "n_perm": self.var_nperm,
            "speed": self.var_speed, "objective": self.var_objective, "n_states": self.var_states,
            "max_steps": self.var_max_steps, "smooth": self.var_smooth, "outdir": self.var_out,
            "gif": self.var_gif, "html": self.var_html, "gif_method": self.var_gif_method,
            "compare": self.var_compare,
        }

    # ------------------------------------------------------------- settings --
    def _apply_settings(self, values):
        """Load remembered values, ignoring unknown keys and invalid choices."""
        choices = {"input_type": INPUT_TYPES, "time_axis": TIME_AXES, "speed": SPEEDS,
                   "objective": list(OBJECTIVES), "gif_method": PROJECTIONS}
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
        shown only for the input types that use them (see ``DATA_ROWS``)."""
        c = self._card(form, "Data")
        e = self._entry(c, self.var_file)
        b = ttk.Button(c, text="Browse", style="Small.TButton", command=self._browse_data)
        self._row(c, 0, "File", e, b, name="file",
                  tip="Recording (.npy / .npz / .csv / .pkl, or a COCO .json for masks). "
                      "Leave empty for the synthetic demo.")
        self.hint_file = self._hint(c, 1, DEMO_HINT)
        self.cb_type = self._combo(c, self.var_type, INPUT_TYPES)
        self._row(c, 2, "Input type", self.cb_type, name="input_type",
                  tip="spikes = list of spike-time arrays; pose / mask / timeseries = matrix; "
                      "fiber_photometry = signals (optional dF/F); coco_mask = COCO json.")
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

    def _build_behaviour_card(self, form):
        """Behaviour section: optional file, circular names, null shuffles."""
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

    def _build_model_card(self, form):
        """Model section: backbone (+ CPU hint), speed chips, objective, K, steps."""
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
        self._row(c, 5, "States", self._combo(c, self.var_states,
                                              ["auto"] + [str(k) for k in range(2, 11)],
                                              readonly=False),
                  name="n_states",
                  tip="Number of discrete states K. 'auto' picks K in 2-10 by silhouette.")
        self._row(c, 6, "Max steps", self._entry(c, self.var_max_steps), name="max_steps",
                  tip="Override the preset step budget (blank = preset). Early stopping may "
                      "end training sooner.")
        self._row(c, 7, "Smoothing", self._entry(c, self.var_smooth), name="smooth",
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
                  name="gif_method", tip="2-D map used by the trajectory GIF and the HTML explorer.")
        box = ttk.Frame(c, style="Panel.TFrame")
        box.grid(row=2, column=1, columnspan=2, sticky="w", pady=(px(2), px(2)))
        ttk.Checkbutton(box, text="Trajectory GIF", variable=self.var_gif).pack(side="left")
        ttk.Checkbutton(box, text="HTML explorer", variable=self.var_html).pack(
            side="left", padx=(px(14), 0))
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
        """Notebook with the six result tabs."""
        px = self.px
        main = ttk.Frame(parent, style="App.TFrame", padding=(px(14), px(12), px(14), px(10)))
        self.nb = ttk.Notebook(main)
        self.nb.pack(fill="both", expand=True)
        self.panes = {}
        empty_run = ("No results yet.\nPress Run (Ctrl+R). With an empty data file the "
                     "synthetic demo is used.")
        self.panes["overview"] = FigurePane(self.nb, self, "Dashboard", empty_run,
                                            SCALE_CAP["overview"])
        self.nb.add(self.panes["overview"], text="Overview")
        self.panes["embedding"] = FigurePane(self.nb, self, "Embedding map", empty_run,
                                             SCALE_CAP["embedding"])
        self._build_embedding_controls(self.panes["embedding"].controls)
        self.nb.add(self.panes["embedding"], text="Embedding")
        self.tab_decoding = self._build_decoding_tab(self.nb)
        self.nb.add(self.tab_decoding, text="Decoding")
        self.panes["training"] = FigurePane(self.nb, self, "Training curves", empty_run,
                                            SCALE_CAP["training"])
        self.nb.add(self.panes["training"], text="Training")
        self.tab_compare = self._build_compare_tab(self.nb)
        self.nb.add(self.tab_compare, text="Compare")
        self.tab_log = self._build_log_tab(self.nb)
        self.nb.add(self.tab_log, text="Log")
        return main

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
            self.hint_file.configure(text=(f"{os.path.getsize(p) / 1e6:.1f} MB" if ok
                                           else "file not found"))

    def _on_backbone(self):
        """Update the CPU-cost hint under the backbone selector."""
        key = self._bb_key.get(self.var_backbone.get())
        if key is None:
            return
        text, col = COST_TEXT.get(CPU_COST_HINT.get(key, "fast"), COST_TEXT["fast"])
        self.lbl_cost.configure(text=text)
        self.lbl_cost_dot.configure(foreground=PALETTE[col])

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
        bb = self._bb_key.get(self.var_backbone.get())
        if bb is None:
            raise _FormError("choose a backbone", F.get("backbone"))
        cfg["backbone"] = bb
        cfg["speed"] = self.var_speed.get()
        cfg["objective"] = self.var_objective.get()
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
    def run(self):
        """Start a single-backbone job (Run button)."""
        if self.busy:
            return
        try:
            cfg = self._collect_config()
        except _FormError as err:
            self._form_error(err)
            return
        self._begin_job("run", self._run_worker, cfg,
                        f"{label(cfg['backbone'])} on "
                        f"{os.path.basename(cfg['file']) or 'synthetic demo'}")

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
        self._stop.clear()
        self.btn_run.state(["disabled"])
        self.btn_cmp.state(["disabled"])
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
        self.btn_run.state(["!disabled"])
        self.btn_cmp.state(["!disabled"])
        self.btn_stop.state(["disabled"])
        self.var_elapsed.set(f"{el:.1f} s")
        what = "Comparison" if kind == "compare" else "Run"
        if status == "done":
            self._progress_determinate(100)
            self._set_status(f"{what} finished in {el:.1f} s  |  outputs in "
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
        """Worker: read the recording + behaviour (or build the demo)."""
        out = dict(behavior=None, behavior_times=None, circular=list(cfg["circular"]))
        if not cfg["file"]:
            rep.phase("simulating the synthetic demo recording")
            ds = make_dataset(seed=0)
            out.update(kind="spikes", payload=ds["spike_times"], bin_size=float(ds["dt"]),
                       name="synthetic demo")
            if not cfg["beh_file"]:
                out["behavior"], out["behavior_times"] = ds["behavior"], ds["times"]
                if "phase" not in out["circular"]:
                    out["circular"].append("phase")
        else:
            path, typ = cfg["file"], cfg["input_type"]
            rep.phase(f"loading {os.path.basename(path)}")
            if typ == "spikes":
                kind, payload = "spikes", _data.load_spikes(path)
            elif typ == "coco_mask" or (typ == "mask" and path.lower().endswith(".json")):
                kind, payload = "coco_mask", path
            else:
                kind, payload = typ, _data.load_array(path)
            out.update(kind=kind, payload=payload, bin_size=cfg["bin_size"],
                       name=os.path.basename(path))
        if cfg["beh_file"]:
            out["behavior"], out["behavior_times"] = _data.load_behavior(cfg["beh_file"])
        rep.check()
        return out

    @staticmethod
    def _fit(emb, data, cfg):
        """Worker: dispatch to the right ``Embedder.fit_*`` for the input kind."""
        kind, p = data["kind"], data["payload"]
        if kind == "spikes":
            emb.fit_spikes(p)
        elif kind == "coco_mask":
            s = cfg["mask_size"]
            emb.fit_coco_masks(p, resize=(s, s), channel_key=cfg["mask_key"], fps=cfg["fs"])
        elif kind == "fiber_photometry":
            emb.fit_fiber(p, time_axis=cfg["time_axis"], dff=cfg["dff"], fs=cfg["fs"])
        else:
            emb.fit_features(p, time_axis=cfg["time_axis"])

    def _run_worker(self, cfg):
        """Worker: fit one backbone, write outputs, build every figure."""
        bb = cfg["backbone"]
        rep = _Reporter(self._post, self._stop, [bb], "embedding + automatic state selection")
        data = self._load_data(cfg, rep)
        emb = Embedder(bb, bin_size=data["bin_size"], n_states=cfg["n_states"],
                       speed=cfg["speed"], objective=cfg["objective"],
                       max_steps=cfg["max_steps"], smooth_sigma=cfg["smooth"],
                       log=rep.log, progress=rep.progress)
        rep.phase(f"{bb}: preparing input and model")
        self._fit(emb, data, cfg)

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
        what = "decoding behaviour, " if beh is not None else ""
        rep.phase(f"{bb}: {what}writing figures, tables, GIF and explorer  ({n_st - 1}/{n_st})")
        met = emb.run(cfg["outdir"], behavior=beh, behavior_times=bt, animate=cfg["gif"],
                      gif_method=cfg["gif_method"], html=cfg["html"], circular=circ,
                      n_perm=cfg["n_perm"])
        rep.check()
        rep.phase(f"{bb}: building the views  ({n_st}/{n_st})")
        aligned = emb._align(beh, bt, None, circ) if beh is not None else None
        color = f"beh:{next(iter(aligned))}" if aligned else "time"
        method = cfg["view_method"] if cfg["view_method"] in PROJECTIONS else "umap"
        figs = dict(
            overview=viz.dashboard(emb, behavior=aligned, circular=circ),
            embedding=_embedding_figure(emb, method, color, aligned, circ),
            decoding=(viz.decoding_figure(emb.decoding, emb.times)
                      if emb.decoding is not None else None),
            training=viz.training_figure(emb.history, title=f"{label(bb)} training"),
        )
        self._log_metrics(rep, met)
        res = _RunResult(emb=emb, behaviour=aligned, circular=circ, outdir=cfg["outdir"],
                         figures=figs, metrics=met, view=(method, color))
        self._post("call", lambda: self._show_run(res))

    @staticmethod
    def _log_metrics(rep, met):
        """Worker: one compact line with the headline metrics."""
        keys = ("n_states", "silhouette", "cluster_structure", "circularity",
                "participation_ratio", "train_steps", "train_seconds", "total_seconds")
        parts = [f"{k}={_fmt(met[k])}" for k in keys if k in met]
        rep.post("log", ("metrics: " + "  ".join(parts), "ok"))

    def _compare_worker(self, cfg):
        """Worker: fit every listed backbone, write the comparison, build the grid."""
        bbs = cfg["compare"]
        rep = _Reporter(self._post, self._stop, bbs, "embedding, auto K and decoding")
        data = self._load_data(cfg, rep)
        s = cfg["mask_size"]
        cmp = compare_embedders(
            data["payload"], input=data["kind"], backbones=bbs, time_axis=cfg["time_axis"],
            fs=cfg["fs"], dff=cfg["dff"], behavior=data["behavior"],
            behavior_times=data["behavior_times"], circular=data["circular"], log=rep.log,
            mask_resize=(s, s), mask_channel_key=cfg["mask_key"], n_perm=cfg["n_perm"],
            bin_size=data["bin_size"], n_states=cfg["n_states"], speed=cfg["speed"],
            objective=cfg["objective"], max_steps=cfg["max_steps"],
            smooth_sigma=cfg["smooth"], progress=rep.progress)
        rep.check()
        outdir = os.path.join(cfg["outdir"], "compare")
        rep.phase("writing comparison figures and tables (UMAP of every backbone)")
        cmp.run(outdir)
        rep.check()
        rep.phase("building the comparison grid")
        method = cfg["cmp_method"] if cfg["cmp_method"] in PROJECTIONS else "umap"
        color = cfg["cmp_color"] if cfg["cmp_color"] in ("state", "time") else "state"
        fig = viz.compare_embeddings_figure(cmp.embedders, method=method, color=color)
        rep.post("log", ("\n" + cmp.metrics.round(3).to_string(index=False), None))
        best = cmp.best("silhouette")
        rep.post("log", (f"best by silhouette: {best}", "ok"))
        if cmp.decoding is not None:
            for tgt in cmp.decoding["target"].unique():
                rep.post("log", (f"best held-out decoding of {tgt}: "
                                 f"{cmp.best('decoding:' + str(tgt))}", "ok"))
        res = _CompareResult(cmp=cmp, outdir=outdir, figure=fig, view=(method, color))
        self._post("call", lambda: self._show_compare(res))

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
        self.last_outdir = res.outdir
        P = self.panes
        struct = m.get("cluster_structure", "undefined")
        cap = (f"silhouette {_fmt(m.get('silhouette'))} ("
               + ("no" if struct == "none" else struct) + " cluster structure)")
        if emb.decoding is not None and len(emb.decoding.table):
            te = emb.decoding.table[emb.decoding.table.features == "embedding"]
            cap += "  |  decoding " + ", ".join(f"{r.target} {r.score:.2f}"
                                                 for r in te.itertuples())
        P["overview"].show(res.figures["overview"], caption=cap)
        self._set_color_choices(res.behaviour, res.view[1])
        self.var_view_proj.set(res.view[0])
        P["embedding"].show(res.figures["embedding"], caption=self._embed_caption(emb, *res.view))
        if res.figures["decoding"] is not None:
            tab = emb.decoding.table
            P["decoding"].show(res.figures["decoding"], caption=(
                f"{tab['target'].nunique()} targets  |  blocked {int(tab['n_folds'].max())}-fold CV"
                f"  |  {int(tab['n_perm'].max())} circular-shift shuffles"))
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
        self.nb.select(P["overview"])

    def _fill_decoding(self, tab):
        """Fill the Decoding table (grey rows: Holm p >= 0.05)."""
        rows, tags = [], []
        for r in tab.itertuples(index=False):
            feats = "embedding" if r.features == "embedding" else "raw PCA"
            rows.append((r.target, feats, r.metric, _fmt(r.score), _fmt(r.null_p95),
                         _fmt_p(r.p_holm)))
            ok = np.isfinite(r.p_holm) and r.p_holm < 0.05
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
        """Main thread: display a re-rendered comparison grid."""
        if cmp is self.cmp:
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
            (self._on_view_change if which == "embedding" else self._on_cmp_view_change)()

    def _show_compare(self, res):
        """Main thread: put a finished comparison on the Compare tab."""
        cmp = res.cmp
        self.cmp = cmp
        self.last_outdir = res.outdir
        self.panes["compare"].show(res.figure, caption=(
            f"{len(cmp.embedders)} backbones  |  {_short_path(res.outdir, 2)}"))
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
