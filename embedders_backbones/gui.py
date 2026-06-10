"""A small Tkinter GUI for embedders_backbones.

Pick an input file + type + backbone, press Run; training happens on a background
thread and the embedding (PCA/t-SNE/UMAP) is displayed in the window and saved.
Tkinter ships with Python, so there is no extra dependency.
"""

from __future__ import annotations

import os
import threading
import traceback

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import data as _data
from .core import Embedder
from .backbones import list_backbones, label

INPUT_TYPES = ["spikes", "pose", "mask", "fiber_photometry", "timeseries"]
PROJECTIONS = ["umap", "tsne", "pca"]


class App:
    def __init__(self, root):
        self.root = root
        root.title("embedders_backbones")
        root.geometry("1180x760")
        self.emb = None

        left = ttk.Frame(root, padding=10); left.pack(side="left", fill="y")
        right = ttk.Frame(root, padding=6); right.pack(side="right", fill="both", expand=True)

        # --- controls ---
        self.var_file = tk.StringVar()
        self.var_type = tk.StringVar(value="spikes")
        self.var_back = tk.StringVar(value="dtc")
        self.var_proj = tk.StringVar(value="umap")
        self.var_color = tk.StringVar(value="time")
        self.var_states = tk.StringVar(value="auto")
        self.var_epochs = tk.IntVar(value=80)
        self.var_bin = tk.DoubleVar(value=0.2)
        self.var_taxis = tk.IntVar(value=1)
        self.var_out = tk.StringVar(value=os.path.join(os.getcwd(), "embedder_out"))

        r = 0
        ttk.Label(left, text="Input file").grid(row=r, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.var_file, width=30).grid(row=r, column=1)
        ttk.Button(left, text="...", width=3, command=self._browse).grid(row=r, column=2); r += 1
        for lbl, var, vals in [("Input type", self.var_type, INPUT_TYPES),
                               ("Backbone", self.var_back, list_backbones()),
                               ("Projection", self.var_proj, PROJECTIONS),
                               ("Colour by", self.var_color, ["time", "state"])]:
            ttk.Label(left, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Combobox(left, textvariable=var, values=vals, width=27,
                         state="readonly").grid(row=r, column=1, columnspan=2, sticky="w"); r += 1
        for lbl, var in [("# states (auto/int)", self.var_states), ("Epochs", self.var_epochs),
                         ("Bin size s (spikes)", self.var_bin), ("Time axis (0/1)", self.var_taxis)]:
            ttk.Label(left, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Entry(left, textvariable=var, width=12).grid(row=r, column=1, sticky="w"); r += 1
        ttk.Label(left, text="Output dir").grid(row=r, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.var_out, width=30).grid(row=r, column=1)
        ttk.Button(left, text="...", width=3, command=self._browse_out).grid(row=r, column=2); r += 1

        self.btn = ttk.Button(left, text="Run (single backbone)", command=self._run)
        self.btn.grid(row=r, column=0, columnspan=3, sticky="we", pady=6); r += 1
        ttk.Button(left, text="Re-plot projection",
                   command=self._replot).grid(row=r, column=0, columnspan=3, sticky="we"); r += 1
        ttk.Separator(left, orient="horizontal").grid(row=r, column=0, columnspan=3,
                                                      sticky="we", pady=6); r += 1
        ttk.Label(left, text="Compare backbones").grid(row=r, column=0, sticky="w")
        self.var_compare = tk.StringVar(value="dtc,tst,bilstm")
        ttk.Entry(left, textvariable=self.var_compare, width=27).grid(
            row=r, column=1, columnspan=2, sticky="w"); r += 1
        self.btn_cmp = ttk.Button(left, text="Compare", command=self._compare)
        self.btn_cmp.grid(row=r, column=0, columnspan=3, sticky="we"); r += 1

        self.log = tk.Text(left, width=44, height=18, font=("Consolas", 8))
        self.log.grid(row=r, column=0, columnspan=3, pady=6)

        # --- figure ---
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._msg(f"{len(list_backbones())} backbones available. Pick a file and press Run.\n"
                  "(demo: leave file empty -> a synthetic ring dataset)")

    # ---- helpers ----
    def _browse(self):
        p = filedialog.askopenfilename(filetypes=[("data", "*.npy *.npz *.csv *.pkl"), ("all", "*.*")])
        if p:
            self.var_file.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory()
        if p:
            self.var_out.set(p)

    def _msg(self, s):
        self.log.insert("end", s + "\n"); self.log.see("end"); self.root.update_idletasks()

    def _load_input(self):
        path = self.var_file.get().strip()
        typ = self.var_type.get()
        if not path:
            return ("demo", None)
        if typ == "spikes":
            return ("spikes", _data.load_spikes(path))
        return (typ, _data.load_array(path))

    # ---- run ----
    def _run(self):
        self.btn.config(state="disabled")
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self):
        try:
            kind, payload = self._load_input()
            ns = self.var_states.get().strip() or "auto"
            emb = Embedder(backbone=self.var_back.get(), bin_size=self.var_bin.get(),
                           n_states=ns, epochs=self.var_epochs.get(),
                           device="auto", log=self._msg)
            self._msg(f"== {label(self.var_back.get())} on {kind} ==")
            if kind == "demo":
                emb.fit_spikes(_demo_spikes())
            elif kind == "spikes":
                emb.fit_spikes(payload)
            elif kind == "fiber_photometry":
                emb.fit_fiber(payload, time_axis=self.var_taxis.get())
            else:
                emb.fit_features(payload, time_axis=self.var_taxis.get())
            self.emb = emb
            out = self.var_out.get()
            emb.run(out, gif_method=self.var_proj.get())
            self._msg("metrics: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                               for k, v in emb.metrics().items()))
            self._draw()
            self._msg(f"saved figures + gif + metrics to {out}")
        except Exception as exc:  # noqa: BLE001
            self._msg("ERROR: " + repr(exc)); self._msg(traceback.format_exc())
        finally:
            self.root.after(0, lambda: self.btn.config(state="normal"))

    def _replot(self):
        if self.emb is not None:
            self._draw()

    def _compare(self):
        self.btn_cmp.config(state="disabled")
        threading.Thread(target=self._compare_worker, daemon=True).start()

    def _compare_worker(self):
        try:
            from .compare import compare_embedders
            from . import viz as V
            bbs = [b.strip() for b in self.var_compare.get().split(",") if b.strip()]
            kind, payload = self._load_input()
            data = _demo_spikes() if kind == "demo" else payload
            ikind = "spikes" if kind in ("demo", "spikes") else kind
            ns = self.var_states.get().strip() or "auto"
            self._msg(f"== comparing {bbs} on {ikind} ==")
            cmp = compare_embedders(data, input=ikind, backbones=bbs,
                                    time_axis=self.var_taxis.get(), n_states=ns,
                                    epochs=self.var_epochs.get(), device="auto", log=self._msg)
            out = os.path.join(self.var_out.get(), "compare"); cmp.run(out)
            self._cmp = cmp
            self._msg("\n" + cmp.metrics.round(3).to_string(index=False))
            self._msg(f"best by silhouette: {cmp.best('silhouette')}")
            self.fig.clear()
            V.compare_embeddings_figure(cmp.embedders, method=self.var_proj.get(),
                                        color=self.var_color.get(), fig=self.fig)
            self.canvas.draw()
            self._msg(f"saved comparison (metrics + figures) to {out}")
        except Exception as exc:  # noqa: BLE001
            self._msg("ERROR: " + repr(exc)); self._msg(traceback.format_exc())
        finally:
            self.root.after(0, lambda: self.btn_cmp.config(state="normal"))

    def _draw(self):
        self.fig.clear()
        self.emb.figure(methods=(self.var_proj.get(),), color=self.var_color.get(), fig=self.fig)
        self.canvas.draw()


def _demo_spikes(n=40, dur=400.0, dt=0.05, seed=0):
    rng = np.random.default_rng(seed); T = int(dur / dt); t = np.arange(T) * dt
    phase = 2 * np.pi * 0.02 * t + 0.3 * np.sin(2 * np.pi * 0.005 * t)
    pref = rng.uniform(0, 2 * np.pi, n)
    rates = 2 + 8 * np.exp(2.0 * (np.cos(phase[None] - pref[:, None]) - 1))
    counts = rng.poisson(rates * dt)
    return [np.sort(t[np.repeat(np.arange(T), counts[j])] + rng.uniform(0, dt, counts[j].sum()))
            for j in range(n)]


def launch():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
