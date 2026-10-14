"""The unified ``Embedder`` API: pick a backbone, fit it to any modality, get
embeddings / states / projections / metrics / decoding, and render figures.

    from embedders_backbones import Embedder
    emb = Embedder(backbone="tcn", speed="fast")          # CPU-friendly preset
    emb.fit_spikes(spike_times)              # or fit_features / fit_fiber / fit_matrix
    emb.decode({"x": x, "speed": v})         # cross-validated decoding + null test
    emb.run("out_dir", behavior={"x": x})    # figures (png/pdf/svg) + csv + gif + html
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import backbones as BK
from . import data as _data
from . import analysis as _an
from . import training as _tr
from . import decoding as _dec


class Embedder:
    """Self-supervised sequence embedder for one recording.

    Main arguments
    --------------
    backbone        one of ``list_backbones()`` ('pca' is the no-training baseline)
    speed           'fast' | 'balanced' | 'accurate' (step budget, window, width,
                    wall-clock cap; see ``training.SPEED_PRESETS``)
    objective       'masked' | 'contrastive' | 'hybrid' (see ``training``)
    n_states        None/'auto' -> chosen by silhouette in [k_min, k_max]
    input_pca       'auto' (64 PCs when > 128 channels) | int | None
    smooth_sigma    Gaussian smoothing of embeddings in frames (0 = off)
    likelihood      'gaussian' | 'poisson' (raw counts, spikes) | 'auto'
                    ('poisson' when raw counts are available)
    mask_mode       'time' | 'neuron' | 'both' (what the masked objective hides)
    state_model     'auto' (backbone default: K-means / DEC / CRF) | 'kmeans' |
                    'hmm' | 'arhmm' (temporal state models, K by held-out
                    likelihood when n_states is None)
    max_steps / epochs / win_len / batch / time_budget_s override the preset.
    """

    def __init__(self, backbone="dtc", *, bin_size=0.2, embedding_dim=None, n_states=None,
                 k_min=2, k_max=10, speed="fast", max_steps=None, epochs=None,
                 win_len=None, stride=None, batch=None, objective="masked",
                 contrastive_weight=0.5, temperature=0.2, contrastive_offset=10,
                 mask_ratio=0.4, mask_span=16, val_frac=0.1, lr=1e-3,
                 input_pca="auto", smooth_sigma=4.0, device="auto", n_threads=None,
                 time_budget_s="preset", seed=0, log=print, progress=None, model_kwargs=None,
                 likelihood="gaussian", mask_mode="time", neuron_frac=0.25, state_model="auto"):
        if backbone not in BK.BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; choose {BK.list_backbones()}")
        self.backbone = backbone
        self.bin_size = bin_size
        # None -> backbone default (32; 8 for the low-dimensional GPFA / CEBRA)
        self.embedding_dim = int(embedding_dim if embedding_dim is not None
                                 else BK.DEFAULT_EMBEDDING_DIM.get(backbone, 32))
        # n_states None / "auto" -> pick K by silhouette over [k_min, k_max]
        self.n_states = None if (n_states in (None, "auto", 0, "")) else int(n_states)
        self.k_min, self.k_max = k_min, k_max
        self.speed = speed
        self.settings = _tr.resolve_settings(speed, max_steps=max_steps, win_len=win_len, batch=batch)
        self.max_steps, self.epochs = max_steps, epochs
        self.win_len = int(self.settings["win_len"])
        self.stride, self.batch = stride, batch
        self.objective = objective
        self.contrastive_weight, self.temperature = contrastive_weight, temperature
        self.contrastive_offset = contrastive_offset
        self.mask_ratio, self.mask_span = mask_ratio, mask_span
        self.val_frac, self.lr = val_frac, lr
        self.input_pca = input_pca
        self.smooth_sigma = smooth_sigma
        self.device, self.n_threads = device, n_threads
        self.time_budget_s = time_budget_s
        self.seed = seed
        self.log, self.progress = log, progress
        self.model_kwargs = dict(model_kwargs or {})
        self.likelihood, self.mask_mode, self.neuron_frac = likelihood, mask_mode, neuron_frac
        self.state_model = state_model
        self.likelihood_used = None
        self.counts = None
        self.state_info = None
        self.cosmoothing = None

        self.n_states_used = None
        self.silhouette_scores = None
        self.X = self.X_input = self.times = self.net = None
        self.Z = self.Z_raw = self.states = self.losses = None
        self.history = None
        self.reducer = None
        self.decoding = None
        self.timings = {}
        self._z_stats = None
        self._proj = {}

    # ----- fitting from different inputs ----- #
    def fit_spikes(self, spike_times, t_start=None, t_end=None, top_n=None):
        X, times, counts, keep = _data.spikes_to_matrix(spike_times, self.bin_size, t_start,
                                                        t_end, top_n, return_index=True)
        self.channel_index = keep
        return self._fit(X, times, counts=counts)

    def fit_features(self, X, time_axis=1, times=None, counts=None):
        """Generic [features, time] matrix. Non-negative integer data (binned
        counts) are kept as raw counts automatically (for the Poisson
        likelihood and co-smoothing)."""
        if counts is None:
            raw = np.asarray(X, dtype=np.float64)
            raw = raw[None, :] if raw.ndim == 1 else (raw.T if time_axis == 0 else raw)
            fin = raw[np.isfinite(raw)]
            if fin.size and (fin >= 0).all() and np.allclose(fin, np.round(fin)) and fin.max() > 1:
                counts = np.nan_to_num(raw)
        Xz, times = _data.features_to_matrix(X, time_axis=time_axis, times=times)
        return self._fit(Xz, times, counts=counts)

    def fit_coco_masks(self, coco, *, resize=(64, 64), channel_key=None,
                       category_ids=None, track_ids=None, fps=None,
                       standardize=True):
        X, times = _data.coco_masks_to_matrix(
            coco, resize=resize, channel_key=channel_key,
            category_ids=category_ids, track_ids=track_ids, fps=fps,
            standardize=standardize)
        return self._fit(X, times)

    def fit_fiber(self, signal, time_axis=1, times=None, dff=False, fs=None, control=None):
        Xz, times = _data.fiber_photometry_to_matrix(signal, time_axis, times, dff, fs,
                                                     control=control)
        return self._fit(Xz, times)

    def fit_matrix(self, X, times=None, counts=None):
        """Model-ready matrix X [C, T] (already scaled); optional raw counts."""
        X = np.asarray(X, dtype=np.float32)
        if times is None:
            times = np.arange(X.shape[1], dtype=float) * self.bin_size
        return self._fit(X, np.asarray(times, float), counts=counts)

    def _fit(self, X, times, counts=None):
        t0 = time.perf_counter()
        self.X_input = np.asarray(X, np.float32)
        self.times = np.asarray(times, float)
        self.counts = None if counts is None else np.asarray(counts, np.float32)
        lik = self.likelihood
        if lik == "auto":
            lik = "poisson" if self.counts is not None else "gaussian"
        if lik == "poisson" and self.counts is None:
            raise ValueError("likelihood='poisson' needs raw counts (fit_spikes, integer "
                             "data in fit_features, or counts= in fit_matrix)")
        self.likelihood_used = lik
        input_pca = self.input_pca
        if lik == "poisson" and _data.InputReducer(input_pca).resolve(X.shape[0]) is not None:
            if self.log:
                self.log(f"  [{self.backbone}] Poisson likelihood reconstructs every channel: "
                         "input PCA disabled")
            input_pca = None
        self.reducer = _data.InputReducer(input_pca, seed=self.seed)
        self.X = self.reducer.fit_transform(self.X_input)
        if self.log:
            red = (f" -> {self.X.shape[0]} PCs ({100 * self.reducer.explained_variance:.1f}% var)"
                   if self.reducer.pca is not None else "")
            self.log(f"[{self.backbone}] fitting on {self.X_input.shape[0]} channels{red} x "
                     f"{self.X.shape[1]} frames  (speed={self.speed}, objective={self.objective})")
        self.timings["reduce_s"] = time.perf_counter() - t0

        if self.backbone == "cebra" and "max_iterations" not in self.model_kwargs:
            self.model_kwargs = dict(self.model_kwargs,
                                     max_iterations=BK.CEBRA_ITERATIONS.get(self.speed, 500))
        self.net, self.history = _tr.train(
            self.backbone, self.X, embedding_dim=self.embedding_dim, speed=self.speed,
            max_steps=self.max_steps, epochs=self.epochs, win_len=self.win_len,
            batch=self.batch, stride=self.stride, lr=self.lr, mask_ratio=self.mask_ratio,
            mask_span=self.mask_span, objective=self.objective,
            contrastive_weight=self.contrastive_weight, temperature=self.temperature,
            contrastive_offset=self.contrastive_offset, val_frac=self.val_frac,
            device=self.device, seed=self.seed, log=self.log, model_kwargs=self.model_kwargs,
            n_threads=self.n_threads, time_budget_s=self.time_budget_s, progress=self.progress,
            likelihood=lik, mask_mode=self.mask_mode, neuron_frac=self.neuron_frac,
            counts=self.counts if lik == "poisson" else None)
        self.losses = self.history["train_loss"]
        self.timings["train_s"] = self.history["seconds"]

        t1 = time.perf_counter()
        self.Z_raw = _tr.embed_sequence(self.net, self.X, win_len=self.history["win_len"],
                                        device=self.device,
                                        normalize=self.objective == "contrastive")
        self.Z, mu, sd = _an.smooth_standardize(self.Z_raw, self.smooth_sigma, return_stats=True)
        self._z_stats = (mu, sd)
        self.timings["embed_s"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        if self.state_model in ("hmm", "arhmm"):
            self._fit_temporal_states()
            self.timings["cluster_s"] = time.perf_counter() - t2
            self.timings["total_s"] = time.perf_counter() - t0
            self._proj = {}
            self.decoding = None
            return self
        if self.n_states is None:
            k, self.silhouette_scores = _an.auto_k(self.Z, self.k_min, self.k_max, self.seed)
            self.n_states_used = k
            if self.log:
                s_best = self.silhouette_scores[k]
                self.log(f"[{self.backbone}] auto K by silhouette -> K={k} "
                         f"(sil={s_best:.3f}, {_an.cluster_structure(s_best)} structure)")
                if s_best < 0.25:
                    self.log("  WARNING: silhouette < 0.25 means no substantial cluster structure; "
                             "the states are an arbitrary discretisation of a continuous embedding")
                if k == max(self.silhouette_scores):
                    self.log(f"  WARNING: K reached the upper bound k_max={self.k_max}; the "
                             "silhouette keeps rising, typical of a continuous manifold. "
                             "Prefer a fixed n_states or decoding over state counts.")
        else:
            self.n_states_used = self.n_states
        self.states = _an.cluster(self.backbone, self.Z, self.n_states_used, self.seed)
        self.timings["cluster_s"] = time.perf_counter() - t2
        self.timings["total_s"] = time.perf_counter() - t0
        self._proj = {}
        self.decoding = None
        return self

    def _fit_temporal_states(self):
        """Discrete states from a temporal model (sticky Gaussian HMM or AR-HMM)
        on the embedding; K chosen by held-out likelihood when n_states is None."""
        from .states import fit_states
        # The temporal model gets the UNSMOOTHED embedding: it does its own
        # temporal integration. On Gaussian-smoothed features the AR-HMM ends up
        # modelling the smoothing kernel (validation: ARI 0.36 smoothed vs 1.00
        # raw, benchmarks/results/states_validation.*). fit_states z-scores.
        res = fit_states(self.Z_raw, method=self.state_model, k=self.n_states,
                         k_range=(self.k_min, self.k_max), seed=self.seed, log=self.log)
        self.state_info = res
        self.states = np.asarray(res["labels"], dtype=int)
        self.n_states_used = int(res["k"])
        self.silhouette_scores = None
        if self.log:
            how = "held-out likelihood" if self.n_states is None else "user"
            self.log(f"[{self.backbone}] {self.state_model.upper()} states: K={self.n_states_used} "
                     f"(chosen by {how})")
            sel = res.get("selection")
            if sel is not None and getattr(sel, "attrs", {}).get("at_upper_bound"):
                self.log(f"  WARNING: K reached the upper bound k_max={self.k_max}: held-out "
                         "likelihood keeps rising, typical of a continuous manifold. Prefer a "
                         "fixed n_states or decoding over state counts.")

    @classmethod
    def auto(cls, data, *, goal="auto", behavior=None, behavior_times=None, fs=None,
             log=print, **overrides):
        """Profile ``data``, follow the advisor's recommendation and fit.

        Returns the fitted Embedder; ``emb.recommendation.report()`` explains
        every choice. ``overrides`` replace any recommended Embedder setting."""
        from .advisor import advise
        rec = advise(data, goal=goal, behavior=behavior, behavior_times=behavior_times, fs=fs)
        if log:
            log(f"[advisor] {rec.profile.modality}: {rec.embedder_kwargs['backbone']} via "
                f"{rec.fit_method} (goal={rec.goal})")
        return rec.fit(data, log=log, **overrides)

    # ----- label-free evaluation ----- #
    def cosmooth(self, heldout_frac=0.2, test_frac=0.2, **overrides):
        """Co-smoothing score (bits/spike on held-out neurons and held-out time)
        with this embedder's settings; needs raw counts. Stored in
        ``self.cosmoothing`` and returned."""
        from .cosmoothing import cosmooth_score
        if self.counts is None:
            raise ValueError("co-smoothing needs raw counts (spikes or integer data)")
        kw = dict(speed=self.speed, embedding_dim=self.embedding_dim, objective=self.objective,
                  likelihood=self.likelihood, mask_mode=self.mask_mode,
                  neuron_frac=self.neuron_frac, smooth_sigma=self.smooth_sigma,
                  max_steps=self.max_steps, model_kwargs=self.model_kwargs,
                  time_budget_s=self.time_budget_s, input_pca=self.input_pca)
        kw.update(overrides)
        self.cosmoothing = cosmooth_score(self.backbone, self.counts, heldout_frac=heldout_frac,
                                          test_frac=test_frac, seed=self.seed, log=self.log, **kw)
        return self.cosmoothing

    # ----- new data / persistence ----- #
    def transform(self, X):
        """Embed a new model-ready matrix X [C, T] (same channels and the same
        preprocessing as the training input) with the trained encoder. Uses the
        training smoothing and standardisation, so embeddings are comparable."""
        self._check()
        from scipy.ndimage import gaussian_filter1d
        Xr = self.reducer.transform(np.asarray(X, np.float32))
        Z = _tr.embed_sequence(self.net, Xr, win_len=self.history["win_len"], device=self.device,
                               normalize=self.objective == "contrastive")
        if self.smooth_sigma and self.smooth_sigma > 0:
            Z = gaussian_filter1d(Z, self.smooth_sigma, axis=0)
        mu, sd = self._z_stats
        return ((Z - mu) / sd).astype(np.float32)

    def save(self, path):
        """Save the trained encoder + preprocessing to a single ``.pt`` file."""
        import pickle
        import torch
        self._check()
        payload = dict(
            version=2, backbone=self.backbone, embedding_dim=self.embedding_dim,
            num_features=int(self.X.shape[0]), win_len=int(self.history["win_len"]),
            model_kwargs=self.model_kwargs, width=self.settings.get("width", 1.0),
            objective=self.objective, smooth_sigma=self.smooth_sigma, z_stats=self._z_stats,
            reducer=pickle.dumps(self.reducer), state_dict=self.net.state_dict())
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path, device="auto", log=print):
        """Load an encoder saved with ``save``; use ``transform`` on new data."""
        import pickle
        import torch
        p = torch.load(path, map_location="cpu", weights_only=False)
        emb = cls(p["backbone"], embedding_dim=p["embedding_dim"], objective=p["objective"],
                  smooth_sigma=p["smooth_sigma"], device=device, log=log,
                  model_kwargs=p["model_kwargs"])
        mk = dict(BK.width_overrides(p["backbone"], p["width"])); mk.update(p["model_kwargs"])
        net = BK.make_model(p["backbone"], p["num_features"], embedding_dim=p["embedding_dim"],
                            max_len=max(p["win_len"], 8), **mk)
        net.load_state_dict(p["state_dict"])
        emb.net = net.to(_tr.resolve_device(device)).eval()
        emb.history = dict(win_len=p["win_len"])
        emb.reducer = pickle.loads(p["reducer"])
        emb._z_stats = p["z_stats"]
        emb.X = np.zeros((p["num_features"], 1), np.float32)
        emb.Z = np.zeros((1, p["embedding_dim"]), np.float32)   # marks as fitted
        return emb

    # ----- access ----- #
    def embedding(self):
        self._check(); return self.Z

    def state_sequence(self):
        self._check(); return self.states

    def projection(self, method="umap", max_points=None):
        """2-D projection (cached). Returns (P [n, 2], frame_index [n]); a
        uniform strided subset of frames is used for slow methods."""
        self._check()
        m = method.lower().replace("t-sne", "tsne")
        key = (m, max_points)
        if key not in self._proj:
            self._proj[key] = _an.project_subset(self.Z, m, self.seed, max_points)
        return self._proj[key]

    def metrics(self):
        self._check()
        m = _an.metrics(self.Z, self.seed)
        m.update(backbone=self.backbone, n_channels=int(self.X_input.shape[0]),
                 n_model_inputs=int(self.X.shape[0]), n_bins=int(self.X.shape[1]),
                 n_states=int(np.unique(self.states).size),
                 silhouette=_an.cluster_silhouette(self.Z, self.states, self.seed),
                 k_selected_by=("silhouette" if self.n_states is None else "user"),
                 speed=self.speed, objective=self.objective,
                 train_steps=int(self.history.get("steps_done", 0)),
                 best_step=int(self.history.get("best_step", 0)),
                 stopped_early=bool(self.history.get("stopped_early", False)),
                 best_val_loss=float(self.history.get("best_val", float("nan"))),
                 n_params=int(self.history.get("n_params", 0)),
                 likelihood=self.likelihood_used, mask_mode=self.mask_mode,
                 state_model=self.state_model,
                 val_bits_per_spike=float(self.history.get("best_val_bps", float("nan"))),
                 cosmooth_bps=(float(self.cosmoothing["bps"]) if self.cosmoothing
                               else float("nan")),
                 train_seconds=float(self.timings.get("train_s", 0.0)),
                 total_seconds=float(self.timings.get("total_s", 0.0)))
        if self.silhouette_scores is not None and self.n_states_used in self.silhouette_scores:
            m["silhouette_at_k"] = self.silhouette_scores[self.n_states_used]
            m["k_at_upper_bound"] = bool(self.n_states_used == max(self.silhouette_scores))
        m["cluster_structure"] = _an.cluster_structure(m["silhouette"])
        return m

    # ----- decoding ----- #
    def decode(self, behavior, behavior_times=None, *, tasks=None, circular=(),
               model="linear", n_folds=5, gap="auto", n_perm=200, baseline=True,
               psid=False, bayes=True):
        """Cross-validated decoding of behaviour from the embedding.

        behavior        array [T] / [T, k] / dict {name: array} / DataFrame
        behavior_times  time stamps of the behaviour samples; when given, every
                        variable is resampled onto the embedding time base
        circular        names of angular variables (radians)
        baseline        also decode from the raw input (PCA, same dims and
                        smoothing) and report ``delta_vs_raw``
        Returns a DataFrame (also stored in ``self.decoding``)."""
        self._check()
        beh = self._align(behavior, behavior_times, tasks, circular)
        base = (_dec.raw_baseline_features(self.X, self.Z.shape[1], self.smooth_sigma, self.seed)
                if baseline else None)
        t0 = time.perf_counter()
        # Bayesian population decoder on raw counts (Zhang et al. 1998) when
        # counts exist; never pass the z-scored model input (negative values)
        counts = self.counts if (bayes and self.counts is not None) else None
        res = _dec.decode_behavior(self.Z, beh, tasks=tasks, circular=circular, model=model,
                                   n_folds=n_folds, gap=gap, n_perm=n_perm,
                                   smooth_sigma=self.smooth_sigma, baseline=base, seed=self.seed,
                                   counts=counts)
        if psid:
            res.table = pd.concat([res.table, self._psid_rows(beh, res.table, circular, n_folds)],
                                  ignore_index=True)
        res.table.insert(0, "backbone", self.backbone)
        self.timings["decode_s"] = time.perf_counter() - t0
        self.decoding = res
        if self.log:
            for _, r in res.table.iterrows():
                self.log(f"  decode {r['target']:<14s} [{r['features']}] {r['metric']}="
                         f"{r['score']:.3f} (null95={r['null_p95']:.3f}, p={r['p_value']:.4f})")
        return res.table

    def _psid_rows(self, beh, table, circular, n_folds):
        """Cross-fitted PSID decoding (Sani et al. 2021) of every continuous or
        circular target, on the same blocked folds and purge gap as the
        embedding rows. No permutation null (each shift would need a refit)."""
        from .integrations import psid_decode_cv
        rows = []
        for name, y in beh.items():
            sel = table[(table.target == name) & (table.features == "embedding")]
            if not len(sel) or sel.iloc[0]["task"] == "classification":
                continue
            r0 = sel.iloc[0]
            circ = r0["task"] == "circular"
            folds = _dec.blocked_folds(len(y), n_folds, int(r0["gap"]))
            try:
                yhat = psid_decode_cv(self.X, np.asarray(y, float), folds, circular=circ)
            except Exception as exc:  # noqa: BLE001
                if self.log:
                    self.log(f"  PSID skipped for {name}: {exc!r}")
                continue
            yy = np.asarray(y, float)
            ok = np.isfinite(yhat) & np.isfinite(yy)
            if circ:
                score = float(np.cos(yhat[ok] - yy[ok]).mean())
            else:
                score = float(1 - ((yy[ok] - yhat[ok]) ** 2).sum() / (((yy[ok] - yy[ok].mean()) ** 2).sum() + 1e-12))
            rows.append(dict(target=name, features="psid", task=r0["task"], model="psid",
                             metric=r0["metric"], score=score, p_value=np.nan, p_holm=np.nan,
                             chance=r0["chance"], n_frames=int(ok.sum()), gap=r0["gap"],
                             n_features=int(self.X.shape[0])))
        return pd.DataFrame(rows)

    def _align(self, behavior, behavior_times, tasks=None, circular=()):
        beh = _dec._as_dict(behavior)
        if behavior_times is None:
            return beh
        tasks = dict(tasks or {})
        for c in circular or ():
            tasks[c] = "circular"
        out = {}
        for k, v in beh.items():
            task = tasks.get(k) or _dec.infer_task(v)
            out[k] = _dec.align_behavior(v, behavior_times, self.times, task)
        return out

    # ----- figures ----- #
    def plot_embeddings(self, save_path, methods=("pca", "tsne", "umap"), color="time",
                        behavior=None, behavior_name=None, formats=None, circular=False):
        from . import viz as _viz
        self._check()
        fig = self.figure(methods=methods, color=color, behavior=behavior,
                          behavior_name=behavior_name, circular=circular)
        return _viz.save_figure(fig, save_path, formats)

    def figure(self, methods=("pca", "tsne", "umap"), color="time", fig=None,
               behavior=None, behavior_name=None, circular=False):
        """Return a matplotlib Figure of the embedding (for GUI display).
        ``circular=True`` colours an angular behaviour with a cyclic map."""
        from . import viz as _viz
        self._check()
        projections = {m: self.projection(m) for m in methods}
        return _viz.embedding_figure(self.Z, methods=methods, color=color, states=self.states,
                                     seed=self.seed, projections=projections, times=self.times,
                                     behavior=behavior, behavior_name=behavior_name,
                                     circular=circular,
                                     title=f"{BK.label(self.backbone)} embedding", fig=fig)

    def plot_state_sequence(self, save_path, behavior=None, formats=None):
        from . import viz as _viz
        self._check()
        return _viz.state_sequence(self.states, save_path, times=self.times,
                                   behavior=behavior, title=f"{BK.label(self.backbone)} states",
                                   formats=formats)

    def animate(self, save_path, method="umap", **kw):
        from . import viz as _viz
        self._check()
        P, idx = self.projection(method)
        return _viz.trajectory_gif(self.Z, save_path, method=method, seed=self.seed,
                                   projection=(P, idx), states=self.states,
                                   title=f"{BK.label(self.backbone)} trajectory", **kw)

    # ----- one-call pipeline ----- #
    def run(self, outdir, behavior=None, behavior_times=None, animate=True, gif_method="umap",
            prefix=None, methods=("pca", "tsne", "umap"), decode=True, circular=(),
            html=True, formats=None, n_perm=200, dynamics=True, decode_model="linear",
            psid=False, explorer_3d="pca"):
        """Write every output for this recording to ``outdir``:
        figures (png/pdf/svg), dashboard, GIF, interactive HTML, metrics /
        decoding / parameters CSV+JSON and the embedding .npz. ``dynamics``
        adds topology (persistence barcode), latent flow field, recurrence
        plot, Rastermap raster and state-behaviour contingency figures."""
        from . import viz as _viz
        self._check()
        prefix = prefix or self.backbone
        os.makedirs(outdir, exist_ok=True)
        P = lambda name: os.path.join(outdir, f"{prefix}_{name}")       # noqa: E731
        beh = self._align(behavior, behavior_times, None, circular) if behavior is not None else None
        first = next(iter(beh.items())) if beh else (None, None)

        if beh and decode:
            self.decode(beh, circular=circular, n_perm=n_perm, model=decode_model, psid=psid)
            self.decoding.to_csv(P("decoding.csv"))
        self.plot_embeddings(P("embeddings"), methods=methods, formats=formats)
        if first[0] is not None:
            self.plot_embeddings(P(f"embeddings_by_{first[0]}"), methods=methods, color="behavior",
                                 behavior=first[1], behavior_name=first[0], formats=formats,
                                 circular=first[0] in (circular or ()))
        self.plot_state_sequence(P("states"), behavior=first[1], formats=formats)
        _viz.save_figure(_viz.training_figure(self.history, title=f"{BK.label(self.backbone)} training"),
                         P("training"), formats)
        _viz.save_figure(_viz.state_dynamics_figure(self.states, self.frame_dt()),
                         P("state_dynamics"), formats)
        if self.decoding is not None:
            _viz.save_figure(_viz.decoding_figure(self.decoding, self.times), P("decoding"), formats)
        _viz.save_figure(_viz.dashboard(self, behavior=beh, circular=circular), P("dashboard"), formats)
        if dynamics:
            self.write_dynamics(P, beh, circular, gif_method, formats)
        if animate:
            self.animate(P("trajectory.gif"), method=gif_method)
        if html:
            # 3-D view: PCA by default (instant on a CPU); "umap" / "tphate" / False
            _viz.interactive_html(self, P("explorer.html"), behavior=beh, method=gif_method,
                                  circular=circular, view3d=explorer_3d)

        met = self.metrics()
        pd.DataFrame([met]).to_csv(P("metrics.csv"), index=False)
        with open(P("metrics.json"), "w") as fh:
            json.dump(met, fh, indent=2, default=float)
        pd.DataFrame({"step": self.history["step"], "train_loss": self.history["train_loss"],
                      "val_loss": self.history["val_loss"], "lr": self.history["lr"]}
                     ).to_csv(P("training_curve.csv"), index=False)
        if self.silhouette_scores:
            pd.DataFrame({"k": list(self.silhouette_scores), "silhouette": list(
                self.silhouette_scores.values())}).to_csv(P("k_selection.csv"), index=False)
        st = _an.state_statistics(self.states, self.n_states_used)
        pd.DataFrame({"state": np.arange(len(st["occupancy"])), "occupancy": st["occupancy"],
                      "mean_dwell_frames": st["mean_dwell"]}).to_csv(P("state_stats.csv"), index=False)
        pd.DataFrame(st["transitions"]).to_csv(P("transitions.csv"))
        with open(P("params.json"), "w") as fh:
            json.dump(self.params(), fh, indent=2, default=str)
        np.savez_compressed(P("embedding.npz"), Z=self.Z, Z_raw=self.Z_raw, states=self.states,
                            times=self.times)
        if self.log:
            self.log(f"[{self.backbone}] wrote outputs to {outdir}")
        return met

    def write_dynamics(self, P, beh=None, circular=(), method="umap", formats=None):
        """Dynamics / geometry figures + their statistics (``P(name)`` -> path stem).

        Each figure is independent: a failure is logged and skipped so the rest
        of ``run`` still completes."""
        from . import viz as _viz
        from . import viz_dynamics as VD
        stats, circular = {}, tuple(circular or ())
        first = next(iter(beh.items())) if beh else (None, None)

        def keep(prefix, d):
            for k, v in (d or {}).items():
                if np.isscalar(v) and not isinstance(v, str):
                    stats[f"{prefix}_{k}"] = float(v)

        jobs = [("topology", lambda: VD.persistence_figure(self.Z, seed=self.seed)),
                ("flow_field", lambda: VD.flow_field_figure(*self.projection(method), dt=self.frame_dt(),
                                                            states=self.states)),
                ("recurrence", lambda: VD.recurrence_figure(self.Z, times=self.times))]
        if self.X_input is not None and self.X_input.shape[0] >= 8:
            jobs.append(("rastermap", lambda: VD.rastermap_figure(
                self.X_input, times=self.times, states=self.states, behavior=first[1],
                behavior_name=first[0] or "behaviour", circular=first[0] in circular,
                seed=self.seed)))
        for name, v in (beh or {}).items():
            if name not in circular and _dec.infer_task(v) == "classification":
                jobs.append((f"contingency_{name}", lambda v=v, name=name: VD.contingency_figure(
                    self.states, v, label_name=name, seed=self.seed)))
        for name, make in jobs:
            try:
                fig = make()
                _viz.save_figure(fig, P(name), formats)
                st = getattr(fig, "nemba_stats", None)
                if name.startswith("contingency") and isinstance(st, dict):
                    if hasattr(st.get("table"), "to_csv"):
                        st["table"].to_csv(P(f"{name}.csv"))
                    keep(name, {k: v for k, v in st.items() if k != "table"})
                else:
                    keep(name, st if isinstance(st, dict) else {})
                if name == "rastermap" and getattr(fig, "nemba_order", None) is not None:
                    pd.DataFrame({"rank": np.arange(len(fig.nemba_order)),
                                  "channel": np.asarray(fig.nemba_order)}).to_csv(
                        P("neuron_order.csv"), index=False)
            except Exception as exc:  # noqa: BLE001
                if self.log:
                    self.log(f"  [{self.backbone}] {name} figure skipped: {exc!r}")
        if stats:
            pd.DataFrame([stats]).to_csv(P("dynamics_stats.csv"), index=False)
        self.dynamics_stats = stats
        return stats

    def frame_dt(self):
        """Duration of one frame in the units of ``times``."""
        if self.times is not None and len(self.times) > 1:
            return float(np.median(np.diff(self.times)))
        return float(self.bin_size)

    def params(self):
        """Every setting needed to reproduce this run."""
        keys = ("backbone", "bin_size", "embedding_dim", "n_states", "n_states_used", "k_min",
                "k_max", "speed", "max_steps", "epochs", "win_len", "objective",
                "contrastive_weight", "temperature", "contrastive_offset", "mask_ratio",
                "mask_span", "val_frac", "lr", "input_pca", "smooth_sigma", "seed", "model_kwargs",
                "likelihood", "likelihood_used", "mask_mode", "neuron_frac", "state_model")
        out = {k: getattr(self, k) for k in keys}
        out["resolved_settings"] = self.settings
        out["history"] = {k: v for k, v in (self.history or {}).items()
                          if not isinstance(v, list)}
        out["timings_s"] = self.timings
        return out

    def _check(self):
        if self.Z is None:
            raise RuntimeError("call .fit_spikes()/.fit_features()/.fit_fiber()/.fit_matrix() first")
