"""Reference external methods wrapped for NEMBA: CEBRA and PSID.

CEBRA (Schneider, Lee & Mathis, Nature 2023, doi 10.1038/s41586-023-06031-6)
-----------------------------------------------------------------------------
``CEBRABackbone`` runs the *reference* ``cebra`` package (``cebra.CEBRA``) in
its self-supervised time-contrastive mode (``conditional="time"``: positives
are frames ``time_offsets`` bins apart) and exposes it with the NEMBA backbone
contract (``trainable = False``, ``fit_numpy``, ``transform_numpy``,
``encode``, ``forward``). The trained convolutional encoder is copied into a
registered sub-module, so ``state_dict`` save / load restores a usable model
without pickling the scikit-learn estimator.

The behaviour-conditioned CEBRA (``conditional="time_delta"`` with labels) is
deliberately NOT a backbone: it uses the behaviour to shape the embedding, so
decoding that behaviour from it is circular unless the encoder is cross-fitted
(trained on training folds only, embedded and decoded on held-out folds).
Running it through the generic "embed then decode" pipeline would leak labels.

Also ``cebra_consistency(embeddings)``: CEBRA's consistency metric (R^2 of a
linear map between pairs of embeddings) as a tidy DataFrame.

PSID (Sani, Pesaran & Shanechi, Nat Neurosci 2021, doi 10.1038/s41593-020-00733-0)
------------------------------------------------------------------------------------
Preferential subspace identification: a linear state-space model whose first
``n1`` latent states are dedicated to behaviour. Thin, leak-free wrappers
around the ``PSID`` package: ``fit_psid``, ``psid_latents`` and the
cross-fitted ``psid_decode_cv`` (PSID is refitted on the training blocks of
every fold; the test block is never seen during fitting).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._embtcn import NeuralEmbTCNOutput


# --------------------------------------------------------------------------- #
# optional-dependency guards                                                   #
# --------------------------------------------------------------------------- #

def _require_cebra():
    """Import ``cebra`` or raise an ImportError that says how to install it."""
    try:
        import cebra  # noqa: F401
        import cebra.models  # noqa: F401
    except Exception as exc:  # noqa: BLE001  (cebra can fail with non-ImportErrors too)
        raise ImportError(
            "The 'cebra' backbone needs the CEBRA package (Schneider et al. 2023). "
            "Install it with `pip install cebra` (CPU is fine) and check that "
            f"`import cebra` works in this environment. Original error: {exc!r}") from exc
    return cebra


def _require_psid():
    """Import the ``PSID`` package or raise an ImportError with install hint."""
    try:
        import PSID  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "PSID decoding needs the PSID package (Sani et al. 2021). Install it with "
            f"`pip install PSID`. Original error: {exc!r}") from exc
    return PSID


# --------------------------------------------------------------------------- #
# CEBRA backbone                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class CEBRABackboneConfig:
    """Configuration of the CEBRA-time backbone (reference ``cebra`` package).

    ``max_iterations=2000`` is a CPU budget (CEBRA's own default is 10000
    steps, meant for a GPU). Measured on 60 channels x 6000 frames, batch 512,
    offset10 model, on a 16-thread laptop CPU that was fully loaded by other
    jobs: ~170 ms per step, so ~6 min for 2000 steps (less on an idle
    machine). Lower ``max_iterations`` / ``batch_size`` for quick looks. ``causal`` is
    accepted for interface parity only: the offset-N models have a symmetric
    receptive field (acausal), so ``causal=True`` is ignored with a warning.
    ``seed`` seeds torch and numpy before training (CEBRA itself has no seed).
    """
    num_features: int = 45
    embedding_dim: int = 8
    max_len: int = 1024              # unused, accepted for interface parity
    causal: bool = False
    model_architecture: str = "offset10-model"
    time_offsets: int = 10
    temperature: float = 1.0
    learning_rate: float = 3e-4
    batch_size: int = 512
    max_iterations: int = 2000
    distance: str = "cosine"
    conditional: str = "time"
    device: str = "cpu"
    seed: int = 0
    num_hidden_units: int = 32       # CEBRA default width
    verbose: bool = False

    @classmethod
    def from_dict(cls, payload: dict) -> "CEBRABackboneConfig":
        """Build from a dict, silently ignoring keys that are not fields."""
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in payload.items() if k in fields})

    def to_dict(self) -> dict:
        """Plain-dict view of the config."""
        return asdict(self)


class CEBRABackbone(nn.Module):
    """CEBRA-time (self-supervised, time-contrastive) as a NEMBA backbone.

    ``fit_numpy(X [C, T])`` trains the reference ``cebra.CEBRA`` estimator on
    X.T and copies its encoder into ``self.model``; ``transform_numpy`` then
    reproduces ``cebra.CEBRA.transform`` exactly (replicate padding of
    ``offset.left`` / ``offset.right - 1`` frames, one convolutional pass).
    Only ``conditional="time"`` is supported here (see the module docstring
    for why behaviour-conditioned CEBRA must be cross-fitted instead)."""

    trainable = False

    def __init__(self, cfg: CEBRABackboneConfig) -> None:
        """Build an untrained encoder with the same architecture CEBRA will train."""
        super().__init__()
        cebra = _require_cebra()
        if cfg.causal:
            warnings.warn("CEBRA offset models are acausal (symmetric receptive field); "
                          "causal=True is ignored", stacklevel=2)
            cfg.causal = False
        if cfg.conditional != "time":
            raise ValueError("CEBRABackbone only supports conditional='time' (self-supervised). "
                             "Behaviour-conditioned CEBRA must be cross-fitted before decoding; "
                             "use cebra.CEBRA directly inside a CV loop.")
        self.cfg = cfg
        self.k = int(cfg.embedding_dim)
        # identical construction to cebra.CEBRA._prepare_model (single session)
        self.model = cebra.models.init(cfg.model_architecture, num_neurons=int(cfg.num_features),
                                       num_units=int(cfg.num_hidden_units), num_output=self.k)
        off = self.model.get_offset()
        self.pad_left, self.pad_right = int(off.left), int(off.right) - 1
        for p in self.model.parameters():
            p.requires_grad_(False)                   # never trained by NEMBA's SGD loop
        self.register_buffer("fitted", torch.tensor(False))
        self.loss_history_: list = []
        self.estimator_ = None

    def fit_numpy(self, X: np.ndarray) -> "CEBRABackbone":
        """Train reference CEBRA-time on X [C, T] (frames are samples)."""
        cebra = _require_cebra()
        c = self.cfg
        X = np.ascontiguousarray(np.asarray(X, np.float32).T)                  # [T, C]
        torch.manual_seed(c.seed)
        np.random.seed(c.seed)
        est = cebra.CEBRA(model_architecture=c.model_architecture, device=c.device,
                          criterion="infonce", distance=c.distance, conditional=c.conditional,
                          temperature=c.temperature, time_offsets=c.time_offsets,
                          max_iterations=int(c.max_iterations), batch_size=int(c.batch_size),
                          learning_rate=c.learning_rate, output_dimension=self.k,
                          num_hidden_units=int(c.num_hidden_units), verbose=c.verbose)
        est.fit(X)
        # copy the trained encoder into the registered sub-module (persistable)
        state = {k: v.detach().cpu() for k, v in est.model_.state_dict().items()}
        dev = next(self.model.parameters()).device
        self.model.load_state_dict(state)
        self.model.to(dev)
        with torch.no_grad():
            self.fitted.fill_(True)
        log = getattr(est.solver_, "log", {}) or {}
        self.loss_history_ = [float(v) for v in log.get("total", [])]
        self.estimator_ = est
        return self

    def _check_fitted(self) -> None:
        """Raise if the encoder has not been trained (or loaded) yet."""
        if not bool(self.fitted):
            raise RuntimeError("CEBRABackbone is not fitted; call fit_numpy(X) first")

    @torch.no_grad()
    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, C, L] -> z [B, E, L] (each window padded like CEBRA's transform)."""
        self.model.eval()
        dev = next(self.model.parameters()).device
        xp = F.pad(x.to(device=dev, dtype=torch.float32), (self.pad_left, self.pad_right),
                   mode="replicate")
        return self.model(xp).to(x.device)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Embedding Z [T, E] of the whole recording X [C, T] in one pass."""
        self._check_fitted()
        x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))[None]   # [1, C, T]
        return self.encode(x)[0].T.cpu().numpy().astype(np.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        """Embeddings only (CEBRA has no reconstruction head)."""
        return NeuralEmbTCNOutput(embeddings=self.encode(x, mask))


def cebra_consistency(embeddings: list, labels: list | None = None, between: str = "runs",
                      dataset_ids: list | None = None, names: list | None = None,
                      num_discretization_bins: int = 100) -> pd.DataFrame:
    """CEBRA consistency between embeddings, one row per ordered pair.

    The score of (i -> j) is the R^2 of a linear regression predicting
    embedding j from embedding i (Schneider et al. 2023, Fig. 1). With
    ``between="runs"`` the embeddings must come from the same recording (same
    frames); ``between="datasets"`` aligns different recordings on
    ``labels`` (one array per embedding) as CEBRA does.

    Uses ``cebra.integrations.sklearn.metrics.consistency_score`` (also
    reachable as ``cebra.sklearn.metrics``) when cebra imports; otherwise a
    numerically identical scikit-learn fallback for ``between="runs"``.
    Note that the score is in-sample (fit and score on the same frames), as
    in CEBRA; it measures linear identifiability, not generalisation.

    Returns a DataFrame with columns: source, target, score, between, method.
    """
    embs = [np.asarray(e.detach().cpu() if torch.is_tensor(e) else e, dtype=np.float64)
            for e in embeddings]
    if len(embs) < 2:
        raise ValueError("need at least two embeddings")
    names = list(names) if names is not None else [str(i) for i in range(len(embs))]
    if len(names) != len(embs):
        raise ValueError("names must have one entry per embedding")
    method = "cebra"
    try:
        cebra = _require_cebra()
        try:
            from cebra.integrations.sklearn.metrics import consistency_score
        except ImportError:                                   # older / newer layouts
            consistency_score = cebra.sklearn.metrics.consistency_score
        if labels is not None:
            labels = [np.asarray(l) for l in labels]
        scores, pairs, _ = consistency_score(
            embeddings=embs, between=between, labels=labels, dataset_ids=dataset_ids,
            num_discretization_bins=num_discretization_bins)
        scores = np.asarray(scores, float).ravel()
        pairs = np.asarray(pairs).reshape(-1, 2)

        def _label(v):
            """Pair entry -> display name (integer positions map to ``names``)."""
            v = _py(v)
            if dataset_ids is None and isinstance(v, (int, np.integer)) and 0 <= v < len(names):
                return names[int(v)]
            return str(v)

        rows = [dict(source=_label(a), target=_label(b), score=float(s))
                for (a, b), s in zip(pairs, scores)]
    except ImportError:
        if between != "runs":
            raise
        from sklearn.linear_model import LinearRegression
        method = "sklearn_fallback"
        rows = []
        for i, a in enumerate(embs):
            for j, b in enumerate(embs):
                if i != j:
                    s = LinearRegression().fit(a, b).score(a, b)
                    rows.append(dict(source=names[i], target=names[j], score=float(s)))
    df = pd.DataFrame(rows)
    df["between"] = between
    df["method"] = method
    return df


def _py(v):
    """numpy scalar -> plain Python value (for dict lookups)."""
    return v.item() if hasattr(v, "item") else v


# --------------------------------------------------------------------------- #
# PSID                                                                         #
# --------------------------------------------------------------------------- #

def _as_time_first(Y, T: int, name: str) -> np.ndarray:
    """Behaviour as float [T, k] (accepts [T] or [T, k])."""
    Y = np.asarray(Y, np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2 or Y.shape[0] != T:
        raise ValueError(f"{name} must be [T] or [T, k] with T={T}, got shape {Y.shape}")
    return Y


def psid_default_dims(n_channels: int, n_behaviour: int, i: int = 10) -> tuple[int, int]:
    """Default (nx, n1) when the user does not choose them.

    n1 = min(2 * nz, 8): two states per behaviour column allow damped
    oscillations / second-order dynamics; nx = n1 + min(8, n_channels): a few
    extra states absorb behaviour-unrelated neural dynamics. Both are capped
    by PSID's identifiability limits (n1 <= nz * i, nx <= ny * i). These are
    heuristics: Sani et al. pick nx and n1 by inner cross-validation, which
    you should do when the dimensions matter for the conclusion."""
    n1 = int(min(2 * n_behaviour, 8, n_behaviour * i))
    nx = int(min(n1 + min(8, n_channels), n_channels * i))
    return nx, min(n1, nx)


def fit_psid(X, Y, nx: int, n1: int, i: int = 10, **kwargs):
    """Fit PSID on neural data X [C, T] and behaviour Y [T, k] (or [T]).

    Returns the ``PSID.LSSM`` model (time-first convention; PSID's own names
    are Y = neural, Z = behaviour). Extra keyword arguments go to
    ``PSID.PSID``. ``X`` may also be a list of [C, T_s] segments with ``Y`` a
    matching list of [T_s, k] arrays (discontinuous training data)."""
    PSID = _require_psid()
    if isinstance(X, (list, tuple)):
        Yn = [np.asarray(x, np.float64).T for x in X]
        Zb = [_as_time_first(y, len(yn), "Y") for y, yn in zip(Y, Yn)]
        C, nz = Yn[0].shape[1], Zb[0].shape[1]
    else:
        Yn = np.asarray(X, np.float64).T                                   # [T, C]
        Zb = _as_time_first(Y, Yn.shape[0], "Y")
        C, nz = Yn.shape[1], Zb.shape[1]
    nx, n1, i = int(nx), int(n1), int(i)
    if n1 > nx:
        raise ValueError(f"n1={n1} must not exceed nx={nx}")
    if nx > C * i or n1 > nz * i:
        raise ValueError(f"PSID needs nx <= C*i ({C * i}) and n1 <= k*i ({nz * i}); "
                         f"got nx={nx}, n1={n1}, i={i}. Increase i or reduce the dimensions.")
    return PSID.PSID(Yn, Zb, nx=nx, n1=n1, i=i, time_first=True, **kwargs)


def psid_latents(model, X, filtered: bool = True) -> np.ndarray:
    """Kalman-filter latent states of X [C, T] -> [T, nx].

    ``filtered=True`` returns x(t | y(1..t)) (uses the current frame);
    ``False`` the one-step prediction x(t | y(1..t-1)), which is what PSID's
    own ``predict`` and its behaviour read-out ``Cz`` use. Both are causal."""
    Yn = np.asarray(X, np.float64).T
    Xp, _, Xf = model.kalman(Yn)[:3]
    return np.asarray(Xf if filtered else Xp)


def _contiguous_runs(idx: np.ndarray) -> list[np.ndarray]:
    """Split sorted indices into runs of consecutive frames."""
    idx = np.asarray(idx)
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) != 1) + 1
    return np.split(idx, cut)


def psid_decode_cv(X, y, folds, nx: int | None = None, n1: int | None = None, i: int = 10,
                   circular: bool = False, return_details: bool = False, **psid_kwargs):
    """Cross-fitted PSID decoding of behaviour y from neural data X [C, T].

    For every (train, test) pair in ``folds`` (e.g. from
    ``decoding.blocked_folds``, which already removes a purge gap around the
    test block), PSID is fitted on the training frames only, passed as their
    contiguous runs (so no artificial continuity is created across the
    removed test block), and behaviour is predicted on the test block by
    running the model's Kalman filter over the test block alone, starting
    from the zero state (the first few frames are a short filter burn-in).
    Prediction uses PSID's own read-out: one-step-ahead states x(t|t-1) and
    the KF-refitted ``Cz`` (the same states Cz was fitted on).

    y        : [T] or [T, k] behaviour. With ``circular=True`` (angles in
               radians, 1-D) sin and cos are decoded and recombined by atan2.
    nx, n1   : state dimensions; ``None`` uses ``psid_default_dims``.
    i        : PSID horizon (block rows); training runs shorter than 2i + 2
               frames are dropped (PSID cannot use them).

    Returns out-of-fold predictions with the shape of y (NaN on frames that
    are in no test block). With ``return_details=True`` returns
    ``(y_hat, info)`` where info has nx, n1, i and per-fold scores (R^2, or
    mean cos(error) for circular targets)."""
    _require_psid()
    Xn = np.asarray(X, np.float64)
    C, T = Xn.shape
    y = np.asarray(y, np.float64)
    if circular:
        if y.ndim != 1:
            raise ValueError("circular=True expects a 1-D angle series (radians)")
        Yb = np.c_[np.sin(y), np.cos(y)]
    else:
        Yb = _as_time_first(y, T, "y")
    if Yb.shape[0] != T:
        raise ValueError(f"y has {Yb.shape[0]} frames but X has {T}")
    k = Yb.shape[1]
    dnx, dn1 = psid_default_dims(C, k, i)
    nx = dnx if nx is None else int(nx)
    n1 = min(dn1 if n1 is None else int(n1), nx)
    min_run = 2 * int(i) + 2

    pred = np.full((T, k), np.nan)
    fold_scores = []
    for tr, te in folds:
        runs = [r for r in _contiguous_runs(np.sort(tr)) if len(r) >= min_run]
        if not runs or len(te) < 2:
            continue
        model = fit_psid([Xn[:, r] for r in runs], [Yb[r] for r in runs], nx=nx, n1=n1, i=i,
                         **psid_kwargs)
        te = np.sort(te)
        for r in _contiguous_runs(te):                         # filter each test block alone
            zp = model.predict(Xn[:, r].T)[0]                  # (Zp, Yp, Xp)
            pred[r] = np.asarray(zp).reshape(len(r), k)
        fold_scores.append(_score_block(Yb[te], pred[te], circular))

    if circular:
        y_hat = np.arctan2(pred[:, 0], pred[:, 1])
    else:
        y_hat = pred[:, 0] if y.ndim == 1 else pred
    if not return_details:
        return y_hat
    return y_hat, dict(nx=nx, n1=n1, i=int(i), fold_scores=fold_scores, circular=circular)


def _score_block(Y: np.ndarray, P: np.ndarray, circular: bool) -> float:
    """Fold score: mean R^2 over behaviour columns, or mean cos(angle error)."""
    if circular:
        err = np.arctan2(P[:, 0], P[:, 1]) - np.arctan2(Y[:, 0], Y[:, 1])
        return float(np.mean(np.cos(err)))
    ss_res = ((Y - P) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1.0 - ss_res / ss_tot))
