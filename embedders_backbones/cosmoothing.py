"""Co-smoothing: a label-free score for choosing an embedding.

Idea (Neural Latents Benchmark, Pei et al. 2021; anticipated by the
leave-neuron-out goodness of fit of GPFA, Yu et al. 2009): a good latent
description of the population should predict neurons it has never seen.

Protocol used here for one continuous recording
------------------------------------------------
1. split neurons into held-in (80 %) and held-out (20 %), and time into a
   training block (first 80 %) and a test block (last 20 %);
2. train the encoder on the held-in neurons, training block only;
3. embed the whole recording from the held-in neurons;
4. fit one Poisson GLM per held-out neuron (latents -> counts) on the
   training block;
5. score the test block in **bits per spike**: log-likelihood gain over a
   flat mean-rate model of each held-out neuron, divided by the number of
   spikes and log 2 (positive = better than the mean rate).

No behaviour labels are used, so co-smoothing can choose among backbones
without the selection bias of picking the best decoder on the same labels.
"""

from __future__ import annotations

import math
import time

import numpy as np

# --------------------------------------------------------------------------- #
# user-configurable defaults                                                   #
# --------------------------------------------------------------------------- #
HELDOUT_FRAC = 0.2        # fraction of neurons held out
TEST_FRAC = 0.2           # final fraction of time used for scoring
MIN_SPIKES = 10           # held-out neurons need this many spikes in train and test
GLM_ALPHA = 1e-2          # L2 penalty of the Poisson readout (on standardised latents)


def poisson_bits_per_spike(y, rate, null_rate):
    """Bits per spike of predicted ``rate`` vs a constant ``null_rate`` (per
    neuron) for observed counts ``y`` (arrays [T, n] or [T])."""
    y = np.asarray(y, float)
    rate = np.clip(np.asarray(rate, float), 1e-9, None)
    null_rate = np.clip(np.asarray(null_rate, float), 1e-9, None)
    ll_model = (y * np.log(rate) - rate).sum()
    ll_null = (y * np.log(null_rate) - null_rate).sum() if np.ndim(null_rate) else \
        (y * np.log(null_rate) - null_rate).sum()
    n = y.sum()
    return float((ll_model - ll_null) / (n * math.log(2))) if n > 0 else float("nan")


def _zscore_with(train, full):
    mu = train.mean(1, keepdims=True)
    sd = train.std(1, keepdims=True) + 1e-9
    return ((full - mu) / sd).astype(np.float32)


def cosmooth_score(backbone, counts, *, times=None, heldout_frac=HELDOUT_FRAC,
                   test_frac=TEST_FRAC, seed=0, log=None, glm_alpha=GLM_ALPHA,
                   **embedder_kw):
    """Co-smoothing bits/spike of ``backbone`` on raw counts [C, T].

    Extra keyword arguments go to ``Embedder`` (speed, objective, likelihood,
    mask_mode, smooth_sigma, ...). Returns a dict with ``bps`` (all held-out
    neurons pooled), ``bps_per_neuron``, the held-out indices and timing."""
    from sklearn.linear_model import PoissonRegressor
    from .core import Embedder

    t0 = time.perf_counter()
    Y = np.asarray(counts, dtype=np.float64)
    C, T = Y.shape
    t_split = int(round((1.0 - test_frac) * T))
    if C < 5 or t_split < 200 or T - t_split < 50:
        raise ValueError(f"co-smoothing needs >= 5 channels and enough frames (got {C} x {T})")

    # held-out neurons: active enough in both blocks
    ok = (Y[:, :t_split].sum(1) >= MIN_SPIKES) & (Y[:, t_split:].sum(1) >= MIN_SPIKES)
    cand = np.flatnonzero(ok)
    rng = np.random.default_rng(seed)
    n_out = int(max(1, round(heldout_frac * C)))
    if cand.size < n_out + 2:
        raise ValueError("not enough active neurons for a held-out set")
    held_out = np.sort(rng.choice(cand, n_out, replace=False))
    held_in = np.setdiff1d(np.arange(C), held_out)

    # model input: sqrt counts of held-in neurons, z-scored with training-block statistics
    Xin = _zscore_with(np.sqrt(Y[held_in, :t_split]), np.sqrt(Y[held_in]))
    kw = dict(embedder_kw)
    kw.setdefault("log", None)
    kw.setdefault("seed", seed)
    kw.setdefault("n_states", 2)              # states are irrelevant here: skip K search
    if kw.get("likelihood") == "poisson" or kw.get("likelihood") == "auto":
        emb = Embedder(backbone, **kw).fit_matrix(Xin[:, :t_split], times=None,
                                                  counts=Y[held_in, :t_split])
    else:
        emb = Embedder(backbone, **kw).fit_matrix(Xin[:, :t_split])
    Z = emb.transform(Xin)                    # [T, E], all frames, encoder never saw the test block

    zs_mu, zs_sd = Z[:t_split].mean(0), Z[:t_split].std(0) + 1e-8
    Zs = (Z - zs_mu) / zs_sd
    rates = np.empty((T - t_split, n_out))
    for j, n in enumerate(held_out):
        glm = PoissonRegressor(alpha=glm_alpha, max_iter=300)
        glm.fit(Zs[:t_split], Y[n, :t_split])
        rates[:, j] = glm.predict(Zs[t_split:])
    y_test = Y[held_out, t_split:].T
    null = y_test.mean(0, keepdims=True)                     # NLB convention: test-block mean
    per = np.array([poisson_bits_per_spike(y_test[:, j], rates[:, j], null[0, j])
                    for j in range(n_out)])
    bps = poisson_bits_per_spike(y_test, rates, np.broadcast_to(null, y_test.shape))
    out = dict(backbone=backbone, bps=bps, bps_per_neuron=per, heldout_idx=held_out,
               n_heldout=int(n_out), n_test_frames=int(T - t_split),
               seconds=time.perf_counter() - t0)
    if log:
        log(f"  [{backbone}] co-smoothing: {bps:.3f} bits/spike on {n_out} held-out neurons "
            f"({out['seconds']:.1f}s)")
    return out
