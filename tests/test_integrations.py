"""Tests for the external-method integrations: CEBRA (reference package),
PSID (cross-fitted decoding) and the native GPFA backbone.

Every test is small (the whole file runs in well under 90 s on a laptop CPU)
and skips cleanly when an optional package is missing. Fit times are printed
(run with ``pytest -s`` to see them)."""

import time

import numpy as np
import pytest
import scipy.linalg as sla
import torch

from embedders_backbones import backbones as BK
from embedders_backbones import training as TR
from embedders_backbones.decoding import blocked_folds


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _plug_in(monkeypatch, net):
    """Make ``training.train`` build ``net`` (no registry edit needed)."""
    monkeypatch.setattr(BK, "make_model", lambda name, num_features, **kw: net)


def _ring_data(T=1500, C=16, seed=0):
    """Neurons tuned to a slowly drifting angle (ring manifold), z-scored, [C, T]."""
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.normal(0.0, 0.06, T))
    pref = np.linspace(0, 2 * np.pi, C, endpoint=False)
    X = np.exp(2.0 * np.cos(theta[None, :] - pref[:, None])) + 0.3 * rng.standard_normal((C, T))
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    return X.astype(np.float32), theta


def _aligned_r2(Z, X):
    """R^2 of each true latent X [p, T] regressed (with intercept) on Z [T, k]."""
    A = np.c_[Z, np.ones(len(Z))]
    W, *_ = np.linalg.lstsq(A, X.T, rcond=None)
    res = X.T - A @ W
    return 1.0 - res.var(0) / X.T.var(0)


# --------------------------------------------------------------------------- #
# CEBRA                                                                        #
# --------------------------------------------------------------------------- #

def test_cebra_backbone_through_train_contract(monkeypatch):
    """Reference CEBRA-time trains through train()'s trainable=False path,
    reproduces cebra's own transform, embeds the whole recording and survives
    a state_dict round trip."""
    pytest.importorskip("cebra")
    from embedders_backbones.integrations import (CEBRABackbone, CEBRABackboneConfig,
                                                  cebra_consistency)
    X, _ = _ring_data()
    cfg = CEBRABackboneConfig.from_dict(dict(num_features=X.shape[0], embedding_dim=3,
                                             max_len=256, max_iterations=100, batch_size=256,
                                             unknown_key="ignored"))
    net = CEBRABackbone(cfg)
    assert net.trainable is False
    _plug_in(monkeypatch, net)
    t0 = time.perf_counter()
    out, hist = TR.train("cebra_plugin", X, embedding_dim=3, speed="balanced", log=None)
    print(f"\n  CEBRA fit (100 it, {X.shape[0]} x {X.shape[1]}): {time.perf_counter() - t0:.1f}s")
    assert out is net and bool(net.fitted) and hist["seconds"] > 0
    assert hist["steps_done"] == 0                        # no SGD in NEMBA's loop

    Z = net.transform_numpy(X)
    assert Z.shape == (X.shape[1], 3) and np.isfinite(Z).all()
    # identical to the reference estimator's transform
    Zref = net.estimator_.transform(X.T)
    np.testing.assert_allclose(Z, Zref, atol=1e-5)
    # whole-sequence path of embed_sequence and the torch forward fallback
    Ze = TR.embed_sequence(net, X)
    assert Ze.shape == Z.shape and np.isfinite(Ze).all()
    emb = net(torch.from_numpy(X[None, :, :200])).embeddings
    assert tuple(emb.shape) == (1, 3, 200)
    # persistence through state_dict only
    net2 = CEBRABackbone(CEBRABackboneConfig.from_dict(cfg.to_dict()))
    net2.load_state_dict(net.state_dict())
    np.testing.assert_allclose(net2.transform_numpy(X), Z, atol=1e-6)
    # consistency metric (cebra's implementation) returns a tidy table
    rng = np.random.default_rng(0)
    df = cebra_consistency([Z, Z @ rng.standard_normal((3, 3)) + 0.01 * rng.standard_normal(Z.shape)],
                           names=["a", "b"])
    assert list(df.columns) == ["source", "target", "score", "between", "method"]
    assert len(df) == 2 and set(df.source) == {"a", "b"} and (df.score > 0.95).all()


# --------------------------------------------------------------------------- #
# PSID                                                                         #
# --------------------------------------------------------------------------- #

def _rot(theta, r):
    """2-D damped rotation (eigenvalues r exp(+-i theta))."""
    return r * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _lds(T=3000, C=20, seed=0):
    """Linear dynamical system: x1 (2 states) drives behaviour and neurons,
    x2 (4 states, larger variance) drives the neurons only."""
    rng = np.random.default_rng(seed)
    A = np.zeros((6, 6))
    A[:2, :2] = _rot(0.08, 0.98)
    A[2:4, 2:4] = _rot(0.25, 0.95)
    A[4:, 4:] = _rot(0.03, 0.99)
    x = np.zeros((T, 6))
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.standard_normal(6) * np.r_[1, 1, 2, 2, 2, 2]
    Y = x @ rng.standard_normal((C, 6)).T + 3.0 * rng.standard_normal((T, C))
    z = x[:, :2] @ rng.standard_normal(2) + 0.3 * rng.standard_normal(T)
    return Y.T, z, x


def test_psid_cross_fitted_decoding_and_latents():
    """Cross-fitted PSID recovers the behaviour-relevant subspace on held-out
    blocks; latents have shape [T, nx]; circular targets go through sin/cos."""
    pytest.importorskip("PSID")
    from embedders_backbones.integrations import fit_psid, psid_decode_cv, psid_latents
    X, z, x = _lds()
    T = X.shape[1]
    folds = blocked_folds(T, 5, gap=20)
    t0 = time.perf_counter()
    y_hat, info = psid_decode_cv(X, z, folds, nx=6, n1=2, i=10, return_details=True)
    print(f"\n  PSID 5-fold cross-fitted decoding ({X.shape[0]} x {T}): "
          f"{time.perf_counter() - t0:.1f}s")
    ok = np.isfinite(y_hat)
    assert ok.mean() > 0.99                               # every test block predicted
    r2 = 1 - np.sum((z[ok] - y_hat[ok]) ** 2) / np.sum((z[ok] - z[ok].mean()) ** 2)
    assert r2 > 0.6, r2
    assert len(info["fold_scores"]) == 5 and min(info["fold_scores"]) > 0.5

    model = fit_psid(X, z, nx=6, n1=2, i=10)
    L = psid_latents(model, X)
    assert L.shape == (T, 6) and np.isfinite(L).all()

    ang = np.arctan2(x[:, 1], x[:, 0])                    # angle of the behaviour-related state
    a_hat = psid_decode_cv(X, ang, folds, nx=6, n1=2, circular=True)
    ok = np.isfinite(a_hat)
    assert np.mean(np.cos(a_hat[ok] - ang[ok])) > 0.6


# --------------------------------------------------------------------------- #
# GPFA                                                                         #
# --------------------------------------------------------------------------- #

def _sample_gpfa(q, T, taus, seed=0, noise=0.5):
    """Continuous recording drawn from the GPFA generative model."""
    from embedders_backbones.gpfa import se_kernel
    rng = np.random.default_rng(seed)
    X = np.stack([sla.cholesky(se_kernel(T, t), lower=True) @ rng.standard_normal(T)
                  for t in taus])
    C = rng.standard_normal((q, len(taus)))
    d = rng.standard_normal(q)
    R = noise * rng.uniform(0.5, 1.5, q)
    Y = C @ X + d[:, None] + np.sqrt(R)[:, None] * rng.standard_normal((q, T))
    return Y, X


def test_gpfa_recovers_latents_timescales_and_em_is_monotone():
    """On data from its own generative model GPFA recovers the latents (up to
    a linear map), the timescales (within 30 %), and EM never decreases the
    log-likelihood."""
    from embedders_backbones.gpfa import GPFA
    taus = [4.0, 10.0, 25.0]
    Y, Xtrue = _sample_gpfa(30, 3000, taus, seed=1)
    m = GPFA(3, seg_len=100, max_iter=60, tol=1e-4).fit(Y)
    print(f"\n  GPFA fit (30 x 3000, 3 latents): {m.fit_seconds_:.1f}s, "
          f"{m.n_iter_} EM iterations, tau={np.round(np.sort(m.timescales_), 2)}")
    Z = m.transform(Y)
    assert Z.shape == (3000, 3) and np.isfinite(Z).all()
    assert (_aligned_r2(Z, Xtrue) > 0.9).all()
    rel = np.abs(np.sort(m.timescales_) - np.sort(taus)) / np.sort(taus)
    assert (rel < 0.3).all(), rel
    ll = np.asarray(m.log_likelihoods_)
    assert np.all(np.diff(ll) >= -1e-6 * np.abs(ll[1:])), np.diff(ll).min()
    assert ll[-1] > ll[0]


def test_gpfa_backbone_contract_and_persistence(monkeypatch):
    """GPFABackbone trains through train()'s trainable=False path, exposes
    timescales_ (frames), embeds without seams and reloads from state_dict."""
    from embedders_backbones.gpfa import GPFABackbone, GPFABackboneConfig
    Y, Xtrue = _sample_gpfa(20, 1200, [3.0, 12.0], seed=2)
    Y = ((Y - Y.mean(1, keepdims=True)) / Y.std(1, keepdims=True)).astype(np.float32)
    cfg = GPFABackboneConfig.from_dict(dict(num_features=20, embedding_dim=2, max_len=256,
                                            seg_len=60, max_iter=40, unknown_key=1))
    net = GPFABackbone(cfg)
    assert net.trainable is False
    _plug_in(monkeypatch, net)
    out, hist = TR.train("gpfa_plugin", Y, embedding_dim=2, speed="balanced", log=None)
    assert out is net and bool(net.fitted)
    assert net.timescales_.shape == (2,) and np.all(net.timescales_ > 0)
    Z = TR.embed_sequence(net, Y)
    assert Z.shape == (1200, 2) and np.isfinite(Z).all()
    assert (_aligned_r2(Z, Xtrue) > 0.8).all()
    # no seams: frame-to-frame jumps at window borders look like elsewhere
    jumps = np.linalg.norm(np.diff(Z, axis=0), axis=1)
    assert jumps.max() < 6 * np.median(jumps) + 1e-6
    # torch fallback and persistence through state_dict only
    emb = net(torch.from_numpy(Y[None, :, :100])).embeddings
    assert tuple(emb.shape) == (1, 2, 100)
    net2 = GPFABackbone(GPFABackboneConfig.from_dict(cfg.to_dict()))
    net2.load_state_dict(net.state_dict())
    np.testing.assert_allclose(net2.transform_numpy(Y), net.transform_numpy(Y), atol=1e-5)
    np.testing.assert_allclose(net2.timescales_, net.timescales_)


def test_gpfa_timing_60x6000x8():
    """Fit cost on the reference size (60 neurons x 6000 frames x 8 latents).
    The default budget is 100 EM iterations; here 15 keep the suite fast and
    the per-iteration cost is printed for extrapolation."""
    from embedders_backbones.gpfa import GPFA
    rng = np.random.default_rng(3)
    T, q, p = 6000, 60, 8
    lat = np.stack([np.convolve(rng.standard_normal(T + 200), np.hanning(8 + 4 * i), "same")[100:-100]
                    for i in range(p)])
    Y = rng.standard_normal((q, p)) @ lat + rng.standard_normal((q, T))
    m = GPFA(p, seg_len=100, max_iter=15).fit(Y)
    per_it = m.fit_seconds_ / max(m.n_iter_, 1)
    print(f"\n  GPFA 60 x 6000 x 8: {m.fit_seconds_:.1f}s for {m.n_iter_} EM iterations "
          f"({per_it:.2f}s / iteration, ~{100 * per_it:.0f}s for the default 100)")
    assert np.isfinite(m.transform(Y)).all()
    assert m.fit_seconds_ < 60
