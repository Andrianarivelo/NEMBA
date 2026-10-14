"""Tests for the dynamics / topology figures (viz_dynamics), cross-session
alignment (align) and time-aware projections (manifold). Fast: < 60 s."""

import sys

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d
from scipy.stats import special_ortho_group
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

from embedders_backbones import align as AL
from embedders_backbones import manifold as MF
from embedders_backbones import viz
from embedders_backbones import viz_dynamics as VD
from embedders_backbones.synthetic import make_dataset

HAS_RIPSER = MF._HAS_RIPSER


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def small_recording():
    """Short synthetic recording: counts [N, T], a smoothed PCA latent,
    discovered-like states and the ground-truth behaviour."""
    ds = make_dataset(n_neurons=30, duration=150.0, dt=0.1, seed=3)
    X = ds["counts"].astype(float)
    Z = PCA(8, random_state=0).fit_transform(gaussian_filter1d(X, 3, axis=1).T)
    from sklearn.cluster import KMeans
    states = KMeans(4, n_init=4, random_state=0).fit_predict(Z)
    return dict(X=X, Z=Z, states=states, times=ds["times"], beh=ds["behavior"])


def _ring_session(n_frames, n_neurons, seed, noise=0.5, n_pcs=6):
    """Ring latent (shared across sessions) read out by a session-specific
    random linear map plus noise, then reduced to its own PCs."""
    g = np.random.default_rng(seed)
    th = np.angle(np.exp(1j * np.cumsum(0.05 * g.standard_normal(n_frames) + 0.02)))
    W = g.standard_normal((2, n_neurons))
    X = np.c_[np.cos(th), np.sin(th)] @ W + noise * g.standard_normal((n_frames, n_neurons))
    return PCA(n_pcs, random_state=0).fit_transform(X), th


# --------------------------------------------------------------------------- #
# topology                                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not HAS_RIPSER, reason="ripser not installed")
def test_persistence_separates_circle_from_blob():
    """A noisy circle has one long normalised H1 bar, a Gaussian blob none."""
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 600)
    circle = np.c_[np.cos(th), np.sin(th)] + 0.05 * rng.standard_normal((600, 2))
    blob = rng.standard_normal((600, 3))
    assert MF.persistence(circle, max_points=600)["h1_norm"] > 0.5
    assert MF.persistence(blob, max_points=600)["h1_norm"] < 0.2


@pytest.mark.skipif(not HAS_RIPSER, reason="ripser not installed")
def test_persistence_matches_analysis_circularity():
    """Same subset and normalisation as analysis.circularity."""
    from embedders_backbones.analysis import circularity
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((900, 4))
    assert MF.persistence(Z, max_points=700, seed=0)["h1_norm"] == pytest.approx(
        circularity(Z, seed=0))


def test_persistence_figure_without_ripser(monkeypatch, tmp_path):
    """Graceful message panel when ripser is missing."""
    monkeypatch.setattr(MF, "_HAS_RIPSER", False)
    fig = VD.persistence_figure(np.random.default_rng(0).standard_normal((100, 3)))
    assert isinstance(fig, Figure)
    viz.save_figure(fig, tmp_path / "noripser", formats=("png",))


# --------------------------------------------------------------------------- #
# consistency and alignment                                                    #
# --------------------------------------------------------------------------- #

def test_linear_consistency_rotation_vs_noise():
    """A rotated + scaled copy is perfectly consistent; independent noise is not."""
    rng = np.random.default_rng(0)
    Za = gaussian_filter1d(rng.standard_normal((2000, 6)), 5, axis=0)
    Zb = 3.0 * Za @ special_ortho_group.rvs(6, random_state=2) + 1.0
    Zc = gaussian_filter1d(rng.standard_normal((2000, 6)), 5, axis=0)
    assert AL.linear_consistency(Za, Zb)["symmetric"] > 0.95
    assert AL.linear_consistency(Za, Zc)["symmetric"] < 0.1
    M = AL.consistency_matrix({"a": Za, "b": Zb, "c": Zc})
    assert isinstance(M, pd.DataFrame) and M.shape == (3, 3)
    assert np.isnan(np.diag(M.to_numpy())).all()
    assert M.loc["a", "b"] > 0.95 and M.loc["a", "c"] < 0.1


def test_procrustes_recovers_rotation():
    """Known rotation + scale + shift (small noise) is recovered."""
    rng = np.random.default_rng(0)
    Z_ref = rng.standard_normal((1000, 5))
    Q = special_ortho_group.rvs(5, random_state=3)
    Z_src = 0.5 * (Z_ref @ Q.T) + 2.0 + 0.01 * rng.standard_normal((1000, 5))
    aligned, T, r2 = AL.procrustes_align(Z_src, Z_ref)
    assert r2 > 0.99
    assert np.allclose(T["R"], Q, atol=0.02)          # Z_src @ Q ~ Z_ref
    assert T["scale"] == pytest.approx(2.0, rel=0.02)
    assert np.allclose(AL.apply_transform(Z_src, T), aligned)


def test_cca_align_time_matched():
    """Canonical correlations ~1 for a noisy invertible linear map."""
    rng = np.random.default_rng(0)
    A = gaussian_filter1d(rng.standard_normal((1500, 4)), 3, axis=0)
    B = A @ rng.standard_normal((4, 4)) + 0.01 * rng.standard_normal((1500, 4))
    C1, C2, corr, T = AL.cca_align(A, B)
    assert corr.shape == (4,) and corr.min() > 0.95
    assert np.allclose(C1.std(0), 1.0, atol=0.05)
    assert AL._pooled_r2(B, AL.apply_transform(A, T)) > 0.95


@pytest.mark.parametrize("method", ["procrustes", "cca"])
def test_align_sessions_ring(method):
    """Sessions with different neurons and lengths, same ring latent:
    behaviour-anchored alignment generalises to held-out anchors, and
    permuting the anchor correspondence destroys it."""
    Z_ref, th_ref = _ring_session(4000, 40, seed=1)
    Z_new, th_new = _ring_session(2500, 55, seed=2)
    aligned, T, r2 = AL.align_sessions(Z_ref, th_ref, Z_new, th_new, method=method,
                                       circular=True, n_null=10)
    assert aligned.shape == (2500, Z_ref.shape[1])
    assert r2 > 0.8
    assert np.percentile(T["null_r2"], 95) < 0.5


def test_behavior_anchors_discrete_and_circular():
    """One anchor per label; circular bins on [-pi, pi)."""
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((900, 3))
    lab = np.repeat([0, 1, 2], 300)
    Z[lab == 1] += 5.0
    C, keys, cnt = AL.behavior_anchors(Z, lab)
    assert list(keys) == [0, 1, 2] and list(cnt) == [300, 300, 300]
    assert C[1].mean() > 4.0 and abs(C[0].mean()) < 0.5
    C, keys, cnt = AL.behavior_anchors(Z, rng.uniform(-np.pi, np.pi, 900), n_bins=8,
                                       circular=True)
    assert C.shape == (8, 3) and cnt.sum() == 900 and keys.min() > -np.pi


# --------------------------------------------------------------------------- #
# manifold                                                                     #
# --------------------------------------------------------------------------- #

def _noisy_trajectory(n=700, d=20, noise=0.6, seed=1):
    """Slow 3-D random curve embedded in d dims plus white noise."""
    g = np.random.default_rng(seed)
    lat = gaussian_filter1d(np.cumsum(g.standard_normal((n, 3)), 0), 5, axis=0)
    X = np.tanh(lat / lat.std()) @ g.standard_normal((3, d))
    return X + noise * X.std() * g.standard_normal((n, d))


def test_tphate_shape_and_temporal_neighbours():
    """Finite (n, 2) output; time neighbours are better preserved than by PCA
    (trustworthiness with *time* as the reference space)."""
    X = _noisy_trajectory()
    Y, idx, info = MF.tphate(X, return_info=True)
    assert Y.shape == (len(X), 2) and np.isfinite(Y).all()
    assert info["t"] >= 1 and info["lag"] >= 1
    t = np.arange(len(X), dtype=float)[:, None]
    tw_tphate = trustworthiness(t, Y, n_neighbors=10)
    tw_pca = trustworthiness(t, PCA(2).fit_transform(X), n_neighbors=10)
    assert tw_tphate > tw_pca + 0.02


def test_flushed_matrix_power_matches_numpy():
    """Denormal-safe repeated squaring equals numpy's matrix_power."""
    rng = np.random.default_rng(0)
    P = MF._row_normalise(rng.random((60, 60)))
    for t in (1, 2, 7, 16):
        assert np.allclose(MF._matrix_power_flushed(P, t), np.linalg.matrix_power(P, t),
                           atol=1e-6)


def test_tphate_strided_subset_and_project3d():
    """Strided subset like analysis.project_subset; 3-D projections."""
    X = _noisy_trajectory(n=1300)
    Y, idx = MF.tphate(X, max_points=500, mds="classic")
    assert len(idx) == len(Y) <= 500 and np.all(np.diff(idx) == idx[1] - idx[0])
    P3, idx3 = MF.project3d(X, "pca", max_points=1000)
    assert P3.shape == (len(idx3), 3)
    P3, idx3 = MF.project3d(X, "tphate", max_points=400)
    assert P3.shape == (len(idx3), 3) and np.isfinite(P3).all()


# --------------------------------------------------------------------------- #
# figures                                                                      #
# --------------------------------------------------------------------------- #

def _save(fig, tmp_path, name):
    """Every figure must be a Figure and save to png without error."""
    assert isinstance(fig, Figure)
    out = viz.save_figure(fig, tmp_path / name, formats=("png",))
    assert (tmp_path / f"{name}.png").exists() and len(out) == 1


def test_every_figure_renders(small_recording, tmp_path):
    """Smoke test of all figure functions on realistic data."""
    r = small_recording
    Z, S, t = r["Z"], r["states"], r["times"]
    if HAS_RIPSER:
        _save(VD.persistence_figure(Z, max_points=300), tmp_path, "persistence")
    P = PCA(2).fit_transform(Z)
    idx = np.arange(len(Z))
    _save(VD.flow_field_figure(P, idx, dt=0.1), tmp_path, "flow")
    _save(VD.flow_field_figure(P[::3], idx[::3], dt=0.1, states=S), tmp_path, "flow_states")
    _save(VD.recurrence_figure(Z, times=t, max_points=500), tmp_path, "recurrence")
    _save(VD.recurrence_figure(Z, max_points=300, metric="euclidean"), tmp_path, "recurrence_e")
    _save(VD.contingency_figure(S, r["beh"]["state"], n_perm=20), tmp_path, "contingency")
    M = AL.consistency_matrix({"z": Z, "z_rot": Z[:, ::-1], "z_smooth":
                               gaussian_filter1d(Z, 8, axis=0)})
    _save(VD.consistency_figure(M), tmp_path, "consistency")


def test_contingency_table_perfect_relabelling():
    """States that are a permutation of the labels: ARI = 1, Hungarian order."""
    lab = np.repeat([0, 1, 2], 200)
    states = np.array([2, 0, 1])[lab]
    ct = VD.contingency_table(states, lab, n_perm=20)
    assert ct["ari"] == pytest.approx(1.0) and ct["homogeneity"] == pytest.approx(1.0)
    assert list(ct["order"]) == [2, 0, 1]
    assert np.allclose(np.diag(ct["table"].to_numpy()), 1.0)


def test_rastermap_figure_with_and_without_package(small_recording, tmp_path, monkeypatch):
    """Rastermap ordering when installed, documented fallback otherwise."""
    r = small_recording
    X, S, t, ph = r["X"], r["states"], r["times"], r["beh"]["phase"]
    try:
        import rastermap  # noqa: F401
        has_rm = True
    except ImportError:
        has_rm = False
    if has_rm:
        fig = VD.rastermap_figure(X, times=t, states=S, behavior=ph, behavior_name="phase",
                                  circular=True, max_cols=500)
        _save(fig, tmp_path, "raster_rm")
        assert fig.nemba_sort_method == "rastermap"
        assert sorted(fig.nemba_order) == list(range(X.shape[0]))
    monkeypatch.setitem(sys.modules, "rastermap", None)     # import now fails
    fig = VD.rastermap_figure(X, times=t, states=S, behavior=r["beh"]["speed"],
                              behavior_name="speed", max_cols=500)
    _save(fig, tmp_path, "raster_fallback")
    assert fig.nemba_sort_method == "PC1-PC2 angle"
    assert sorted(fig.nemba_order) == list(range(X.shape[0]))
    fig = VD.rastermap_figure(X)                             # no tracks at all
    _save(fig, tmp_path, "raster_bare")
