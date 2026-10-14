"""Tests for the extra decoders of ``embedders_backbones.decoding``:
gradient-boosted trees (model="gbm") and the Bayesian Poisson population
decoder (bayes_decode_target), plus their integration in decode_behavior.

All simulations are small so the file runs in well under 90 s on a laptop."""

import warnings

import numpy as np
import pytest

from embedders_backbones import decoding as DEC

# columns of the decode_behavior table before the new decoders were added
# (backward compatibility: they must stay exactly the same)
BASE_COLUMNS = ["target", "features", "task", "model", "metric", "score", "score_sd",
                "fold_scores", "chance", "null_mean", "null_p95", "p_value", "n_perm",
                "mae_deg", "pearson_r", "n_frames", "n_folds", "gap", "n_features", "p_holm"]


# --------------------------------------------------------------------------- #
# simulation helpers                                                           #
# --------------------------------------------------------------------------- #

def _ar1(T, rho, rng):
    """Unit-innovation AR(1) series (strongly autocorrelated for rho ~ 1)."""
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + rng.normal()
    return x


def _von_mises_population(T, C, rng, kappa=2.0, gain=1.5, base=0.05):
    """Head-direction-like angle (random walk) and Poisson counts [C, T] of
    neurons with von Mises tuning to it (expected counts per frame)."""
    theta = np.angle(np.exp(1j * np.cumsum(0.08 * rng.normal(size=T) + 0.01)))
    pref = rng.uniform(-np.pi, np.pi, C)
    rate = base + gain * np.exp(kappa * (np.cos(theta[None] - pref[:, None]) - 1.0))
    return theta, rng.poisson(rate)


def _class_population(T, C, K, rng, dwell=60):
    """Piecewise-constant labels with ``dwell``-frame runs and Poisson counts
    [C, T] whose rate depends on the class (class-specific rate vectors,
    ~1 count per frame on average)."""
    lab = np.repeat(rng.integers(0, K, T // dwell), dwell)
    rates = rng.gamma(2.0, 0.5, size=(C, K))
    return lab, rng.poisson(rates[:, lab])


# --------------------------------------------------------------------------- #
# gradient boosting                                                            #
# --------------------------------------------------------------------------- #

def test_gbm_finds_nonlinear_signal_that_linear_misses():
    """y = sin(3 z0) is non-monotonic in z0: a linear read-out cannot use it,
    boosted trees can."""
    rng = np.random.default_rng(0)
    T = 2000
    Z = np.c_[_ar1(T, 0.95, rng), _ar1(T, 0.95, rng), rng.normal(size=(T, 2))]
    Z = (Z - Z.mean(0)) / Z.std(0)
    y = np.sin(3 * Z[:, 0]) + 0.3 * rng.normal(size=T)
    lin = DEC.decode_target(Z, y, model="linear", n_perm=19)
    gbm = DEC.decode_target(Z, y, model="gbm", n_perm=19)
    assert gbm["model"] == "gbm" and gbm["task"] == "regression"
    assert gbm["score"] > 0.5
    assert gbm["score"] > lin["score"] + 0.3
    assert gbm["p_value"] <= 0.05                       # = 1 / 20, the floor for 19 shifts


def test_gbm_circular_and_classification_tasks():
    rng = np.random.default_rng(1)
    T = 1200
    Z = np.c_[_ar1(T, 0.95, rng), _ar1(T, 0.95, rng)]
    Z = (Z - Z.mean(0)) / Z.std(0)
    th = np.angle(np.exp(1j * 2.0 * Z[:, 0]))           # angle, non-linear in Z
    rc = DEC.decode_target(Z, th, task="circular", model="gbm", n_perm=4)
    assert rc["score"] > 0.8 and rc["mae_deg"] < 30 and rc["null"].size == 4
    lab = (Z[:, 1] > 0.5).astype(int) + (Z[:, 1] > -0.5).astype(int)   # 3 ordered classes
    rk = DEC.decode_target(Z, lab, model="gbm", n_perm=4)
    assert rk["task"] == "classification" and rk["score"] > 0.8


def test_gbm_null_is_capped_with_a_warning(monkeypatch):
    monkeypatch.setattr(DEC, "GBM_MAX_PERM", 3)
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(600, 3))
    y = Z[:, 0] + 0.5 * rng.normal(size=600)
    with pytest.warns(UserWarning, match="GBM_MAX_PERM"):
        r = DEC.decode_target(Z, y, model="gbm", n_perm=50)
    assert r["null"].size == 3


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        DEC.decode_target(np.zeros((100, 2)), np.arange(100.0), model="gbn")


# --------------------------------------------------------------------------- #
# Bayesian population decoder                                                  #
# --------------------------------------------------------------------------- #

def test_bayes_recovers_circular_variable_from_von_mises_neurons():
    rng = np.random.default_rng(3)
    theta, counts = _von_mises_population(3000, 40, rng)
    r = DEC.bayes_decode_target(counts, theta, task="circular", n_perm=99)
    assert r["model"] == "bayes" and r["metric"] == "mean_cos_dtheta"
    assert r["score"] > 0.9
    assert r["mae_deg"] < 20
    assert r["p_value"] <= 0.02
    # orientation [T, C] is accepted when unambiguous, with identical results
    r2 = DEC.bayes_decode_target(counts.T, theta, task="circular", n_perm=99)
    assert np.isclose(r2["score"], r["score"])


def test_bayes_recovers_discrete_variable_from_class_specific_rates():
    rng = np.random.default_rng(4)
    lab, counts = _class_population(2400, 40, 4, rng)
    r = DEC.bayes_decode_target(counts, lab, n_perm=99)
    assert r["task"] == "classification" and abs(r["chance"] - 0.25) < 1e-9
    assert r["score"] > 0.9
    assert r["p_value"] <= 0.02


def test_bayes_regression_on_a_bounded_track():
    rng = np.random.default_rng(5)
    T, C = 3000, 40
    pos = np.abs(((np.cumsum(0.03 * rng.normal(size=T)) + 1.0) % 2.0) - 1.0)   # reflecting walk
    ctr = rng.uniform(0, 1, C)
    counts = rng.poisson(0.05 + np.exp(-0.5 * ((pos[None] - ctr[:, None]) / 0.07) ** 2))
    r = DEC.bayes_decode_target(counts, pos, task="regression", n_perm=49)
    assert r["score"] > 0.7 and r["pearson_r"] > 0.85


def test_bayes_null_is_calibrated_for_independent_labels():
    """Counts tuned to one AR(1) process, target = a circularly shifted AR(1)
    of an unrelated process: p-values must not be systematically small."""
    ps = []
    for s in range(12):
        rng = np.random.default_rng(500 + s)
        x = _ar1(1500, 0.99, rng)
        x /= x.std()
        pref = rng.normal(size=20)
        counts = rng.poisson(0.2 + np.exp(-0.5 * (x[None] - pref[:, None]) ** 2))
        y = np.roll(_ar1(1500, 0.99, rng), 500)
        ps.append(DEC.bayes_decode_target(counts, y, task="regression", n_perm=49,
                                          seed=s)["p_value"])
    ps = np.asarray(ps)
    assert np.sum(ps < 0.05) <= 3                       # nominal 0.6 of 12
    assert ps.mean() > 0.25                             # uniform p-values average 0.5


def test_bayes_rejects_negative_features():
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError, match="non-negative"):
        DEC.bayes_decode_target(rng.normal(size=(5, 300)), rng.normal(size=300), n_perm=0)


# --------------------------------------------------------------------------- #
# decode_behavior integration                                                  #
# --------------------------------------------------------------------------- #

def test_decode_behavior_adds_bayes_rows_and_keeps_columns():
    rng = np.random.default_rng(7)
    T, C = 1500, 24
    theta, counts = _von_mises_population(T, C, rng)
    lab = np.repeat(rng.integers(0, 2, T // 50), 50)
    counts = counts + rng.poisson(0.5 + 1.0 * lab[None] * (np.arange(C)[:, None] % 2))
    Z = np.c_[np.cos(theta), np.sin(theta), lab] + 0.2 * rng.normal(size=(T, 3))
    base = rng.normal(size=(T, 3))
    beh = {"theta": theta, "state": lab}
    kw = dict(circular=["theta"], n_perm=39, smooth_sigma=1.0, baseline=base)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = DEC.decode_behavior(Z, beh, **kw).table
        new = DEC.decode_behavior(Z, beh, counts=counts, **kw).table
        no_base = DEC.decode_behavior(Z, beh, n_perm=39, circular=["theta"]).table

    # without counts: exactly the previous columns (+ delta_vs_raw with a baseline)
    assert list(no_base.columns) == BASE_COLUMNS
    assert list(old.columns) == BASE_COLUMNS + ["delta_vs_raw"]
    # with counts: same columns, two extra rows
    assert list(new.columns) == list(old.columns)
    assert len(new) == len(old) + 2
    bay = new[new.features == "bayes_counts"].set_index("target")
    assert set(bay.index) == {"theta", "state"}
    assert (bay["model"] == "bayes").all() and (bay["n_features"] == C).all()
    assert bay.loc["theta", "task"] == "circular" and bay.loc["state", "task"] == "classification"
    assert bay.loc["theta", "score"] > 0.8 and bay.loc["state", "score"] > 0.8
    # Holm within the feature set, delta_vs_raw against the raw_pca rows
    assert np.allclose(bay["p_holm"].to_numpy(), DEC.holm(bay["p_value"].to_numpy()))
    raw = new[new.features == "raw_pca"].set_index("target")["score"]
    assert np.allclose(bay["delta_vs_raw"], bay["score"] - raw.reindex(bay.index))
    # the pre-existing rows are unchanged by adding the counts
    cols = ["target", "features", "score", "p_value", "p_holm", "delta_vs_raw"]
    assert old[cols].equals(new[new.features != "bayes_counts"][cols].reset_index(drop=True))


def test_matched_count_window():
    assert DEC.matched_count_window(0) == 1
    assert DEC.matched_count_window(4.0) == 15          # boxcar sd ~ 4.3 frames
