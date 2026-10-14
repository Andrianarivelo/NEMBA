"""Tests for ``embedders_backbones.states``: sticky Gaussian HMM, AR-HMM,
blocked-CV model selection and the numba / NumPy kernels.

All data are simulated with a known state sequence, so every test checks
recovery against ground truth. Runtime: well under a minute on a laptop CPU.
"""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from embedders_backbones import states as ST


# --------------------------------------------------------------------------- #
# simulators                                                                  #
# --------------------------------------------------------------------------- #

def _markov_chain(T, K, p_stay, rng):
    """Sticky Markov chain: stay with probability p_stay, else jump uniformly
    to one of the other K - 1 states."""
    s = np.empty(T, dtype=int)
    s[0] = rng.integers(K)
    for t in range(1, T):
        if rng.random() < p_stay:
            s[t] = s[t - 1]
        else:
            s[t] = rng.choice([k for k in range(K) if k != s[t - 1]])
    return s


def sticky_gaussian_hmm(T=3000, K=4, D=8, p_stay=0.98, sep=1.0, noise=1.0, seed=0):
    """Frames z_t = mu_{s_t} + N(0, noise^2 I), state means ~ N(0, sep^2 I)."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, sep, (K, D))
    s = _markov_chain(T, K, p_stay, rng)
    return means[s] + noise * rng.standard_normal((T, D)), s


def switching_ar(T=3000, p_stay=0.99, noise=0.3, seed=0):
    """2-D switching AR(1) with three regimes sharing the same (zero) mean:
    counter-clockwise rotation, clockwise rotation, fast isotropic decay."""
    rng = np.random.default_rng(seed)

    def rot(a):
        """2-D rotation matrix by angle a (radians)."""
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    A = [0.97 * rot(0.25), 0.97 * rot(-0.25), 0.6 * np.eye(2)]
    s = _markov_chain(T, 3, p_stay, rng)
    z = np.zeros((T, 2))
    for t in range(1, T):
        z[t] = A[s[t]] @ z[t - 1] + noise * rng.standard_normal(2)
    return z, s


# --------------------------------------------------------------------------- #
# model selection and recovery                                                #
# --------------------------------------------------------------------------- #

def test_select_k_recovers_four_states_and_labels():
    """Blocked CV + 1-SE rule finds K = 4 on a sticky 4-state HMM in 8-D, and
    the decoded states match the truth (ARI > 0.9)."""
    Z, s = sticky_gaussian_hmm(T=2400, seed=1)
    best_k, table = ST.select_k(Z, method="hmm", k_range=(2, 5), n_folds=3, seed=0, n_restarts=1)
    assert {"k", "heldout_ll_mean", "heldout_ll_se", "bic"} <= set(table.columns)
    assert list(table["k"]) == [2, 3, 4, 5]
    assert best_k == 4
    res = ST.fit_states(Z, method="hmm", k=best_k, seed=0)
    assert adjusted_rand_score(s, res["labels"]) > 0.9
    assert res["posteriors"].shape == (len(Z), 4)


def test_fit_states_auto_k_returns_selection_table():
    """k=None runs select_k inside fit_states and reuses its full-data model."""
    Z, s = sticky_gaussian_hmm(T=1500, K=3, D=4, sep=2.0, seed=2)
    res = ST.fit_states(Z, method="hmm", k=None, k_range=(2, 4), seed=0, n_restarts=1)
    assert res["k"] == 3
    assert res["selection"] is not None and res["selection"]["selected"].sum() == 1
    assert adjusted_rand_score(s, res["labels"]) > 0.9


def test_select_k_flags_continuous_manifold():
    """A slowly drifting point on a ring has no natural number of states: the
    held-out likelihood keeps rising with K, so K hits k_max and a warning is
    issued (the same failure mode as the silhouette, made explicit)."""
    rng = np.random.default_rng(9)
    phase = np.cumsum(0.03 + 0.05 * rng.standard_normal(1200))
    Z = np.c_[np.cos(phase), np.sin(phase)] + 0.05 * rng.standard_normal((1200, 2))
    with pytest.warns(UserWarning, match="upper bound"):
        best_k, table = ST.select_k(Z, method="hmm", k_range=(2, 4), n_folds=3, seed=0, n_restarts=1)
    assert best_k == 4 and table.attrs["at_upper_bound"]


def test_arhmm_beats_gaussian_hmm_on_dynamics_defined_states():
    """Regimes with the same mean but different dynamics (rotation direction,
    decay): only the AR-HMM can tell them apart."""
    Z, s = switching_ar(seed=3)
    ari_hmm = adjusted_rand_score(s, ST.fit_states(Z, "hmm", k=3, seed=0)["labels"])
    ari_ar = adjusted_rand_score(s, ST.fit_states(Z, "arhmm", k=3, seed=0)["labels"])
    assert ari_ar > 0.8
    assert ari_ar > ari_hmm + 0.3


# --------------------------------------------------------------------------- #
# kernels: numba and NumPy must agree                                         #
# --------------------------------------------------------------------------- #

def _random_problem(T=400, K=5, seed=0):
    """Random start / transition probabilities and log emissions."""
    rng = np.random.default_rng(seed)
    pi = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K), size=K) + 5 * np.eye(K)
    A /= A.sum(1, keepdims=True)
    logB = rng.normal(0, 3, (T, K)) - 50.0      # large offset: tests the max-shifts
    return pi, A, logB


@pytest.mark.skipif(not ST.HAS_NUMBA, reason="numba not installed")
def test_numba_and_numpy_kernels_agree():
    """Same log-likelihood (rtol 1e-6), posteriors, expected transitions and
    Viterbi path from both backends."""
    pi, A, logB = _random_problem()
    g_np, xi_np, ll_np = ST.forward_backward(pi, A, logB, backend="numpy")
    g_nb, xi_nb, ll_nb = ST.forward_backward(pi, A, logB, backend="numba")
    assert np.isclose(ll_np, ll_nb, rtol=1e-6)
    np.testing.assert_allclose(g_np, g_nb, atol=1e-9)
    np.testing.assert_allclose(xi_np, xi_nb, rtol=1e-6)
    assert np.isclose(ST.forward_loglik(pi, A, logB, backend="numpy"), ll_np, rtol=1e-10)
    path_np, lp_np = ST.viterbi(pi, A, logB, backend="numpy")
    path_nb, lp_nb = ST.viterbi(pi, A, logB, backend="numba")
    np.testing.assert_array_equal(path_np, path_nb)
    assert np.isclose(lp_np, lp_nb, rtol=1e-10)


@pytest.mark.skipif(not ST.HAS_NUMBA, reason="numba not installed")
def test_numba_and_numpy_fitted_models_agree():
    """A model fitted with each backend reaches the same likelihood and path."""
    Z, _ = sticky_gaussian_hmm(T=1000, K=3, D=4, sep=0.8, seed=4)
    m_np = ST.GaussianHMM(3, n_restarts=1, backend="numpy").fit(Z)
    m_nb = ST.GaussianHMM(3, n_restarts=1, backend="numba").fit(Z)
    assert np.isclose(m_np.loglik_, m_nb.loglik_, rtol=1e-6)
    assert np.isclose(m_np.score(Z), m_nb.score(Z), rtol=1e-6)
    np.testing.assert_array_equal(m_np.predict(Z), m_nb.predict(Z))


def test_forward_loglik_matches_brute_force_enumeration():
    """On a tiny problem the forward recursion equals the sum over all K^T
    state paths (checks the kernels against the definition)."""
    import itertools
    pi, A, logB = _random_problem(T=5, K=3, seed=7)
    total = []
    for path in itertools.product(range(3), repeat=5):
        lp = np.log(pi[path[0]]) + logB[0, path[0]]
        lp += sum(np.log(A[path[t - 1], path[t]]) + logB[t, path[t]] for t in range(1, 5))
        total.append(lp)
    m = max(total)
    brute = m + np.log(np.sum(np.exp(np.array(total) - m)))
    assert np.isclose(ST.forward_loglik(pi, A, logB, backend="numpy"), brute, rtol=1e-12)
    path, lp = ST.viterbi(pi, A, logB, backend="numpy")
    assert np.isclose(lp, m, rtol=1e-12)


def test_env_var_forces_numpy_backend(monkeypatch):
    """NEMBA_STATES_BACKEND=numpy overrides 'auto'; an explicit argument wins."""
    monkeypatch.setenv(ST.BACKEND_ENV, "numpy")
    assert ST._resolve_backend("auto") == "numpy"
    if ST.HAS_NUMBA:
        assert ST._resolve_backend("numba") == "numba"
    with pytest.raises(ValueError):
        ST._resolve_backend("cuda")


# --------------------------------------------------------------------------- #
# sticky prior, EM monotonicity, posteriors, utilities                        #
# --------------------------------------------------------------------------- #

def test_sticky_prior_reduces_switches_on_noisy_data():
    """Heavily overlapping emissions and short recordings: without the sticky
    prior EM settles on a fast-switching solution; with it, fewer switches."""
    sw_free, sw_sticky, sw_true = 0, 0, 0
    for seed in (100, 101, 102):
        Z, s = sticky_gaussian_hmm(T=1500, K=4, D=8, sep=0.3, seed=seed)
        sw_free += ST.n_switches(ST.GaussianHMM(4, sticky=0.0, seed=0).fit(Z).predict(Z))
        sw_sticky += ST.n_switches(ST.GaussianHMM(4, sticky="auto", seed=0).fit(Z).predict(Z))
        sw_true += ST.n_switches(s)
    assert sw_sticky < sw_free
    assert sw_sticky < 2 * sw_true


@pytest.mark.parametrize("cls", [ST.GaussianHMM, ST.ARHMM])
def test_em_objective_is_monotone_and_posteriors_normalised(cls):
    """The MAP-EM objective (log-likelihood + log prior) never decreases, and
    posterior rows sum to 1."""
    Z, _ = sticky_gaussian_hmm(T=1200, K=3, D=4, sep=0.7, seed=5)
    m = cls(3, n_restarts=1, max_iter=60, tol=0.0, seed=0).fit(Z)
    h = m.history_
    assert len(h) > 5
    assert np.all(np.diff(h) >= -1e-7 * np.abs(h[:-1]))
    P = m.predict_proba(Z)
    assert P.shape == (len(Z), 3)
    np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-10)
    assert np.isfinite(m.score(Z)) and m.n_params() > 0


def test_maximum_likelihood_em_increases_score():
    """Without any prior (sticky = 0, no pseudo-counts) EM is plain maximum
    likelihood, so the log-likelihood itself is non-decreasing."""
    Z, _ = sticky_gaussian_hmm(T=1200, K=3, D=4, sep=0.7, seed=6)
    m = ST.GaussianHMM(3, sticky=0.0, trans_pseudocount=0.0, n_restarts=1, max_iter=60,
                       tol=0.0, seed=0).fit(Z)
    ll = m.loglik_history_
    assert np.all(np.diff(ll) >= -1e-7 * np.abs(ll[:-1]))
    assert ll[-1] > ll[0]


def test_list_input_treats_sequences_independently():
    """A list of sequences is fitted jointly; outputs come back as lists."""
    Z, s = sticky_gaussian_hmm(T=1600, K=3, D=4, sep=2.0, seed=8)
    parts = [Z[:700], Z[700:]]
    m = ST.GaussianHMM(3, n_restarts=1).fit(parts)
    labs = m.predict(parts)
    assert isinstance(labs, list) and [len(x) for x in labs] == [700, 900]
    assert adjusted_rand_score(s, np.concatenate(labs)) > 0.9
    assert np.isclose(m.score(parts), m.score(Z[:700]) + m.score(Z[700:]))


def test_dwell_times_and_switches():
    """Run lengths per state and the number of label changes."""
    lab = np.array([0, 0, 1, 1, 1, 0, 2, 2])
    d = ST.dwell_times(lab)
    assert set(d) == {0, 1, 2}
    np.testing.assert_array_equal(d[0], [2, 1])
    np.testing.assert_array_equal(d[1], [3])
    np.testing.assert_array_equal(d[2], [2])
    assert ST.n_switches(lab) == 3
    assert ST.dwell_times(lab, n_states=4)[3].size == 0
    assert ST.dwell_times([]) == {}
