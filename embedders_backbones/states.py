"""Temporal state models for NEMBA embeddings: sticky Gaussian HMM and AR-HMM.

Why
---
``analysis.cluster`` assigns every frame of an embedding ``Z [T, E]`` to a state
independently (K-means). Frame-wise clustering ignores that states persist in
time: noise near a cluster boundary is read as a transition (jitter becomes
fake switches), and ``analysis.auto_k`` picks K by the silhouette, which has
no notion of time and is unreliable on continuous manifolds. A hidden Markov
model adds a transition matrix, so a frame is labelled by its own evidence
*and* by its neighbours (Linderman et al., AISTATS 2017, switching linear
dynamical systems; Weinreb et al., Nat Methods 2024, keypoint-MoSeq, showed
that frame-wise clustering of pose mistakes jitter for syllable transitions).

What is here
------------
GaussianHMM      diagonal Gaussian emission per state: states differ in *position*.
ARHMM            first-order vector autoregressive emission per state,
                 z_t = A_k z_{t-1} + b_k + noise: states differ in *dynamics*.
select_k         number of states by blocked cross-validated held-out
                 log-likelihood per frame and the 1-SE rule (BIC reported too).
fit_states       one call: standardise, select K (optional), fit, decode.
dwell_times      run lengths per state; ``n_switches`` counts label changes.
forward_backward, viterbi, forward_loglik
                 the kernels, usable with any log-emission matrix.

Fitting
-------
MAP Baum-Welch EM. E-step: forward-backward in log space (every step is
max-shifted before exponentiating, so nothing underflows at T = 1e5). M-step in
closed form with

* a sticky Dirichlet prior on every row i of the transition matrix:
  pseudo-count ``trans_pseudocount`` on every entry plus ``sticky`` on the
  self-transition (i, i) (sticky HMM, Fox et al., Ann Appl Stat 2011, used here
  as a MAP prior rather than in a sampler);
* a variance floor ``min_covar`` applied as a constraint (max(var, floor)),
  which keeps the M-step exact so the EM objective never decreases;
* for the AR-HMM, a zero-mean Gaussian prior (precision ``ridge``) on the AR
  coefficients: ridge-regularised weighted least squares, posterior weights.

The quantity EM maximises (log-likelihood + log prior) is stored per iteration
in ``history_`` and is non-decreasing up to floating point.

Speed
-----
The time recursions are compiled with numba when it is importable
(``HAS_NUMBA``), with an identical pure-NumPy fallback. Force the fallback with
``backend="numpy"`` (models and kernels) or the environment variable
``NEMBA_STATES_BACKEND=numpy``. numba is imported lazily, on the first fit.
"""

from __future__ import annotations

import importlib.util
import os
import time
import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d

__all__ = ["HAS_NUMBA", "STICKY_FRAC", "TRANS_PSEUDOCOUNT", "GaussianHMM", "ARHMM",
           "make_model", "select_k", "fit_states", "dwell_times", "n_switches",
           "default_sticky", "forward_backward", "viterbi", "forward_loglik"]

# --------------------------------------------------------------------------- #
# module constants                                                            #
# --------------------------------------------------------------------------- #

#: True when numba can be imported (checked without importing it: cheap).
HAS_NUMBA = importlib.util.find_spec("numba") is not None

#: Environment variable that selects the kernel backend: "auto" | "numba" | "numpy".
BACKEND_ENV = "NEMBA_STATES_BACKEND"

#: Default stickiness: ``sticky="auto"`` adds kappa = STICKY_FRAC * T / K
#: pseudo-transitions to every self-transition, i.e. a prior worth this fraction
#: of the frames an average state occupies. For a state of average occupancy it
#: shrinks the maximum-likelihood switching rate by 1 / (1 + STICKY_FRAC), i.e.
#: the Viterbi cost of leaving the state grows by log(1 + STICKY_FRAC) nats.
#: Being relative to T, its effect does not fade or grow with recording length.
STICKY_FRAC = 0.1

#: Small symmetric Dirichlet pseudo-count on every transition (and start) entry:
#: keeps every transition possible (no -inf on held-out data).
TRANS_PSEUDOCOUNT = 0.1

#: Frames used to fit the K-means initialisation (random subsample beyond this).
MAX_INIT_FRAMES = 20000

#: k-means++ seedings per K-means initialisation (best inertia kept).
KMEANS_N_INIT = 10

_LOG2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
# backend selection                                                           #
# --------------------------------------------------------------------------- #

_NB_KERNELS: dict = {}   # compiled numba kernels, filled on first use


def _resolve_backend(backend="auto"):
    """Return "numba" or "numpy" for a requested backend.

    ``backend="auto"`` defers to the environment variable ``NEMBA_STATES_BACKEND``
    (also "auto" when unset), and "auto" there means numba when available. An
    explicit "numba" without numba installed raises ImportError."""
    b = str(backend or "auto").lower()
    if b == "auto":
        b = os.environ.get(BACKEND_ENV, "auto").strip().lower() or "auto"
    if b == "auto":
        return "numba" if HAS_NUMBA else "numpy"
    if b not in ("numba", "numpy"):
        raise ValueError(f"backend must be 'auto', 'numba' or 'numpy', got {backend!r}")
    if b == "numba" and not HAS_NUMBA:
        raise ImportError("backend='numba' requested but numba is not installed")
    return b


def _numba_kernels():
    """Compile (once per process, cached on disk by numba) and return the kernels.

    On any import / compilation failure, numba is disabled for the session
    (``HAS_NUMBA`` set to False) and the caller falls back to NumPy."""
    global HAS_NUMBA
    if not _NB_KERNELS:
        try:
            import numba
            kernels = None
            # try with the on-disk cache first (fast next sessions), then without
            # it (e.g. read-only install directory)
            for cache in (True, False):
                try:
                    jit = numba.njit(cache=cache, nogil=True)
                    kernels = {name: jit(fn) for name, fn in _LOOP_KERNELS.items()}
                    _warm_up(kernels)            # compile now, so failures surface here
                    break
                except Exception:  # noqa: BLE001
                    if not cache:
                        raise
            _NB_KERNELS.update(kernels)
        except Exception as exc:  # noqa: BLE001
            HAS_NUMBA = False
            warnings.warn(f"numba unavailable ({exc!r}); using the NumPy kernels")
            return None
    return _NB_KERNELS


def _warm_up(kernels):
    """Call every compiled kernel once on a tiny problem (triggers compilation
    for the float64 signatures used everywhere in this module)."""
    log_pi = np.log(np.full(2, 0.5))
    A = np.full((2, 2), 0.5)
    logB = np.zeros((3, 2))
    la = kernels["fwd"](log_pi, A, logB)
    lb = kernels["bwd"](A, logB)
    kernels["post"](la, lb, A, logB)
    kernels["viterbi"](log_pi, np.log(A), logB)


# --------------------------------------------------------------------------- #
# loop kernels (compiled by numba; never run as plain Python, far too slow)  #
# --------------------------------------------------------------------------- #
# Conventions: log_pi [K] log start probabilities, A [K, K] transition
# probabilities (rows sum to 1), logA = log(A), logB [T, K] log emissions.

def _fwd_loop(log_pi, A, logB):
    """Forward pass in log space: la[t, j] = log p(z_0..z_t, s_t = j).

    Each step max-shifts the previous row before exponentiating, so the
    recursion is exact in log space without calling logsumexp K * K times."""
    T, K = logB.shape
    la = np.empty((T, K))
    w = np.empty(K)
    for k in range(K):
        la[0, k] = log_pi[k] + logB[0, k]
    for t in range(1, T):
        m = la[t - 1, 0]
        for i in range(1, K):
            if la[t - 1, i] > m:
                m = la[t - 1, i]
        if m == -np.inf:                     # impossible prefix: stays impossible
            for j in range(K):
                la[t, j] = -np.inf
            continue
        for i in range(K):
            w[i] = np.exp(la[t - 1, i] - m)
        for j in range(K):
            s = 0.0
            for i in range(K):
                s += w[i] * A[i, j]
            la[t, j] = logB[t, j] + m + np.log(s) if s > 0.0 else -np.inf
    return la


def _bwd_loop(A, logB):
    """Backward pass in log space: lb[t, i] = log p(z_{t+1}..z_{T-1} | s_t = i)."""
    T, K = logB.shape
    lb = np.empty((T, K))
    v = np.empty(K)
    for k in range(K):
        lb[T - 1, k] = 0.0
    for t in range(T - 2, -1, -1):
        m = -np.inf
        for j in range(K):
            v[j] = logB[t + 1, j] + lb[t + 1, j]
            if v[j] > m:
                m = v[j]
        if m == -np.inf:
            for i in range(K):
                lb[t, i] = -np.inf
            continue
        for j in range(K):
            v[j] = np.exp(v[j] - m)
        for i in range(K):
            s = 0.0
            for j in range(K):
                s += A[i, j] * v[j]
            lb[t, i] = m + np.log(s) if s > 0.0 else -np.inf
    return lb


def _post_loop(la, lb, A, logB):
    """Posteriors gamma[t, k] = p(s_t = k | z) and summed pairwise posteriors
    xi[i, j] = sum_t p(s_{t-1} = i, s_t = j | z).

    Every per-step pairwise table is normalised by its own sum (which equals the
    likelihood up to the shifts), so no global log-likelihood is needed here."""
    T, K = la.shape
    gamma = np.empty((T, K))
    xi = np.zeros((K, K))
    a = np.empty(K)
    b = np.empty(K)
    for t in range(T):
        m = -np.inf
        for k in range(K):
            if la[t, k] + lb[t, k] > m:
                m = la[t, k] + lb[t, k]
        s = 0.0
        for k in range(K):
            gamma[t, k] = np.exp(la[t, k] + lb[t, k] - m)
            s += gamma[t, k]
        for k in range(K):
            gamma[t, k] /= s
    for t in range(1, T):
        ma = -np.inf
        mb = -np.inf
        for i in range(K):
            if la[t - 1, i] > ma:
                ma = la[t - 1, i]
            if logB[t, i] + lb[t, i] > mb:
                mb = logB[t, i] + lb[t, i]
        for i in range(K):
            a[i] = np.exp(la[t - 1, i] - ma)
            b[i] = np.exp(logB[t, i] + lb[t, i] - mb)
        s = 0.0
        for i in range(K):
            for j in range(K):
                s += a[i] * A[i, j] * b[j]
        if s > 0.0:
            for i in range(K):
                ai = a[i] / s
                for j in range(K):
                    xi[i, j] += ai * A[i, j] * b[j]
    return gamma, xi


def _viterbi_loop(log_pi, logA, logB):
    """Most probable state path (ties resolved to the lowest state index)."""
    T, K = logB.shape
    prev = np.empty(K)
    cur = np.empty(K)
    psi = np.zeros((T, K), dtype=np.int64)
    for k in range(K):
        prev[k] = log_pi[k] + logB[0, k]
    for t in range(1, T):
        for j in range(K):
            best = prev[0] + logA[0, j]
            arg = 0
            for i in range(1, K):
                v = prev[i] + logA[i, j]
                if v > best:                 # strict: first maximum wins, like argmax
                    best = v
                    arg = i
            cur[j] = best + logB[t, j]
            psi[t, j] = arg
        for j in range(K):
            prev[j] = cur[j]
    path = np.empty(T, dtype=np.int64)
    best = prev[0]
    arg = 0
    for k in range(1, K):
        if prev[k] > best:
            best = prev[k]
            arg = k
    path[T - 1] = arg
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path, best


_LOOP_KERNELS = dict(fwd=_fwd_loop, bwd=_bwd_loop, post=_post_loop, viterbi=_viterbi_loop)


# --------------------------------------------------------------------------- #
# NumPy kernels (same algorithm, vectorised over states, loop over time)      #
# --------------------------------------------------------------------------- #

def _fwd_numpy(log_pi, A, logB):
    """NumPy twin of ``_fwd_loop``."""
    T, K = logB.shape
    la = np.empty((T, K))
    la[0] = log_pi + logB[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in range(1, T):
            prev = la[t - 1]
            m = prev.max()
            if m == -np.inf:
                la[t] = -np.inf
                continue
            la[t] = logB[t] + m + np.log(np.exp(prev - m) @ A)
    return la


def _bwd_numpy(A, logB):
    """NumPy twin of ``_bwd_loop``."""
    T, K = logB.shape
    lb = np.empty((T, K))
    lb[T - 1] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in range(T - 2, -1, -1):
            v = logB[t + 1] + lb[t + 1]
            m = v.max()
            if m == -np.inf:
                lb[t] = -np.inf
                continue
            lb[t] = m + np.log(A @ np.exp(v - m))
    return lb


def _post_numpy(la, lb, A, logB):
    """NumPy twin of ``_post_loop`` (fully vectorised over time)."""
    T, K = la.shape
    g = la + lb
    g = np.exp(g - g.max(1, keepdims=True))
    g /= g.sum(1, keepdims=True)
    xi = np.zeros((K, K))
    if T > 1:
        a = np.exp(la[:-1] - la[:-1].max(1, keepdims=True))          # [T-1, K]
        v = logB[1:] + lb[1:]
        b = np.exp(v - v.max(1, keepdims=True))                       # [T-1, K]
        # per-step normaliser s_t = sum_ij a_ti A_ij b_tj, then
        # xi = sum_t (a_t / s_t)^T b_t elementwise-times A, without a [T, K, K] array
        s = ((a @ A) * b).sum(1)
        ok = s > 0
        xi = A * ((a[ok] / s[ok, None]).T @ b[ok])
    return g, xi


def _viterbi_numpy(log_pi, logA, logB):
    """NumPy twin of ``_viterbi_loop`` (np.argmax also keeps the first maximum)."""
    T, K = logB.shape
    psi = np.zeros((T, K), dtype=np.int64)
    cols = np.arange(K)
    prev = log_pi + logB[0]
    for t in range(1, T):
        v = prev[:, None] + logA                 # v[i, j] = delta_{t-1}(i) + log A_ij
        idx = v.argmax(0)
        psi[t] = idx
        prev = v[idx, cols] + logB[t]
    path = np.empty(T, dtype=np.int64)
    path[T - 1] = int(prev.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path, float(prev[path[T - 1]])


def _logsumexp_row(x):
    """log(sum(exp(x))) of a 1-D array, max-shifted (-inf if all -inf)."""
    m = float(np.max(x))
    if m == -np.inf:
        return -np.inf
    return m + float(np.log(np.exp(x - m).sum()))


def _prep_args(startprob, transmat, log_emission):
    """Cast kernel inputs to contiguous float64 and take the log of startprob."""
    logB = np.ascontiguousarray(log_emission, dtype=np.float64)
    if logB.ndim != 2 or logB.shape[0] < 1:
        raise ValueError("log_emission must be a [T, K] array with T >= 1")
    A = np.ascontiguousarray(transmat, dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_pi = np.log(np.ascontiguousarray(startprob, dtype=np.float64))
    return log_pi, A, logB


# --------------------------------------------------------------------------- #
# public kernels                                                              #
# --------------------------------------------------------------------------- #

def forward_backward(startprob, transmat, log_emission, backend="auto"):
    """Posterior state probabilities of one sequence.

    Parameters
    ----------
    startprob : [K] initial state probabilities.
    transmat : [K, K] transition probabilities, rows sum to 1.
    log_emission : [T, K] log p(z_t | s_t = k).
    backend : "auto" | "numba" | "numpy".

    Returns
    -------
    posteriors : [T, K], rows sum to 1.
    xi_sum : [K, K] expected transition counts sum_t p(s_{t-1}=i, s_t=j | z).
    loglik : float, log p(z_0..z_{T-1}).
    """
    log_pi, A, logB = _prep_args(startprob, transmat, log_emission)
    if _resolve_backend(backend) == "numba" and _numba_kernels() is not None:
        nb = _NB_KERNELS
        la = nb["fwd"](log_pi, A, logB)
        lb = nb["bwd"](A, logB)
        gamma, xi = nb["post"](la, lb, A, logB)
    else:
        la = _fwd_numpy(log_pi, A, logB)
        lb = _bwd_numpy(A, logB)
        gamma, xi = _post_numpy(la, lb, A, logB)
    return gamma, xi, _logsumexp_row(la[-1])


def forward_loglik(startprob, transmat, log_emission, backend="auto"):
    """Log-likelihood of one sequence (forward pass only)."""
    log_pi, A, logB = _prep_args(startprob, transmat, log_emission)
    if _resolve_backend(backend) == "numba" and _numba_kernels() is not None:
        la = _NB_KERNELS["fwd"](log_pi, A, logB)
    else:
        la = _fwd_numpy(log_pi, A, logB)
    return _logsumexp_row(la[-1])


def viterbi(startprob, transmat, log_emission, backend="auto"):
    """Most probable state path of one sequence.

    Returns (path [T] int64, log p(path, z))."""
    log_pi, A, logB = _prep_args(startprob, transmat, log_emission)
    with np.errstate(divide="ignore"):
        logA = np.log(A)
    if _resolve_backend(backend) == "numba" and _numba_kernels() is not None:
        path, best = _NB_KERNELS["viterbi"](log_pi, logA, logB)
        return path, float(best)
    return _viterbi_numpy(log_pi, logA, logB)


# --------------------------------------------------------------------------- #
# small utilities                                                             #
# --------------------------------------------------------------------------- #

def default_sticky(T, n_states, frac=None):
    """Self-transition pseudo-count used by ``sticky="auto"``:
    kappa = frac * T / n_states (frac defaults to ``STICKY_FRAC``)."""
    frac = STICKY_FRAC if frac is None else float(frac)
    return float(frac) * float(T) / float(n_states)


def dwell_times(labels, n_states=None):
    """Run lengths (frames) of every state in a label sequence.

    Returns {state: int array of run lengths}, in order of appearance. States in
    ``range(n_states)`` that never occur get an empty array."""
    lab = np.asarray(labels).ravel()
    if lab.size == 0:
        return {} if n_states is None else {k: np.zeros(0, int) for k in range(n_states)}
    starts = np.r_[0, np.flatnonzero(lab[1:] != lab[:-1]) + 1]
    runs = np.diff(np.r_[starts, lab.size])
    first = lab[starts]
    states = range(int(n_states)) if n_states is not None else np.unique(lab)
    return {int(k): runs[first == k].astype(int) for k in states}


def n_switches(labels):
    """Number of label changes between consecutive frames."""
    lab = np.asarray(labels).ravel()
    return int(np.count_nonzero(lab[1:] != lab[:-1])) if lab.size > 1 else 0


def _as_list(Z):
    """Normalise input to (list of 2-D float64 arrays, was_single_array)."""
    if isinstance(Z, (list, tuple)):
        seqs, single = [np.asarray(z, dtype=np.float64) for z in Z], False
    else:
        seqs, single = [np.asarray(Z, dtype=np.float64)], True
    for z in seqs:
        if z.ndim != 2:
            raise ValueError(f"each sequence must be a [T, D] array, got shape {z.shape}")
        if not np.all(np.isfinite(z)):
            raise ValueError("Z contains NaN or inf")
    if len({z.shape[1] for z in seqs}) != 1:
        raise ValueError("all sequences must have the same number of dimensions")
    return seqs, single


def _hard_stats(labels_list, K):
    """One-hot posteriors and transition / start counts from hard labels
    (used to initialise EM from a K-means partition)."""
    gammas = [np.eye(K)[lab] for lab in labels_list]
    xi = np.zeros((K, K))
    start = np.zeros(K)
    for lab in labels_list:
        start[lab[0]] += 1
        np.add.at(xi, (lab[:-1], lab[1:]), 1.0)
    return gammas, xi, start


def _kmeans_labels(seqs, feats, K, seed, n_init=KMEANS_N_INIT):
    """K-means (fit on a random subsample, then assign every frame) on a list of
    per-sequence feature arrays; returns per-sequence label arrays.

    A converged K-means (best of ``n_init`` k-means++ seedings) matters: EM
    from a single k-means++ run was trapped in poor partitions on the NEMBA
    synthetic recordings (objective lower by ~5000 nats, ARI 0.11 vs 0.98)."""
    from sklearn.cluster import KMeans
    F = np.concatenate(feats, 0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), MAX_INIT_FRAMES, replace=False) if len(F) > MAX_INIT_FRAMES else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # sklearn MKL / threadpool notices on Windows
        km = KMeans(n_clusters=K, n_init=int(n_init), random_state=int(seed)).fit(
            F if idx is None else F[idx])
        lab = km.predict(F).astype(np.int64)
    out, i0 = [], 0
    for z in seqs:
        out.append(lab[i0:i0 + len(z)])
        i0 += len(z)
    return out


# --------------------------------------------------------------------------- #
# base class: EM machinery shared by the Gaussian HMM and the AR-HMM          #
# --------------------------------------------------------------------------- #

class _BaseHMM:
    """Shared MAP-EM machinery. Subclasses provide the emission model through
    ``_prepare``, ``_log_emission``, ``_init_labels``, ``_m_step_emission``,
    ``_log_prior_emission``, ``_n_emission_params`` and ``_emission_params``."""

    def __init__(self, n_states, sticky="auto", trans_pseudocount=TRANS_PSEUDOCOUNT,
                 min_covar=1e-3, n_restarts=3, max_iter=100, tol=1e-4, seed=0,
                 backend="auto"):
        """See the subclass docstrings for the meaning of every argument."""
        if int(n_states) < 1:
            raise ValueError("n_states must be >= 1")
        if not (sticky == "auto" or float(sticky) >= 0):
            raise ValueError("sticky must be 'auto' or a non-negative pseudo-count")
        self.n_states = int(n_states)
        self.sticky = sticky
        self.trans_pseudocount = float(trans_pseudocount)
        self.min_covar = float(min_covar)
        self.n_restarts = max(1, int(n_restarts))
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.backend = backend

    # ----- transition part of the model ----------------------------------- #
    def _prior_counts(self):
        """Dirichlet pseudo-counts of the transition rows (sticky diagonal)."""
        K = self.n_states
        return np.full((K, K), self.trans_pseudocount) + self.sticky_ * np.eye(K)

    def _m_step_trans(self, xi, start):
        """MAP transition matrix and start probabilities (mode of the Dirichlet
        posterior: counts plus pseudo-counts, normalised per row)."""
        C = xi + self._prior_counts()
        rs = C.sum(1, keepdims=True)
        K = self.n_states
        # a row with no counts and no prior (only if both pseudo-counts are 0): uniform
        self.transmat_ = np.where(rs > 0, C / np.where(rs > 0, rs, 1.0), 1.0 / K)
        s = start + self.trans_pseudocount
        self.startprob_ = s / s.sum() if s.sum() > 0 else np.full(K, 1.0 / K)

    def _log_prior_trans(self):
        """Unnormalised log Dirichlet prior of transmat_ and startprob_
        (sum of pseudo-count * log probability; 0 * log 0 counted as 0)."""
        P = self._prior_counts()
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(P > 0, P * np.log(self.transmat_), 0.0).sum()
            if self.trans_pseudocount > 0:
                lp += self.trans_pseudocount * np.log(self.startprob_).sum()
        return float(lp)

    # ----- EM ---------------------------------------------------------------- #
    def _e_step(self, prep):
        """Forward-backward on every sequence; returns (gammas, xi_sum,
        start_sum, total log-likelihood)."""
        K = self.n_states
        gammas, xi, start, ll = [], np.zeros((K, K)), np.zeros(K), 0.0
        for i in range(len(prep["seqs"])):
            g, x, l = forward_backward(self.startprob_, self.transmat_,
                                       self._log_emission(prep, i), self.backend)
            gammas.append(g)
            xi += x
            start += g[0]
            ll += l
        return gammas, xi, start, ll

    def _objective(self, ll):
        """EM objective: log-likelihood + log prior (what MAP-EM maximises)."""
        return ll + self._log_prior_trans() + self._log_prior_emission()

    def _run_em(self, prep, restart):
        """One EM run from a K-means-style initialisation; returns
        (objective, loglik, history, loglik_history, n_iter, converged, gammas)."""
        K = self.n_states
        labels = self._init_labels(prep, restart)
        gammas, xi, start = _hard_stats(labels, K)
        self._init_emission(prep, gammas)
        self._m_step_emission(prep, gammas)
        self._m_step_trans(xi, start)
        hist, ll_hist, converged, prev = [], [], False, -np.inf
        n_frames = prep["n_frames"]
        for it in range(self.max_iter):
            gammas, xi, start, ll = self._e_step(prep)
            obj = self._objective(ll)
            hist.append(obj)
            ll_hist.append(ll)
            # relative test: improvement per frame below tol -> converged
            if it > 0 and (obj - prev) / n_frames < self.tol:
                converged = True
                break
            prev = obj
            self._m_step_emission(prep, gammas)
            self._m_step_trans(xi, start)
        else:
            # max_iter reached right after an M-step: evaluate the final parameters
            gammas, xi, start, ll = self._e_step(prep)
            obj = self._objective(ll)
            hist.append(obj)
            ll_hist.append(ll)
        return obj, ll, hist, ll_hist, len(hist), converged, gammas

    def fit(self, Z):
        """Fit by MAP-EM, keeping the best of ``n_restarts`` initialisations.

        Z : [T, D] array, or a list of [T_i, D] arrays (independent sequences:
        no transition is assumed across their boundaries). Returns self."""
        t0 = time.perf_counter()
        seqs, _ = _as_list(Z)
        prep = self._prepare(seqs)
        T = prep["n_frames"]
        if T < 2 * self.n_states:
            raise ValueError(f"{T} frames are too few for {self.n_states} states")
        self.n_features_ = seqs[0].shape[1]
        self.sticky_ = (default_sticky(T, self.n_states) if self.sticky == "auto"
                        else float(self.sticky))
        best, self.restart_objectives_ = None, []
        for r in range(self.n_restarts):
            obj, ll, hist, ll_hist, n_it, conv, gammas = self._run_em(prep, r)
            self.restart_objectives_.append(obj)
            if best is None or obj > best[0]:
                best = (obj, ll, hist, ll_hist, n_it, conv, self._get_params(), gammas)
        obj, ll, hist, ll_hist, n_it, conv, params, gammas = best
        self._set_params(params)
        self.objective_, self.loglik_ = float(obj), float(ll)
        self.history_, self.loglik_history_ = np.asarray(hist), np.asarray(ll_hist)
        self.n_iter_, self.converged_ = n_it, conv
        self._finalise(prep, gammas)
        self.n_frames_ = T
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def _finalise(self, prep, gammas):
        """Hook run after the best restart is restored (default: nothing)."""

    def _get_params(self):
        """Copy of all fitted parameters (to keep the best restart)."""
        p = dict(startprob_=self.startprob_.copy(), transmat_=self.transmat_.copy())
        p.update({k: v.copy() for k, v in self._emission_params().items()})
        return p

    def _set_params(self, p):
        """Restore parameters saved by ``_get_params``."""
        for k, v in p.items():
            setattr(self, k, v)

    # ----- inference ----------------------------------------------------------- #
    def _check_fitted(self):
        """Raise if ``fit`` has not been called."""
        if not hasattr(self, "transmat_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet: call fit(Z) first")

    def predict(self, Z):
        """Most probable state path (Viterbi). Array in -> [T] int array;
        list in -> list of arrays."""
        self._check_fitted()
        seqs, single = _as_list(Z)
        prep = self._prepare(seqs, fitting=False)
        out = [viterbi(self.startprob_, self.transmat_, self._log_emission(prep, i),
                       self.backend)[0] for i in range(len(seqs))]
        return out[0] if single else out

    def predict_proba(self, Z):
        """Posterior state probabilities p(s_t = k | whole sequence), [T, K]
        (list in -> list out). Rows sum to 1."""
        self._check_fitted()
        seqs, single = _as_list(Z)
        prep = self._prepare(seqs, fitting=False)
        out = [forward_backward(self.startprob_, self.transmat_, self._log_emission(prep, i),
                                self.backend)[0] for i in range(len(seqs))]
        return out[0] if single else out

    def score(self, Z):
        """Total log-likelihood log p(Z) (summed over sequences; divide by the
        number of frames for a per-frame value)."""
        self._check_fitted()
        seqs, _ = _as_list(Z)
        prep = self._prepare(seqs, fitting=False)
        return float(sum(forward_loglik(self.startprob_, self.transmat_,
                                        self._log_emission(prep, i), self.backend)
                         for i in range(len(seqs))))

    def n_params(self):
        """Number of free parameters: (K - 1) start + K (K - 1) transitions +
        emission parameters."""
        self._check_fitted()
        K = self.n_states
        return (K - 1) + K * (K - 1) + self._n_emission_params()

    def bic(self, Z):
        """Bayesian information criterion -2 log p(Z) + n_params log(T) (lower is better)."""
        seqs, _ = _as_list(Z)
        T = sum(len(z) for z in seqs)
        return -2.0 * self.score(Z) + self.n_params() * np.log(T)


# --------------------------------------------------------------------------- #
# Gaussian HMM                                                                #
# --------------------------------------------------------------------------- #

class GaussianHMM(_BaseHMM):
    """Sticky hidden Markov model with diagonal Gaussian emissions.

    Parameters
    ----------
    n_states : number of hidden states K.
    sticky : "auto" or a non-negative float. Pseudo-count added to every
        self-transition of the Dirichlet prior on the transition rows. "auto"
        uses ``default_sticky(T, K)`` = STICKY_FRAC * T / K with T the number of
        training frames (resolved value in ``sticky_``). 0 disables it.
    trans_pseudocount : symmetric pseudo-count on every transition / start entry.
    min_covar : floor on every emission variance (data units; z-score Z first,
        as ``fit_states`` does, to make it scale free).
    n_restarts : EM runs from different K-means initialisations; the one with
        the highest objective (log-likelihood + log prior) is kept.
    max_iter, tol : EM stops when the objective gains less than ``tol`` nats per
        frame in one iteration, or after ``max_iter`` iterations.
    seed : random seed of the K-means initialisations.
    backend : "auto" | "numba" | "numpy" (see module docstring).

    Fitted attributes
    -----------------
    means_ [K, D], vars_ [K, D], transmat_ [K, K], startprob_ [K], sticky_,
    loglik_ (training log-likelihood), objective_, history_ (objective per EM
    iteration), loglik_history_, n_iter_, converged_, restart_objectives_,
    fit_seconds_.
    """

    def _prepare(self, seqs, fitting=True):
        """Cache the data, its concatenation and their squares (E- and M-step)."""
        Zc = np.concatenate(seqs, 0)
        Zc2 = Zc * Zc
        sq = [Zc2] if len(seqs) == 1 else [z * z for z in seqs]    # no copy for one sequence
        return dict(seqs=seqs, Zc=Zc, Zc2=Zc2, sq=sq, n_frames=len(Zc))

    def _log_emission(self, prep, i):
        """log N(z_t; mu_k, diag var_k) for every frame and state, [T, K]
        (quadratic form expanded into matrix products: O(T K D))."""
        Z = prep["seqs"][i]
        prec = 1.0 / self.vars_                                        # [K, D]
        const = -0.5 * (Z.shape[1] * _LOG2PI + np.log(self.vars_).sum(1))
        quad = prep["sq"][i] @ prec.T - 2.0 * Z @ (self.means_ * prec).T + (self.means_ ** 2 * prec).sum(1)
        return const[None, :] - 0.5 * quad

    def _init_labels(self, prep, restart):
        """K-means partition of the frames (a new K-means seed per restart)."""
        # restart 0: converged K-means; later restarts: single k-means++ runs (diversity)
        n_init = KMEANS_N_INIT if restart == 0 else 1
        return _kmeans_labels(prep["seqs"], prep["seqs"], self.n_states, self.seed + 1009 * restart, n_init)

    def _init_emission(self, prep, gammas):
        """Allocate emission parameters (overwritten by the first M-step; the
        global moments are kept for a state that receives no frame)."""
        K = self.n_states
        self.means_ = np.tile(prep["Zc"].mean(0), (K, 1))
        self.vars_ = np.tile(np.maximum(prep["Zc"].var(0), self.min_covar), (K, 1))

    def _m_step_emission(self, prep, gammas):
        """Posterior-weighted means and variances (variance floored at min_covar:
        the exact maximiser under that constraint, so EM stays monotone)."""
        G = np.concatenate(gammas, 0)                                  # [T, K]
        Zc = prep["Zc"]
        Nk = G.sum(0)
        ok = Nk > 1e-8                                                 # skip empty states
        if not ok.any():
            return
        Nok = Nk[ok][:, None]
        mu = (G[:, ok].T @ Zc) / Nok
        var = (G[:, ok].T @ prep["Zc2"]) / Nok - mu ** 2
        self.means_[ok] = mu
        self.vars_[ok] = np.maximum(var, self.min_covar)

    def _log_prior_emission(self):
        """Flat prior on means and variances (the floor is a constraint)."""
        return 0.0

    def _n_emission_params(self):
        """K means + K variances per dimension."""
        return 2 * self.n_states * self.n_features_

    def _emission_params(self):
        """Emission parameters to save / restore across restarts."""
        return dict(means_=self.means_, vars_=self.vars_)


# --------------------------------------------------------------------------- #
# autoregressive HMM                                                          #
# --------------------------------------------------------------------------- #

def _local_dynamics_features(seqs, window, n_pcs, seed):
    """Per-frame description of the *local* linear dynamics, for initialising
    the AR-HMM (K-means on positions cannot separate states that share a mean).

    Steps: project on the top ``n_pcs`` principal components u_t; in a sliding
    window of ``window`` frames estimate the local mean m, the local variances
    c0 and the local lag-1 cross-correlation matrix r_ij = cov(u_t,i, u_{t-1},j)
    / sqrt(c0_i c0_j). Rotation direction lives in the antisymmetric part of r,
    decay rate in its diagonal, so states differing only in dynamics separate.
    Returns a list of [T_i, p + p + p * p] standardised feature arrays."""
    Zc = np.concatenate(seqs, 0)
    mu = Zc.mean(0)
    rng = np.random.default_rng(seed)
    sub = Zc[rng.choice(len(Zc), MAX_INIT_FRAMES, replace=False)] if len(Zc) > MAX_INIT_FRAMES else Zc
    _, s, Vt = np.linalg.svd(sub - mu, full_matrices=False)
    p = int(min(n_pcs, Vt.shape[0]))
    V = Vt[:p].T / (s[:p] / np.sqrt(max(len(sub) - 1, 1)) + 1e-12)   # whitened PCs
    feats = []
    for z in seqs:
        u = (z - mu) @ V                                               # [T, p]
        T = len(u)
        lag1 = np.empty((T, p, p))
        lag1[1:] = u[1:, :, None] * u[:-1, None, :]                    # u_t,i * u_{t-1},j
        lag1[0] = lag1[1] if T > 1 else 0.0
        sm = lambda x: uniform_filter1d(x, size=int(window), axis=0, mode="nearest")  # noqa: E731
        m = sm(u)
        c0 = np.maximum(sm(u * u) - m * m, 1e-6)
        c1 = sm(lag1.reshape(T, p * p)).reshape(T, p, p) - m[:, :, None] * m[:, None, :]
        r = c1 / np.sqrt(c0[:, :, None] * c0[:, None, :])
        feats.append(np.concatenate([m, np.log(c0), r.reshape(T, p * p)], 1))
    F = np.concatenate(feats, 0)
    fm, fs = F.mean(0), F.std(0) + 1e-8
    return [(f - fm) / fs for f in feats]


class ARHMM(_BaseHMM):
    """Sticky autoregressive HMM: in state k, z_t = A_k z_{t-1} + b_k + e_t,
    e_t ~ N(0, diag(vars_k)). States are defined by dynamics (rotation,
    decay, drift) rather than by position, as in AR-HMM models of pose /
    behaviour (MoSeq; Wiltschko et al., Neuron 2015; Weinreb et al., 2024).

    Parameters
    ----------
    n_states, sticky, trans_pseudocount, min_covar, n_restarts, max_iter, tol,
    seed, backend : as in ``GaussianHMM`` (min_covar floors the innovation
        variance).
    ridge : precision of the zero-mean Gaussian prior on every AR coefficient
        (not on the intercept). The M-step is the corresponding weighted ridge
        regression (posterior weights), solved per output dimension with the
        current noise variance (an ECM step, so EM stays monotone).
    init_window : window (frames) of the local-dynamics features used for the
        even-numbered restarts; odd restarts use K-means on positions.
    init_pcs : principal components used by those features.

    The first frame of every sequence has no predecessor: it is scored by one
    Gaussian shared by all states (training mean and variance), so it carries
    no state evidence beyond the start probabilities.

    Fitted attributes
    -----------------
    ar_coefs_ [K, D, D] (A_k), ar_intercepts_ [K, D] (b_k), vars_ [K, D]
    (innovation variance), means_ [K, D] (posterior-weighted mean of the frames
    of each state, descriptive only), transmat_, startprob_, and the same
    diagnostics as ``GaussianHMM``.
    """

    def __init__(self, n_states, sticky="auto", trans_pseudocount=TRANS_PSEUDOCOUNT,
                 min_covar=1e-3, n_restarts=3, max_iter=100, tol=1e-4, seed=0,
                 backend="auto", ridge=1.0, init_window=20, init_pcs=4):
        """See the class docstring."""
        super().__init__(n_states, sticky=sticky, trans_pseudocount=trans_pseudocount,
                         min_covar=min_covar, n_restarts=n_restarts, max_iter=max_iter,
                         tol=tol, seed=seed, backend=backend)
        self.ridge = float(ridge)
        self.init_window = int(init_window)
        self.init_pcs = int(init_pcs)

    def _prepare(self, seqs, fitting=True):
        """Lagged pairs XY = [z_{t-1}, z_t] per sequence ([T_i - 1, 2D]) and their
        concatenation; fixes the shared first-frame Gaussian when fitting."""
        Zc = np.concatenate(seqs, 0)
        if fitting:
            self.init_mean_ = Zc.mean(0)
            self.init_var_ = np.maximum(Zc.var(0), self.min_covar)
        XY = [np.hstack([z[:-1], z[1:]]) for z in seqs]
        return dict(seqs=seqs, Zc=Zc, XY=XY, XYc=np.concatenate(XY, 0), n_frames=len(Zc))

    def _log_emission(self, prep, i):
        """[T, K] log emissions: shared Gaussian on frame 0, per-state AR
        residual Gaussian on frames 1..T-1.

        All states are predicted with one matrix product per chunk of frames
        (z_{t-1} @ [A_1' | ... | A_K']), which keeps BLAS efficient and memory
        bounded (about 32 MB per temporary)."""
        z, XY = prep["seqs"][i], prep["XY"][i]
        T, D = z.shape
        K = self.n_states
        out = np.empty((T, K))
        d0 = z[0] - self.init_mean_
        out[0] = -0.5 * (D * _LOG2PI + np.log(self.init_var_).sum() + (d0 ** 2 / self.init_var_).sum())
        if T > 1:
            Wcat = self.weights_[:, :D, :].transpose(1, 0, 2).reshape(D, K * D)  # block k = A_k'
            b = self.weights_[:, D, :][None]                           # [1, K, D]
            iv = (1.0 / self.vars_)[None]                              # [1, K, D]
            const = -0.5 * (D * _LOG2PI + np.log(self.vars_).sum(1))   # [K]
            step = max(256, int(4e6 // (K * D)))
            for s in range(0, T - 1, step):
                e = min(T - 1, s + step)
                R = XY[s:e, None, D:] - (XY[s:e, :D] @ Wcat).reshape(e - s, K, D) - b
                R *= R
                R *= iv
                out[1 + s:1 + e] = const[None] - 0.5 * R.sum(2)
        return out

    def _init_labels(self, prep, restart):
        """Even restarts: K-means on local-dynamics features; odd restarts:
        K-means on positions (so the AR-HMM never starts worse than a
        position-based partition). Restarts 0 and 1 use a converged K-means,
        later ones single k-means++ runs (diversity)."""
        seed = self.seed + 1009 * restart
        if restart % 2 == 0:
            feats = _local_dynamics_features(prep["seqs"], self.init_window, self.init_pcs, seed)
        else:
            feats = prep["seqs"]
        n_init = KMEANS_N_INIT if restart < 2 else 1
        return _kmeans_labels(prep["seqs"], feats, self.n_states, seed, n_init)

    def _init_emission(self, prep, gammas):
        """Start every state from a random-walk-free model (A = 0, b = global
        mean, var = global variance); the first M-step overwrites it."""
        K, D = self.n_states, prep["Zc"].shape[1]
        self.weights_ = np.zeros((K, D + 1, D))
        self.weights_[:, D, :] = prep["Zc"].mean(0)
        self.vars_ = np.tile(np.maximum(prep["Zc"].var(0), self.min_covar), (K, 1))

    def _m_step_emission(self, prep, gammas):
        """Weighted ridge regression per state, then the residual variance (ECM:
        coefficients given the current variance, then the variance given the
        new coefficients; both steps exact, so EM stays monotone).

        For output dimension d the MAP coefficients under N(0, I / ridge) solve
        (Sxx + c_d I) a_d = Sxy[:, d] with c_d = var_kd * ridge, where Sxx, Sxy
        are posterior-weighted *centred* scatter matrices (the unpenalised
        intercept is profiled out: b = mean(y) - A mean(x)). One eigh of Sxx
        solves all D systems at once; it also avoids LAPACK solve on tiny
        matrices, which multi-threaded OpenBLAS can make ~1000x slower."""
        XY = prep["XYc"]
        if len(XY) == 0:
            return
        G = np.concatenate([g[1:] for g in gammas], 0)                 # frames with a predecessor
        D = XY.shape[1] // 2
        for k in range(self.n_states):
            g = G[:, k]
            Nk = g.sum()
            if Nk < 1e-8:                                              # empty state: keep it as is
                continue
            m = (g @ XY) / Nk                                          # weighted means [x_mean, y_mean]
            C = XY - m                                                 # centred pairs [N, 2D]
            S = (C[:, :D] * g[:, None]).T @ C                          # [D, 2D] = [Sxx | Sxy]
            Sxx, Sxy = S[:, :D], S[:, D:]
            syy = g @ (C[:, D:] ** 2)                                  # [D] weighted sum of squares of y
            lam, U = np.linalg.eigh(Sxx)
            lam = np.maximum(lam, 0.0)                                 # PSD up to round-off
            c = self.vars_[k] * self.ridge                             # [D] ridge per output dim
            den = lam[:, None] + c[None, :]
            den = np.maximum(den, 1e-12 * max(float(lam.max()), 1.0))  # ridge = 0 and collinear X
            A = U @ ((U.T @ Sxy) / den)                                # [D_in, D_out]: y = x @ A + b
            b = m[D:] - m[:D] @ A
            # weighted residual sum of squares from the scatter matrices (no pass over the frames)
            rss = syy - 2.0 * (A * Sxy).sum(0) + (A * (Sxx @ A)).sum(0)
            self.weights_[k, :D] = A
            self.weights_[k, D] = b
            self.vars_[k] = np.maximum(np.maximum(rss, 0.0) / Nk, self.min_covar)

    def _log_prior_emission(self):
        """log N(A_k; 0, I / ridge) summed over states, up to a constant."""
        D = self.weights_.shape[2]
        return float(-0.5 * self.ridge * (self.weights_[:, :D, :] ** 2).sum())

    def _n_emission_params(self):
        """Per state: D x D coefficients, D intercepts, D variances."""
        D = self.n_features_
        return self.n_states * (D * D + 2 * D)

    def _emission_params(self):
        """Emission parameters to save / restore across restarts."""
        return dict(weights_=self.weights_, vars_=self.vars_)

    def _finalise(self, prep, gammas):
        """Descriptive per-state means of the frames (posterior weighted)."""
        G = np.concatenate(gammas, 0)
        self.means_ = (G.T @ prep["Zc"]) / np.maximum(G.sum(0), 1e-12)[:, None]

    @property
    def ar_coefs_(self):
        """[K, D, D] matrices A_k with z_t = A_k z_{t-1} + b_k + noise."""
        D = self.weights_.shape[2]
        return np.transpose(self.weights_[:, :D, :], (0, 2, 1))

    @property
    def ar_intercepts_(self):
        """[K, D] intercepts b_k."""
        return self.weights_[:, -1, :]


# --------------------------------------------------------------------------- #
# model selection and the one-call entry point                                #
# --------------------------------------------------------------------------- #

_MODELS = {"hmm": GaussianHMM, "gaussian": GaussianHMM, "arhmm": ARHMM, "ar": ARHMM}


def make_model(method, n_states, **kw):
    """Instantiate a state model: method "hmm" (GaussianHMM) or "arhmm" (ARHMM)."""
    try:
        cls = _MODELS[str(method).lower()]
    except KeyError:
        raise ValueError(f"unknown method {method!r}: use 'hmm' or 'arhmm'") from None
    return cls(n_states, **kw)


def _complement_runs(T, lo, hi):
    """Contiguous training pieces of range(T) once the block [lo, hi) is held out."""
    return [(a, b) for a, b in ((0, lo), (hi, T)) if b > a]


def _select_k(Z, method="hmm", k_range=(2, 10), n_folds=3, seed=0, se_rule="fold",
              log=None, **model_kw):
    """Implementation of ``select_k``; also returns the full-data models {k: model}."""
    import pandas as pd
    seqs, _ = _as_list(Z)
    ks = list(range(int(k_range[0]), int(k_range[1]) + 1))
    if not ks or ks[0] < 1:
        raise ValueError("k_range must be (k_min, k_max) with 1 <= k_min <= k_max")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if se_rule not in ("fold", "paired"):
        raise ValueError("se_rule must be 'fold' or 'paired'")
    # contiguous blocks: fold f holds out block f of every sequence
    bounds = [np.linspace(0, len(z), n_folds + 1).astype(int) for z in seqs]
    if min(int(np.diff(b).min()) for b in bounds) < 2 * ks[-1]:
        raise ValueError("sequences too short for this many folds / states")
    T_all = sum(len(z) for z in seqs)
    ll_folds = np.empty((len(ks), n_folds))
    rows, models = [], {}
    for ik, k in enumerate(ks):
        t0 = time.perf_counter()
        for f in range(n_folds):
            train, test = [], []
            for z, b in zip(seqs, bounds):
                lo, hi = b[f], b[f + 1]
                test.append(z[lo:hi])
                train += [z[a:c] for a, c in _complement_runs(len(z), lo, hi)]
            m = make_model(method, k, seed=seed, **model_kw).fit(train)
            ll_folds[ik, f] = m.score(test) / sum(len(t) for t in test)
        full = make_model(method, k, seed=seed, **model_kw).fit(seqs)
        models[k] = full
        rows.append(dict(k=k, heldout_ll_mean=ll_folds[ik].mean(),
                         heldout_ll_se=ll_folds[ik].std(ddof=1) / np.sqrt(n_folds),
                         bic=-2.0 * full.loglik_ + full.n_params() * np.log(T_all),
                         loglik=full.loglik_, n_params=full.n_params(),
                         fit_s=time.perf_counter() - t0))
        if log:
            r = rows[-1]
            log(f"  [{method}] K={k}: held-out LL/frame {r['heldout_ll_mean']:.4f} "
                f"+/- {r['heldout_ll_se']:.4f}, BIC {r['bic']:.1f} ({r['fit_s']:.1f}s)")
    df = pd.DataFrame(rows)
    ib = int(df["heldout_ll_mean"].values.argmax())
    # SE of the fold-wise difference to the best K (paired: same held-out blocks)
    diff = ll_folds - ll_folds[ib][None, :]
    df["paired_se"] = diff.std(1, ddof=1) / np.sqrt(n_folds)
    if se_rule == "fold":     # classic 1-SE rule: within one SE of the best model's mean
        ok = df["heldout_ll_mean"].values >= df["heldout_ll_mean"].values[ib] - df["heldout_ll_se"].values[ib]
    else:                     # paired variant: mean difference within one SE of that difference
        ok = df["heldout_ll_mean"].values >= df["heldout_ll_mean"].values[ib] - df["paired_se"].values
    best_k = int(df["k"].values[np.flatnonzero(ok)[0]])
    df["selected"] = df["k"] == best_k
    df.attrs.update(best_k=best_k, best_k_max_ll=int(df["k"].values[ib]),
                    best_k_bic=int(df["k"].values[df["bic"].values.argmin()]),
                    se_rule=se_rule, method=method, n_folds=n_folds,
                    at_upper_bound=bool(best_k == ks[-1]))
    if log:
        log(f"  [{method}] 1-SE rule ({se_rule}) -> K={best_k} (max held-out LL at "
            f"K={df.attrs['best_k_max_ll']}, min BIC at K={df.attrs['best_k_bic']})")
    if best_k == ks[-1]:
        # held-out likelihood still rising at k_max: typical of a continuous
        # manifold (a ring is modelled better by ever more Gaussians), where
        # "number of states" is not identifiable from density alone
        msg = (f"select_k: K reached the upper bound k_max={ks[-1]}; the held-out likelihood "
               "keeps rising, typical of a continuous manifold. Prefer a fixed K.")
        warnings.warn(msg)
        if log:
            log(f"  WARNING: {msg}")
    return best_k, df, models


def select_k(Z, method="hmm", k_range=(2, 10), n_folds=3, seed=0, se_rule="fold",
             log=None, **model_kw):
    """Choose the number of states by blocked cross-validation.

    The recording is cut into ``n_folds`` contiguous blocks (each sequence of a
    list is cut the same way). For every K and fold, the model is fitted on the
    remaining contiguous pieces (as separate sequences, so no fake transition
    joins them) and the held-out block is scored (log-likelihood per frame).
    BIC of a fit on all data is reported alongside.

    Rule (1-SE): the smallest K whose mean held-out log-likelihood per frame is
    within one standard error of the best mean. ``se_rule="fold"`` (default)
    uses the standard error across folds of the best K (classic rule; with few
    heterogeneous blocks it is conservative); ``se_rule="paired"`` uses the
    standard error of the fold-wise difference to the best K (more sensitive).

    Returns
    -------
    best_k : int
    table : pandas DataFrame with columns k, heldout_ll_mean, heldout_ll_se, bic
        (plus loglik, n_params, fit_s, paired_se, selected); ``table.attrs``
        holds best_k_bic, best_k_max_ll and at_upper_bound.

    Caveat: on a continuous manifold (e.g. a ring) the held-out likelihood of a
    Gaussian HMM keeps rising with K, exactly like the silhouette does, so the
    selected K hits ``k_range[1]`` (a warning is issued): the data then have no
    natural number of states and a fixed K should be used.
    """
    best_k, df, _ = _select_k(Z, method, k_range, n_folds, seed, se_rule, log, **model_kw)
    return best_k, df


def fit_states(Z, method="hmm", k=None, k_range=(2, 10), sticky="auto", seed=0, log=None,
               standardize=True, n_folds=3, se_rule="fold", **kw):
    """Fit a temporal state model to an embedding and decode its state sequence.

    Input: prefer the *unsmoothed* embedding. Temporal smoothing is a temporal
    prior that the HMM already provides through its transition matrix; on
    Gaussian-smoothed features (sigma 4 frames) the AR-HMM mostly models the
    smoothing kernel and loses the states (NEMBA synthetic, 5 seeds: ARI 0.36 smoothed
    vs 1.00 unsmoothed, benchmarks/validate_states.py), and the Gaussian HMM
    loses a little (0.97 vs 1.00).

    Parameters
    ----------
    Z : [T, E] embedding (or a list of such arrays, one per independent sequence).
    method : "hmm" (sticky Gaussian HMM: states = regions of the embedding) or
        "arhmm" (sticky AR-HMM: states = local linear dynamics).
    k : number of states; None selects it with ``select_k`` over ``k_range``.
    sticky : self-transition pseudo-count; "auto" (default) = STICKY_FRAC * T / K,
        a prior worth 10 % of the frames an average state occupies: it shrinks
        the estimated switching rate by 1/1.1 for an average state (more for
        rare states), enough to break the self-reinforcing high-switching EM
        fixed point on noisy data while costing a genuine, well-supported
        transition only log(1.1) ~ 0.1 nat. 0 disables it.
    seed : seed of the initialisations.
    log : optional callable (e.g. ``print``) for progress messages.
    standardize : z-score every dimension first (the likelihood is invariant to
        it, but ``min_covar`` and the AR ridge then become scale free).
    n_folds, se_rule : passed to ``select_k`` when k is None.
    **kw : other model arguments (n_restarts, max_iter, tol, min_covar,
        trans_pseudocount, backend, ridge, init_window, ...).

    Returns
    -------
    dict(labels [T] (Viterbi), posteriors [T, K], model, k, selection (DataFrame
    or None), loglik (training log-likelihood), method, scaler (mean, sd) used
    for standardisation or None, fit_seconds). For list input, labels and
    posteriors are lists.
    """
    t0 = time.perf_counter()
    seqs, single = _as_list(Z)
    scaler = None
    if standardize:
        Zc = np.concatenate(seqs, 0)
        mu, sd = Zc.mean(0), Zc.std(0)
        sd = np.where(sd > 0, sd, 1.0)                       # constant columns stay at 0
        seqs = [(z - mu) / sd for z in seqs]
        scaler = (mu, sd)
    data = seqs[0] if single else seqs
    method = str(method).lower()
    if k is None:
        k, selection, models = _select_k(data, method, k_range, n_folds, seed, se_rule,
                                         log, sticky=sticky, **kw)
        model = models[k]
    else:
        selection = None
        model = make_model(method, int(k), sticky=sticky, seed=seed, **kw).fit(data)
    labels = model.predict(data)
    posteriors = model.predict_proba(data)
    if log:
        sw = n_switches(labels) if single else sum(n_switches(x) for x in labels)
        log(f"  [{method}] K={k}: log-likelihood {model.loglik_:.1f}, {sw} switches, "
            f"{model.n_iter_} EM iterations, kappa={model.sticky_:.1f} "
            f"({time.perf_counter() - t0:.1f}s)")
    return dict(labels=labels, posteriors=posteriors, model=model, k=int(k),
                selection=selection, loglik=float(model.loglik_), method=method,
                scaler=scaler, fit_seconds=time.perf_counter() - t0)
