"""Gaussian-process factor analysis (GPFA), native numpy / scipy implementation.

Model (Yu, Cunningham, Santhanam, Ryu, Shenoy & Sahani, J Neurophysiol 2009,
doi 10.1152/jn.90941.2008)
----------------------------------------------------------------------------
    y_t = C x_t + d + e_t,          e_t ~ N(0, diag(R))
    x_i(.) ~ GP(0, k_i),            k_i(t, s) = (1 - s2) exp(-(t - s)^2 / (2 tau_i^2))
                                                 + s2 * delta(t, s)

``y_t`` [q] is the observed population vector, ``x_t`` [p] the latent state,
each latent dimension ``i`` has its own squared-exponential timescale ``tau_i``
(in frames) and a fixed GP noise variance ``s2`` (1e-3 as in the paper), so the
prior variance of every latent is exactly 1 and ``C`` carries the scale.

Continuous recordings
---------------------
GPFA was written for trials. A continuous recording is cut into pseudo-trials
of ``seg_len`` frames that are treated as independent draws of the model. This
is a composite likelihood (correlations across a cut are ignored): parameters
remain consistent as long as ``seg_len`` is several times the longest
timescale (rule of thumb: ``seg_len >= 5 * tau_max``).

Fitting (EM)
------------
* E-step: exact Gaussian posterior of every segment. With the latents stacked
  latent-major, x = [x_1(1..T); ...; x_p(1..T)], the posterior precision is
  ``M = blkdiag(K_1^-1..K_p^-1) + kron(C' R^-1 C, I_T)`` and the mean is
  ``M^-1 C' R^-1 (y - d)``. ``M`` depends only on the segment length, so all
  segments of the same length share one Cholesky factorisation / inverse.
* Marginal log-likelihood by the matrix determinant lemma and Woodbury
  identity (never forms the qT x qT covariance).
* M-step: closed form for [C d] and R (with a variance floor); timescales by
  L-BFGS ascent on the expected complete-data log-likelihood in log(tau).
  Each partial step can only increase the objective (generalised EM), so the
  log-likelihood is non-decreasing up to numerical precision (and up to the
  variance floor, if it becomes active).
* Initialisation: factor analysis (scikit-learn), then an AMUSE-style rotation
  (whitening + eigenvectors of a symmetrised lagged covariance) so the initial
  latents already separate by timescale, which GP priors make identifiable.

Inference on the whole recording
--------------------------------
``transform`` computes posterior means on overlapping windows (hop = half a
window) and blends them with a Hann taper, so there are no seams at window
borders; edge frames, where a window has the least context, get little weight.
The GP posterior is a smoother (it uses past and future frames): it is acausal.
"""

from __future__ import annotations

import contextlib
import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import lapack
from scipy.optimize import minimize

from ._embtcn import NeuralEmbTCNOutput

LOG_2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
# BLAS thread control                                                          #
# --------------------------------------------------------------------------- #
# Multi-threaded BLAS on 100 x 100 matrices is pure overhead, and on a busy CPU
# it collapses (measured: a batched 8 x 100 x 100 inverse took 880 ms with 16
# OpenBLAS threads on a loaded laptop versus 1.4 ms with one thread). Small
# per-latent algebra therefore always runs single-threaded; only the large
# (p * seg_len)^2 posterior factorisation uses the ``n_threads`` setting.
_CONTROLLER = None


def _blas_threads(n):
    """Context manager limiting BLAS threads to ``n`` (``None`` = unchanged)."""
    global _CONTROLLER
    if n is None:
        return contextlib.nullcontext()
    try:
        if _CONTROLLER is None:
            from threadpoolctl import ThreadpoolController     # sklearn dependency
            _CONTROLLER = ThreadpoolController()
        return _CONTROLLER.limit(limits=int(n), user_api="blas")
    except Exception:  # noqa: BLE001  (threadpoolctl missing / too old: run unchanged)
        return contextlib.nullcontext()


# --------------------------------------------------------------------------- #
# small linear-algebra helpers                                                 #
# --------------------------------------------------------------------------- #

def _chol_inv_logdet(M: np.ndarray) -> tuple[np.ndarray, float]:
    """Inverse and log-determinant of a symmetric positive-definite matrix.

    Uses LAPACK ``potrf`` (Cholesky) + ``potri`` (inverse from the factor),
    which is about three times cheaper than solving against the identity.
    Raises ``np.linalg.LinAlgError`` if ``M`` is not positive definite."""
    L, info = lapack.dpotrf(M, lower=1, clean=1)
    if info != 0:
        raise np.linalg.LinAlgError(f"Cholesky failed (info={info}); matrix not positive definite")
    logdet = 2.0 * float(np.log(np.diag(L)).sum())
    Minv, info = lapack.dpotri(L, lower=1)
    if info != 0:
        raise np.linalg.LinAlgError(f"potri failed (info={info})")
    # potri only fills the lower triangle: mirror it to get the full inverse
    Minv = np.tril(Minv) + np.tril(Minv, -1).T
    return Minv, logdet


def se_kernel(T: int, tau: float, noise_var: float = 1e-3) -> np.ndarray:
    """GPFA squared-exponential covariance [T, T] for one latent.

    ``k(t, s) = (1 - noise_var) exp(-(t - s)^2 / (2 tau^2)) + noise_var delta``
    so the marginal variance is exactly 1 (Yu et al. 2009)."""
    t = np.arange(T, dtype=np.float64)
    D2 = (t[:, None] - t[None, :]) ** 2
    return (1.0 - noise_var) * np.exp(-D2 / (2.0 * tau ** 2)) + noise_var * np.eye(T)


def _segment_bounds(T: int, seg_len: int) -> list[tuple[int, int]]:
    """Non-overlapping pseudo-trials [start, stop) covering 0..T exactly once.

    The last segment keeps the remainder (it may be shorter); segments are
    grouped by length later so every distinct length is factorised once."""
    L = int(max(1, min(seg_len, T)))
    return [(s, min(s + L, T)) for s in range(0, T, L)]


def _window_starts(T: int, L: int) -> list[int]:
    """Starts of overlapping windows of length L (hop L // 2), last one flush
    with the end of the recording, so every frame is covered."""
    if T <= L:
        return [0]
    hop = max(1, L // 2)
    starts = list(range(0, T - L + 1, hop))
    if starts[-1] != T - L:
        starts.append(T - L)
    return starts


def _autocorr_tau(z: np.ndarray, max_lag: int) -> float:
    """Timescale estimate of a 1-D series for an SE kernel.

    The SE autocorrelation exp(-lag^2 / (2 tau^2)) equals exp(-1/2) at
    lag = tau, so tau is the (linearly interpolated) first lag at which the
    empirical autocorrelation falls below exp(-1/2)."""
    z = np.asarray(z, float) - np.mean(z)
    n = len(z)
    if n < 4 or z.std() == 0:
        return 1.0
    f = np.fft.rfft(z, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: min(n, max_lag + 1)]
    ac = ac / ac[0]
    thr = np.exp(-0.5)
    below = np.flatnonzero(ac < thr)
    if not below.size:
        return float(max_lag)
    k = int(below[0])
    # linear interpolation between lag k-1 (above threshold) and lag k (below)
    a0, a1 = ac[k - 1], ac[k]
    return float((k - 1) + (a0 - thr) / max(a0 - a1, 1e-12))


# --------------------------------------------------------------------------- #
# the estimator (pure numpy, independent of torch)                             #
# --------------------------------------------------------------------------- #

class GPFA:
    """Gaussian-process factor analysis fitted by EM on a continuous recording.

    Parameters
    ----------
    n_latents    : number of latent dimensions p.
    seg_len      : pseudo-trial length in frames (fit) and inference window
                   length (transform). Should be >= ~5x the longest timescale.
    max_iter     : maximum number of EM iterations.
    tol          : convergence when the last LL improvement is smaller than
                   ``tol`` times the total improvement since the first
                   iteration (the criterion of Yu et al.'s reference code).
    noise_var    : fixed GP noise variance s2 (1e-3 in the paper).
    min_var_frac : floor on each private variance R_j, as a fraction of the
                   channel's variance (avoids Heywood cases, as in the paper).
    tau_iters    : L-BFGS iterations for the timescales in each M-step.
    tau_bounds   : (min, max) timescale in frames; max defaults to 2 * seg_len.
    seed         : seed of the factor-analysis initialisation.
    n_threads    : BLAS threads for the (p * seg_len)^2 posterior factorisation
                   (None = library default). Small per-latent algebra always
                   runs on one thread (faster, see ``_blas_threads``).
    verbose      : print the log-likelihood every 10 iterations.

    Fitted attributes
    -----------------
    C_ [q, p], d_ [q], R_ [q], timescales_ [p] (frames), log_likelihoods_
    (list, one value per E-step), n_iter_, converged_, fit_seconds_.
    """

    def __init__(self, n_latents: int = 8, *, seg_len: int = 100, max_iter: int = 100,
                 tol: float = 1e-4, noise_var: float = 1e-3, min_var_frac: float = 0.01,
                 tau_iters: int = 10, tau_bounds: tuple | None = None, seed: int = 0,
                 n_threads: int | None = None, verbose: bool = False) -> None:
        """Store the hyper-parameters (see the class docstring)."""
        self.n_latents = int(n_latents)
        self.seg_len = int(seg_len)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.noise_var = float(noise_var)
        self.min_var_frac = float(min_var_frac)
        self.tau_iters = int(tau_iters)
        self.tau_bounds = tau_bounds
        self.seed = int(seed)
        self.n_threads = n_threads
        self.verbose = bool(verbose)
        self._cache: dict = {}                      # segment length -> (Minv, logdets)
        self._d2: dict = {}                         # segment length -> squared lags [L, L]

    # ----------------------------------------------------------------- params
    def set_params(self, C, d, R, timescales) -> "GPFA":
        """Load model parameters (e.g. restored from a saved state)."""
        self.C_ = np.asarray(C, np.float64)
        self.d_ = np.asarray(d, np.float64).ravel()
        self.R_ = np.asarray(R, np.float64).ravel()
        self.timescales_ = np.asarray(timescales, np.float64).ravel()
        self.n_latents = int(self.C_.shape[1])
        self._cache = {}                            # parameters changed: drop factorisations
        return self

    def _check_fitted(self) -> None:
        """Raise if the model has no parameters yet."""
        if not hasattr(self, "C_"):
            raise RuntimeError("GPFA is not fitted; call fit(X) first")

    def _tau_limits(self) -> tuple[float, float]:
        """Allowed timescale range in frames (log-space box for L-BFGS-B)."""
        lo, hi = (0.5, 2.0 * self.seg_len) if self.tau_bounds is None else self.tau_bounds
        return float(lo), float(hi)

    # ------------------------------------------------------------ kernels
    def _sq_lags(self, T: int) -> np.ndarray:
        """Squared time lags (t - s)^2 [T, T], cached per segment length."""
        if T not in self._d2:
            t = np.arange(T, dtype=np.float64)
            self._d2[T] = (t[:, None] - t[None, :]) ** 2
        return self._d2[T]

    def _signal_kernels(self, T: int, tau: np.ndarray) -> np.ndarray:
        """Signal part (1 - s2) exp(-D^2 / 2 tau^2) of every kernel, [p, T, T]."""
        D2 = self._sq_lags(T)
        return (1.0 - self.noise_var) * np.exp(-D2[None] / (2.0 * tau[:, None, None] ** 2))

    def _posterior_factor(self, T: int) -> tuple[np.ndarray, float, float]:
        """Posterior covariance M^-1 [pT, pT] shared by every segment of length T,
        plus log|K_bar| and log|M| (cached until the parameters change)."""
        if T in self._cache:
            return self._cache[T]
        p = self.n_latents
        A = (self.C_.T / self.R_) @ self.C_                        # C' R^-1 C, [p, p]
        # latent-major layout: M[(i,t),(j,s)] = A_ij delta_ts + delta_ij Kinv_i[t,s]
        M = np.kron(A, np.eye(T))
        logdetK = 0.0
        with _blas_threads(1):                                     # tiny T x T blocks
            Ksig = self._signal_kernels(T, self.timescales_)
            eye = self.noise_var * np.eye(T)
            for i in range(p):
                Kinv_i, ld = _chol_inv_logdet(Ksig[i] + eye)
                logdetK += ld
                M[i * T:(i + 1) * T, i * T:(i + 1) * T] += Kinv_i
        with _blas_threads(self.n_threads):                        # the one large factorisation
            Minv, logdetM = _chol_inv_logdet(M)
        self._cache[T] = (Minv, logdetK, logdetM)
        return self._cache[T]

    # ------------------------------------------------------------ E-step
    def _estep(self, Y: np.ndarray, groups: dict) -> tuple[float, dict]:
        """Exact posterior of every pseudo-trial + sufficient statistics.

        Y      : [q, T] data.
        groups : {segment length: index array [n_seg, length]} covering every
                 frame exactly once.
        Returns (log-likelihood, stats) where stats holds the posterior means
        Xhat [p, T], sum_t E[x_t x_t'] [p, p] and, per segment length, the
        summed second moments S [p, L, L] used by the timescale update."""
        p, (q, T) = self.n_latents, Y.shape
        self._cache = {}                                            # params changed since last call
        Yc = Y - self.d_[:, None]
        Bfull = (self.C_ / self.R_[:, None]).T @ Yc                # C' R^-1 (y - d), [p, T]
        quad = float(np.sum(Yc ** 2 / self.R_[:, None]))
        sum_logR = float(np.log(self.R_).sum())
        Xhat = np.zeros((p, T))
        Exx = np.zeros((p, p))
        S_by_len = {}
        ll = -0.5 * quad
        for L, idx in groups.items():
            n_seg = idx.shape[0]
            Minv, logdetK, logdetM = self._posterior_factor(L)
            # gather b for every segment, latent-major: [p, n_seg, L] -> [p*L, n_seg]
            Bg = Bfull[:, idx].transpose(0, 2, 1).reshape(p * L, n_seg)
            Mu = Minv @ Bg                                          # posterior means
            ll += -0.5 * n_seg * (q * L * LOG_2PI + L * sum_logR + logdetK + logdetM)
            ll += 0.5 * float(np.sum(Bg * Mu))                      # + b' M^-1 b
            Mu3 = Mu.reshape(p, L, n_seg)
            Xhat[:, idx] = Mu3.transpose(0, 2, 1)                   # scatter back to frames
            M4 = Minv.reshape(p, L, p, L)
            # sum over the L frames of the per-frame posterior covariance, times n_seg
            Exx += n_seg * np.einsum("itjt->ij", M4)
            # per-latent summed second moments for the GP timescale update
            S = n_seg * np.stack([M4[i, :, i, :] for i in range(p)])
            S += np.einsum("itn,isn->its", Mu3, Mu3)
            S_by_len[L] = (S, n_seg)
        Exx += Xhat @ Xhat.T
        return float(ll), dict(Xhat=Xhat, Exx=Exx, S=S_by_len)

    # ------------------------------------------------------------ M-step
    def _mstep_obs(self, Y: np.ndarray, st: dict, var_floor: np.ndarray) -> None:
        """Closed-form update of [C d] and diag(R) from the E-step moments."""
        p, (q, T) = self.n_latents, Y.shape
        Xhat = st["Xhat"]
        Ex = Xhat.sum(1)
        # augmented moments of x_tilde = [x; 1]
        Sxx = np.zeros((p + 1, p + 1))
        Sxx[:p, :p] = st["Exx"]
        Sxx[:p, p] = Sxx[p, :p] = Ex
        Sxx[p, p] = T
        Syx = np.concatenate([Y @ Xhat.T, Y.sum(1, keepdims=True)], axis=1)   # [q, p+1]
        Cd = np.linalg.solve(Sxx, Syx.T).T                                     # Syx Sxx^-1
        self.C_, self.d_ = Cd[:, :p], Cd[:, p]
        R = (np.sum(Y ** 2, axis=1) - np.sum(Cd * Syx, axis=1)) / T
        self.R_ = np.maximum(R, var_floor)

    def _tau_objective(self, log_tau: np.ndarray, S_by_len: dict) -> tuple[np.ndarray, np.ndarray]:
        """Expected GP log-prior per latent and its gradient in log(tau).

        f_i = sum_segments [-1/2 log|K_i| - 1/2 tr(K_i^-1 E[x_i x_i'])]
        df_i/dlog(tau_i) = 1/2 tr((K^-1 S K^-1 - n K^-1) dK/dlog(tau)),
        dK/dlog(tau) = (1 - s2) exp(-D^2 / 2 tau^2) * D^2 / tau^2."""
        tau = np.exp(log_tau)
        f = np.zeros_like(tau)
        g = np.zeros_like(tau)
        for L, (S, n_seg) in S_by_len.items():
            D2 = self._sq_lags(L)
            Ksig = self._signal_kernels(L, tau)
            eye = self.noise_var * np.eye(L)
            for i in range(len(tau)):
                Kinv, logdet = _chol_inv_logdet(Ksig[i] + eye)
                KiS = Kinv @ S[i]
                f[i] += -0.5 * n_seg * logdet - 0.5 * np.trace(KiS)
                G = KiS @ Kinv - n_seg * Kinv                       # K^-1 S K^-1 - n K^-1
                # dK / dlog(tau) = Ksig * D^2 / tau^2 (elementwise)
                g[i] += 0.5 * np.sum(G * Ksig[i] * D2) / tau[i] ** 2
        return f, g

    def _mstep_tau(self, S_by_len: dict) -> None:
        """Gradient ascent (L-BFGS-B, log parameterisation) on the timescales.

        The objective is separable across latents, so one joint L-BFGS run is
        used and any latent whose objective did not improve keeps its old value
        (this guarantees the generalised-EM monotonicity)."""
        lo, hi = self._tau_limits()
        x0 = np.log(np.clip(self.timescales_, lo, hi))

        def neg(x):
            """Negative summed objective and gradient for scipy."""
            f, g = self._tau_objective(x, S_by_len)
            return -float(f.sum()), -g

        with _blas_threads(1):                                     # only L x L matrices here
            f0, _ = self._tau_objective(x0, S_by_len)
            res = minimize(neg, x0, jac=True, method="L-BFGS-B",
                           bounds=[(np.log(lo), np.log(hi))] * len(x0),
                           options=dict(maxiter=self.tau_iters))
            f1, _ = self._tau_objective(res.x, S_by_len)
        x_new = np.where(f1 >= f0, res.x, x0)                       # keep improvements only
        self.timescales_ = np.exp(x_new)

    # ------------------------------------------------------------ init
    def _initialise(self, Y: np.ndarray) -> None:
        """Factor analysis + AMUSE rotation + autocorrelation timescales."""
        from sklearn.decomposition import FactorAnalysis
        p, (q, T) = self.n_latents, Y.shape
        fa = FactorAnalysis(n_components=p, random_state=self.seed).fit(Y.T)
        C_fa = fa.components_.T                                     # [q, p]
        z = fa.transform(Y.T)                                       # FA posterior means [T, p]
        z = z - z.mean(0)
        # whiten the factors, then rotate them onto the eigenvectors of a sum of
        # symmetrised lagged covariances (AMUSE / SOBI-light): sources with
        # different autocorrelations become separated, as GPFA's prior requires
        evals, evecs = np.linalg.eigh(np.cov(z.T).reshape(p, p))
        W = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-12))) @ evecs.T
        zw = z @ W
        lags = [l for l in (1, 2, 4, 8, 16, 32) if l < max(2, self.seg_len // 4) and l < T]
        Cl = np.zeros((p, p))
        for l in lags or [1]:
            Cl += zw[l:].T @ zw[:-l] / (T - l)
        Cl = 0.5 * (Cl + Cl.T)
        _, V = np.linalg.eigh(Cl)
        Mrot = W @ V                                                # x0 = z @ Mrot
        x0 = z @ Mrot
        # y = C_fa z = C_fa (Mrot')^-1 x0  ->  C_init = C_fa inv(Mrot')
        self.C_ = C_fa @ np.linalg.inv(Mrot.T)
        self.d_ = fa.mean_.copy()
        var_floor = self._var_floor(Y)
        self.R_ = np.maximum(fa.noise_variance_, var_floor)
        lo, hi = self._tau_limits()
        tau0 = [_autocorr_tau(x0[:, i], max_lag=max(2, self.seg_len // 2)) for i in range(p)]
        self.timescales_ = np.clip(np.asarray(tau0, float), max(lo, 1.0), min(hi, self.seg_len / 2.0))

    def _var_floor(self, Y: np.ndarray) -> np.ndarray:
        """Lower bound on R: min_var_frac x channel variance (and >= 1e-6)."""
        return np.maximum(self.min_var_frac * Y.var(axis=1), 1e-6)

    # ------------------------------------------------------------ public
    def fit(self, X: np.ndarray) -> "GPFA":
        """Fit by EM on a continuous recording X [C, T] (channels x frames)."""
        t0 = time.perf_counter()
        Y = np.asarray(X, np.float64)
        if Y.ndim != 2:
            raise ValueError(f"X must be [channels, frames], got shape {Y.shape}")
        q, T = Y.shape
        if self.n_latents > q:
            raise ValueError(f"n_latents={self.n_latents} > number of channels {q}")
        bounds = _segment_bounds(T, self.seg_len)
        groups: dict = {}
        for s, e in bounds:                                         # group pseudo-trials by length
            groups.setdefault(e - s, []).append(np.arange(s, e))
        groups = {L: np.stack(v) for L, v in groups.items()}
        var_floor = self._var_floor(Y)
        self._initialise(Y)
        self.log_likelihoods_ = []
        self.converged_ = False
        self.n_iter_ = 0
        for it in range(self.max_iter):
            ll, st = self._estep(Y, groups)
            self.log_likelihoods_.append(ll)
            if self.verbose and it % 10 == 0:
                print(f"  GPFA iter {it:3d}  LL={ll:.3f}  tau={np.round(self.timescales_, 2)}")
            if it >= 2:
                gain = ll - self.log_likelihoods_[-2]
                total = ll - self.log_likelihoods_[0]
                if gain < -1e-6 * abs(ll):
                    warnings.warn(f"GPFA log-likelihood decreased by {-gain:.3g} at iteration {it}",
                                  RuntimeWarning, stacklevel=2)
                if gain < self.tol * abs(total):
                    self.converged_ = True
                    break
            self._mstep_obs(Y, st, var_floor)
            self._mstep_tau(st["S"])
            self.n_iter_ = it + 1
        if not self.converged_:
            # log-likelihood of the final parameters (after the last M-step)
            ll, _ = self._estep(Y, groups)
            self.log_likelihoods_.append(ll)
        self._cache = {}
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def transform(self, X: np.ndarray, orthonormalize: bool = False) -> np.ndarray:
        """Posterior means E[x_t | y] for the whole recording X [C, T] -> [T, p].

        Overlapping windows of ``seg_len`` frames (hop seg_len // 2, last
        window flush with the end) are blended with a Hann taper (no seams).
        ``orthonormalize=True`` returns Yu et al.'s orthonormalised latents
        (C = U D V', x_orth = D V' x: ordered by shared variance explained);
        the per-dimension timescales then no longer apply."""
        self._check_fitted()
        Y = np.asarray(X, np.float64)
        p, (q, T) = self.n_latents, Y.shape
        if q != self.C_.shape[0]:
            raise ValueError(f"X has {q} channels, model was fitted on {self.C_.shape[0]}")
        L = int(min(self.seg_len, T))
        starts = _window_starts(T, L)
        Minv, _, _ = self._posterior_factor(L)
        Bfull = (self.C_ / self.R_[:, None]).T @ (Y - self.d_[:, None])           # [p, T]
        idx = np.asarray(starts)[:, None] + np.arange(L)[None]                     # [n_win, L]
        Bw = Bfull[:, idx].transpose(0, 2, 1).reshape(p * L, len(starts))
        Mu = (Minv @ Bw).reshape(p, L, len(starts))
        w = np.hanning(L + 2)[1:-1] if len(starts) > 1 else np.ones(L)              # positive taper
        acc = np.zeros((p, T))
        wsum = np.zeros(T)
        for j, s in enumerate(starts):
            acc[:, s:s + L] += Mu[:, :, j] * w[None]
            wsum[s:s + L] += w
        Z = (acc / wsum[None]).T                                                    # [T, p]
        if orthonormalize:
            _, Dg, Vt = np.linalg.svd(self.C_, full_matrices=False)
            Z = Z @ (np.diag(Dg) @ Vt).T
        return Z

    def fit_transform(self, X: np.ndarray, orthonormalize: bool = False) -> np.ndarray:
        """``fit(X)`` then ``transform(X)``."""
        return self.fit(X).transform(X, orthonormalize=orthonormalize)

    def log_likelihood(self, X: np.ndarray) -> float:
        """Marginal log-likelihood of X [C, T] under the pseudo-trial model
        (same segmentation as ``fit``); usable on held-out data."""
        self._check_fitted()
        Y = np.asarray(X, np.float64)
        groups: dict = {}
        for s, e in _segment_bounds(Y.shape[1], self.seg_len):
            groups.setdefault(e - s, []).append(np.arange(s, e))
        groups = {L: np.stack(v) for L, v in groups.items()}
        ll, _ = self._estep(Y, groups)
        return ll


# --------------------------------------------------------------------------- #
# NEMBA backbone wrapper                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class GPFABackboneConfig:
    """Configuration of the GPFA backbone.

    ``embedding_dim`` is the number of latents (clamped to the number of
    channels). Cost of one EM iteration grows as (embedding_dim * seg_len)^3,
    so 8 latents x 100 frames is cheap but 32 x 100 is ~60x slower.
    ``causal`` is accepted for interface parity only: the GP posterior is a
    smoother, so ``causal=True`` is ignored with a warning."""
    num_features: int = 45
    embedding_dim: int = 8
    max_len: int = 1024          # unused, accepted for interface parity
    causal: bool = False
    seg_len: int = 100           # pseudo-trial / inference window length (frames)
    max_iter: int = 100          # EM iterations
    tol: float = 1e-4            # relative LL-gain stopping rule (see GPFA)
    seed: int = 0
    noise_var: float = 1e-3      # fixed GP noise variance (Yu et al. 2009)
    min_var_frac: float = 0.01   # private-variance floor (fraction of channel variance)
    tau_iters: int = 10          # L-BFGS iterations for the timescales per EM step
    orthonormalize: bool = False  # True: Yu et al.'s orthonormalised latents

    @classmethod
    def from_dict(cls, payload: dict) -> "GPFABackboneConfig":
        """Build from a dict, silently ignoring keys that are not fields."""
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in payload.items() if k in fields})

    def to_dict(self) -> dict:
        """Plain-dict view of the config."""
        return asdict(self)


class GPFABackbone(nn.Module):
    """GPFA as a NEMBA backbone (closed-form EM fit, no SGD).

    Fitted parameters live in float64 buffers (``C``, ``d``, ``R``,
    ``log_tau``), so ``state_dict`` save / load restores a usable model.
    ``transform_numpy(X [C, T]) -> Z [T, E]`` is the native whole-recording
    inference used by ``training.embed_sequence``; ``encode`` is a per-window
    fallback with the usual [B, C, L] -> [B, E, L] torch signature."""

    trainable = False

    def __init__(self, cfg: GPFABackboneConfig) -> None:
        """Allocate the parameter buffers for ``cfg`` (unfitted)."""
        super().__init__()
        if cfg.causal:
            warnings.warn("GPFA posterior means are acausal (smoother); causal=True is ignored",
                          stacklevel=2)
            cfg.causal = False
        self.cfg = cfg
        q = int(cfg.num_features)
        self.k = int(max(1, min(cfg.embedding_dim, q)))
        if self.k * int(cfg.seg_len) > 1600:
            # EM cost per iteration grows as (k * seg_len)^3: 8 x 100 is ~0.1 s,
            # 32 x 100 is several seconds (minutes for a full fit)
            warnings.warn(f"GPFA with {self.k} latents x seg_len={cfg.seg_len} is slow "
                          f"(cost ~ (latents * seg_len)^3 per EM iteration); consider "
                          f"embedding_dim <= 8 or a shorter seg_len", stacklevel=2)
        self.register_buffer("C", torch.zeros(q, self.k, dtype=torch.float64))
        self.register_buffer("d", torch.zeros(q, dtype=torch.float64))
        self.register_buffer("R", torch.ones(q, dtype=torch.float64))
        self.register_buffer("log_tau", torch.zeros(self.k, dtype=torch.float64))
        self.register_buffer("fitted", torch.tensor(False))
        self.log_likelihoods_: list = []
        self._model: GPFA | None = None

    def _new_model(self) -> GPFA:
        """Estimator with this backbone's hyper-parameters."""
        c = self.cfg
        return GPFA(self.k, seg_len=c.seg_len, max_iter=c.max_iter, tol=c.tol,
                    noise_var=c.noise_var, min_var_frac=c.min_var_frac,
                    tau_iters=c.tau_iters, seed=c.seed)

    def fit_numpy(self, X: np.ndarray) -> "GPFABackbone":
        """Fit GPFA by EM on X [C, T] and store the parameters in the buffers."""
        m = self._new_model().fit(np.asarray(X, np.float64))
        with torch.no_grad():
            self.C.copy_(torch.from_numpy(m.C_))
            self.d.copy_(torch.from_numpy(m.d_))
            self.R.copy_(torch.from_numpy(m.R_))
            self.log_tau.copy_(torch.from_numpy(np.log(m.timescales_)))
            self.fitted.fill_(True)
        self.log_likelihoods_ = list(m.log_likelihoods_)
        self.n_iter_, self.converged_, self.fit_seconds_ = m.n_iter_, m.converged_, m.fit_seconds_
        self._model = m
        return self

    @property
    def timescales_(self) -> np.ndarray:
        """Learned GP timescales [E] in frames (one per latent dimension)."""
        return np.exp(self.log_tau.detach().cpu().numpy())

    def _estimator(self) -> GPFA:
        """Estimator rebuilt from the buffers (works after load_state_dict)."""
        if not bool(self.fitted):
            raise RuntimeError("GPFABackbone is not fitted; call fit_numpy(X) first")
        m = self._model
        if m is None or not np.allclose(m.C_, self.C.detach().cpu().numpy()):
            m = self._new_model().set_params(self.C.detach().cpu().numpy(),
                                              self.d.detach().cpu().numpy(),
                                              self.R.detach().cpu().numpy(),
                                              self.timescales_)
            self._model = m
        return m

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Posterior-mean latents Z [T, E] (float32) of the whole recording X [C, T]."""
        Z = self._estimator().transform(np.asarray(X, np.float64),
                                        orthonormalize=self.cfg.orthonormalize)
        return Z.astype(np.float32)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Fallback per-window inference: x [B, C, L] -> z [B, E, L] (float32)."""
        xs = x.detach().cpu().numpy()
        z = np.stack([self.transform_numpy(xs[b]).T for b in range(xs.shape[0])])
        return torch.from_numpy(z).to(device=x.device, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        """Embeddings only (GPFA has no SSL reconstruction head)."""
        return NeuralEmbTCNOutput(embeddings=self.encode(x, mask))
