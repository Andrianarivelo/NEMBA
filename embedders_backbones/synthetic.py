"""Synthetic population recordings with known ground truth.

``make_dataset`` simulates Poisson spiking driven by three latent behaviours:

* ``phase``  angle on a ring (radians, like head direction or position on a
  circular track); drifts at a speed-dependent angular velocity;
* ``speed``  non-negative running speed (Ornstein-Uhlenbeck, state dependent);
* ``state``  discrete behavioural state (Markov chain: rest / explore / groom),
  each switching on its own neural assembly and setting the mean speed.

Neurons: ring cells (von Mises tuning to phase, gain grows with speed),
state-assembly cells (fire mostly in one state), and untuned noise cells.
Because the latents are known, embeddings can be scored by how well they
*decode* them on held-out time (and how well their states match ``state``).
"""

from __future__ import annotations

import numpy as np


def make_dataset(n_neurons=60, duration=600.0, dt=0.1, n_states=3, mean_dwell_s=12.0,
                 frac_ring=0.6, frac_state=0.3, seed=0):
    """Simulate a recording. Returns a dict with

    spike_times : list of arrays (s), one per neuron
    counts      : [N, T] binned spike counts at resolution ``dt``
    times       : [T] bin centres (s)
    behavior    : {'phase': [T] rad, 'speed': [T], 'state': [T] int}
    neuron_type : [N] 'ring' | 'state' | 'noise'
    """
    rng = np.random.default_rng(seed)
    T = int(round(duration / dt))
    times = (np.arange(T) + 0.5) * dt

    # --- discrete behavioural state (Markov, geometric dwell times) ---------
    p_stay = np.exp(-dt / mean_dwell_s)
    state = np.empty(T, dtype=int)
    state[0] = rng.integers(n_states)
    for t in range(1, T):
        if rng.random() < p_stay:
            state[t] = state[t - 1]
        else:
            state[t] = rng.choice([k for k in range(n_states) if k != state[t - 1]])

    # --- speed: OU process whose mean depends on the state ------------------
    mean_speed = np.linspace(2.0, 25.0, n_states)[np.argsort(rng.permutation(n_states))]
    mean_speed[0] = 2.0                       # state 0 = rest
    tau, sig = 2.0, 6.0
    speed = np.empty(T)
    speed[0] = mean_speed[state[0]]
    for t in range(1, T):
        mu = mean_speed[state[t]]
        speed[t] = speed[t - 1] + dt / tau * (mu - speed[t - 1]) + sig * np.sqrt(2 * dt / tau) * rng.normal()
    speed = np.clip(speed, 0.0, None)

    # --- ring phase: angular velocity proportional to speed + diffusion ----
    omega = 0.04 * speed * rng.choice([-1, 1])
    phase = np.cumsum(omega * dt + 0.05 * np.sqrt(dt) * rng.normal(size=T))
    phase = np.angle(np.exp(1j * phase))

    # --- neurons ------------------------------------------------------------
    n_ring = int(round(frac_ring * n_neurons))
    n_state = int(round(frac_state * n_neurons))
    n_noise = n_neurons - n_ring - n_state
    rates = np.empty((n_neurons, T))
    pref = rng.uniform(-np.pi, np.pi, n_ring)
    kappa = rng.uniform(1.5, 4.0, n_ring)
    peak = rng.uniform(8.0, 20.0, n_ring)
    gain = 0.5 + speed / (speed.max() + 1e-9)
    tuning = np.exp(kappa[:, None] * (np.cos(phase[None] - pref[:, None]) - 1.0))
    rates[:n_ring] = 0.5 + peak[:, None] * tuning * gain[None]
    owner = rng.integers(n_states, size=n_state)
    for i in range(n_state):
        on = (state == owner[i]).astype(float)
        rates[n_ring + i] = 1.0 + rng.uniform(6.0, 15.0) * on
    rates[n_ring + n_state:] = rng.uniform(1.0, 8.0, (n_noise, 1))

    counts = rng.poisson(rates * dt)
    spike_times = []
    for j in range(n_neurons):
        bins = np.repeat(np.arange(T), counts[j])
        spike_times.append(np.sort((bins + rng.random(bins.size)) * dt))
    types = np.array(["ring"] * n_ring + ["state"] * n_state + ["noise"] * n_noise)
    return dict(spike_times=spike_times, counts=counts.astype(np.float32), times=times,
                behavior=dict(phase=phase, speed=speed, state=state), neuron_type=types,
                dt=dt)


def _markov_states(T, n_states, p_stay, rng):
    """Discrete Markov chain with geometric dwell times."""
    s = np.empty(T, dtype=int)
    s[0] = rng.integers(n_states)
    for t in range(1, T):
        s[t] = s[t - 1] if rng.random() < p_stay else rng.choice(
            [k for k in range(n_states) if k != s[t - 1]])
    return s


def make_metastable(n_neurons=40, duration=400.0, dt=0.1, n_states=4, mean_dwell_s=8.0,
                    seed=0):
    """Discrete metastable population states (no continuous manifold): each state
    has its own random rate pattern. Returns dict like ``make_dataset``."""
    rng = np.random.default_rng(seed)
    T = int(round(duration / dt))
    state = _markov_states(T, n_states, np.exp(-dt / mean_dwell_s), rng)
    patterns = rng.gamma(2.0, 3.0, (n_states, n_neurons))
    counts = rng.poisson(patterns[state].T * dt).astype(np.float32)
    return dict(counts=counts, times=(np.arange(T) + 0.5) * dt, dt=dt,
                behavior=dict(state=state))


def make_sheet(n_neurons=60, duration=600.0, dt=0.1, seed=0):
    """Continuous 2-D latent without a loop (animal in an open box, place-like
    tuning): a sheet manifold, H1 persistence should stay low."""
    rng = np.random.default_rng(seed)
    T = int(round(duration / dt))
    pos = np.zeros((T, 2))
    v = np.zeros(2)
    for t in range(1, T):                      # reflected random walk in [0, 1]^2
        v = 0.95 * v + 0.01 * rng.normal(size=2)
        pos[t] = pos[t - 1] + v
        for d in range(2):
            if pos[t, d] < 0 or pos[t, d] > 1:
                v[d] = -v[d]; pos[t, d] = np.clip(pos[t, d], 0, 1)
    centres = rng.uniform(0, 1, (n_neurons, 2))
    d2 = ((pos[None] - centres[:, None]) ** 2).sum(-1)
    rates = 0.5 + 15.0 * np.exp(-d2 / (2 * 0.12 ** 2))
    counts = rng.poisson(rates * dt).astype(np.float32)
    return dict(counts=counts, times=(np.arange(T) + 0.5) * dt, dt=dt,
                behavior=dict(x=pos[:, 0], y=pos[:, 1]))


def make_calcium(n_neurons=60, duration=600.0, fs=10.0, tau_decay_s=0.6, seed=0):
    """Calcium-imaging-like traces (dF/F): spikes from the ring + state model of
    ``make_dataset`` convolved with an exponential decay plus noise. Returns
    dict(traces [N, T], times, fs, behavior)."""
    ds = make_dataset(n_neurons=n_neurons, duration=duration, dt=1.0 / fs, seed=seed)
    rng = np.random.default_rng(seed + 1)
    k = np.exp(-np.arange(int(5 * tau_decay_s * fs)) / (tau_decay_s * fs))
    traces = np.array([np.convolve(c, k)[: c.size] for c in ds["counts"]])
    traces = 0.4 * traces + 0.08 * rng.normal(size=traces.shape)
    return dict(traces=traces.astype(np.float32), times=ds["times"], fs=fs,
                behavior=ds["behavior"])


def make_photometry(duration=600.0, fs=20.0, n_channels=1, seed=0):
    """Fiber photometry: photobleaching decay + event transients + slow drift +
    motion artefacts shared with an isosbestic control channel.
    Returns dict(signal [C, T], control [C, T], times, fs, behavior)."""
    rng = np.random.default_rng(seed)
    T = int(round(duration * fs))
    t = np.arange(T) / fs
    events = rng.random(T) < 0.15 / fs                       # ~0.15 events per second
    kern = np.exp(-np.arange(int(4 * fs)) / (0.8 * fs)) - np.exp(-np.arange(int(4 * fs)) / (0.1 * fs))
    transients = np.convolve(events.astype(float), kern)[:T]
    motion = np.convolve(rng.normal(size=T), np.ones(int(fs)) / fs, "same") * 0.3
    sig, ctl = [], []
    for c in range(n_channels):
        bleach = 1.0 + 0.6 * np.exp(-t / (150 + 50 * c))
        sig.append(bleach * (1.0 + 0.15 * transients + 0.03 * motion) + 0.01 * rng.normal(size=T))
        ctl.append(bleach * (1.0 + 0.03 * motion) + 0.01 * rng.normal(size=T))
    return dict(signal=np.array(sig, np.float32), control=np.array(ctl, np.float32), times=t,
                fs=fs, behavior=dict(event=events.astype(int)))


def make_pose(n_keypoints=8, duration=300.0, fps=30.0, nan_frac=0.02, seed=0):
    """DeepLabCut-like pose: keypoints of a rigid-ish body moving in an arena
    with three behavioural states (rest / walk / groom) and tracking dropouts.
    Returns dict(X [T, 2K] with columns '<part>_x', '<part>_y', columns,
    times, fps, behavior)."""
    rng = np.random.default_rng(seed)
    T = int(round(duration * fps))
    state = _markov_states(T, 3, np.exp(-1.0 / (fps * 4.0)), rng)
    speed = np.array([0.2, 6.0, 0.5])[state] * (1 + 0.3 * rng.normal(size=T)).clip(0.2)
    heading = np.cumsum(rng.normal(0, 0.05, T) + (state == 2) * 0.15)
    pos = np.cumsum(np.c_[np.cos(heading), np.sin(heading)] * speed[:, None] / fps, 0)
    pos = (pos - pos.min(0)) / (np.ptp(pos, 0) + 1e-9) * 400 + 50
    template = np.c_[np.linspace(-30, 30, n_keypoints), rng.normal(0, 6, n_keypoints)]
    groom = (state == 2)[:, None] * 6 * np.sin(np.arange(T) / fps * 2 * np.pi * 5)[:, None]
    X = np.empty((T, 2 * n_keypoints))
    for k in range(n_keypoints):
        ox, oy = template[k]
        X[:, 2 * k] = pos[:, 0] + ox * np.cos(heading) - oy * np.sin(heading) + rng.normal(0, 1.5, T)
        X[:, 2 * k + 1] = pos[:, 1] + ox * np.sin(heading) + oy * np.cos(heading) + groom[:, 0] \
            + rng.normal(0, 1.5, T)
    drop = rng.random((T, n_keypoints)) < nan_frac             # dropouts hit x and y together
    for k in range(n_keypoints):
        X[drop[:, k], 2 * k:2 * k + 2] = np.nan
    names = [f"part{k}_{a}" for k in range(n_keypoints) for a in ("x", "y")]
    return dict(X=X.astype(np.float32), columns=names, times=np.arange(T) / fps, fps=fps,
                behavior=dict(state=state, speed=speed))


def demo_spikes(n=40, dur=400.0, dt=0.05, seed=0):
    """The v0.1 GUI demo: a population tuned to a slowly drifting ring phase."""
    rng = np.random.default_rng(seed); T = int(dur / dt); t = np.arange(T) * dt
    phase = 2 * np.pi * 0.02 * t + 0.3 * np.sin(2 * np.pi * 0.005 * t)
    pref = rng.uniform(0, 2 * np.pi, n)
    rates = 2 + 8 * np.exp(2.0 * (np.cos(phase[None] - pref[:, None]) - 1))
    counts = rng.poisson(rates * dt)
    return [np.sort(t[np.repeat(np.arange(T), counts[j])] + rng.uniform(0, dt, counts[j].sum()))
            for j in range(n)]
