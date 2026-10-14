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


def demo_spikes(n=40, dur=400.0, dt=0.05, seed=0):
    """The v0.1 GUI demo: a population tuned to a slowly drifting ring phase."""
    rng = np.random.default_rng(seed); T = int(dur / dt); t = np.arange(T) * dt
    phase = 2 * np.pi * 0.02 * t + 0.3 * np.sin(2 * np.pi * 0.005 * t)
    pref = rng.uniform(0, 2 * np.pi, n)
    rates = 2 + 8 * np.exp(2.0 * (np.cos(phase[None] - pref[:, None]) - 1))
    counts = rng.poisson(rates * dt)
    return [np.sort(t[np.repeat(np.arange(T), counts[j])] + rng.uniform(0, dt, counts[j].sum()))
            for j in range(n)]
