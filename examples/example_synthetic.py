"""Run every backbone on a synthetic ring dataset and print circularity.
    python examples/example_synthetic.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedders_backbones import Embedder, list_backbones


def demo_spikes(n=40, dur=300.0, dt=0.05, seed=0):
    rng = np.random.default_rng(seed); T = int(dur / dt); t = np.arange(T) * dt
    phase = 2 * np.pi * 0.02 * t
    pref = rng.uniform(0, 2 * np.pi, n)
    rates = 2 + 8 * np.exp(2.0 * (np.cos(phase[None] - pref[:, None]) - 1))
    counts = rng.poisson(rates * dt)
    return [np.sort(t[np.repeat(np.arange(T), counts[j])] + rng.uniform(0, dt, counts[j].sum()))
            for j in range(n)]


def main():
    spikes = demo_spikes()
    out = os.path.join(os.path.dirname(__file__), "demo_out")
    for bb in ["dtc", "tst", "bilstm", "stt", "gatr"]:
        emb = Embedder(backbone=bb, n_states=8, epochs=40, device="auto").fit_spikes(spikes)
        m = emb.metrics()
        emb.plot_embeddings(os.path.join(out, f"{bb}_embeddings.png"))
        print(f"{bb:10s} circularity={m['circularity']:.3f}  PR={m['participation_ratio']:.2f}")
    print(f"\nfigures in {out}; backbones available: {list_backbones()}")


if __name__ == "__main__":
    main()
