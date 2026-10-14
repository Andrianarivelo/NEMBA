"""End-to-end example on a synthetic recording with known behaviour.

    python examples/example_synthetic.py

1. simulate 60 neurons driven by a ring phase, running speed and 3 states;
2. embed with several backbones (always including the no-training PCA baseline);
3. decode the true behaviour on held-out time (blocked CV + circular-shift null);
4. write per-backbone outputs and a comparison (figures in png/pdf/svg + CSV).
"""

# =========================================================================== #
# PARAMETERS (edit here)                                                      #
# =========================================================================== #
BACKBONES = ["pca", "tcn", "tst", "bilstm"]   # "pca" = honest baseline, no training
SPEED = "fast"                               # fast | balanced | accurate (CPU budget)
OBJECTIVE = "masked"                         # masked | contrastive | hybrid
N_NEURONS = 60
DURATION_S = 600.0
BIN_S = 0.1
N_PERM = 200                                 # circular-shift null size
SEED = 0
OUT_DIR_NAME = "demo_out"
# =========================================================================== #

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedders_backbones import compare_embedders, synthetic  # noqa: E402


def main():
    ds = synthetic.make_dataset(n_neurons=N_NEURONS, duration=DURATION_S, dt=BIN_S, seed=SEED)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_DIR_NAME)

    # one call: fit every backbone, decode the ground truth with each of them
    cmp = compare_embedders(ds["spike_times"], input="spikes", backbones=BACKBONES,
                            behavior=ds["behavior"], behavior_times=ds["times"],
                            circular=["phase"], n_perm=N_PERM, speed=SPEED,
                            objective=OBJECTIVE, bin_size=BIN_S, seed=SEED)
    cmp.run(os.path.join(out, "compare"))

    # full per-backbone report (dashboard, decoding figure, explorer, ...)
    for name, emb in cmp.embedders.items():
        emb.run(os.path.join(out, name), behavior=ds["behavior"], behavior_times=ds["times"],
                circular=["phase"], n_perm=N_PERM, animate=False)

    print("\nheld-out decoding score per backbone (embedding):")
    print(cmp.decoding_table().round(3).to_string())
    print("\nfit time (s):")
    print(cmp.metrics.set_index("backbone")["total_seconds"].round(1).to_string())
    print(f"\noutputs in {out}")


if __name__ == "__main__":
    main()
