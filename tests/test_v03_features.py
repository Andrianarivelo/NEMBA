"""Tests for the v0.3 features integrated in the core: Poisson likelihood,
neuron masking, co-smoothing, FA baseline, per-backbone embedding sizes,
PSID decoding rows, the dynamics bundle of ``run`` and T-PHATE projections."""

import os

import numpy as np
import pytest
import torch

from embedders_backbones import Embedder, synthetic
from embedders_backbones import training as TR
from embedders_backbones.cosmoothing import cosmooth_score, poisson_bits_per_spike


@pytest.fixture(scope="module")
def ds():
    return synthetic.make_dataset(n_neurons=30, duration=240.0, dt=0.1, seed=4)


def test_element_masks_time_and_neuron():
    g = torch.Generator().manual_seed(0)
    em, fm = TR.make_element_mask(4, 10, 64, "neuron", neuron_frac=0.3, generator=g)
    assert fm is None and em.shape == (4, 10, 64)
    per_channel = em.all(dim=2)                         # whole channels hidden
    assert torch.equal(per_channel, em.any(dim=2)) and int(per_channel[0].sum()) == 3
    em, fm = TR.make_element_mask(4, 10, 64, "time", generator=g)
    assert fm.shape == (4, 64) and torch.equal(em[:, 0, :], fm)
    em, fm = TR.make_element_mask(2, 1, 32, "neuron", generator=g)   # 1 channel -> time
    assert fm is not None


def test_poisson_loss_prefers_true_rate():
    y = torch.poisson(torch.full((2, 3, 50), 2.0))
    m = torch.ones_like(y, dtype=torch.bool)
    good = TR.masked_element_loss(y, torch.full_like(y, np.log(2.0)), m, "poisson")
    bad = TR.masked_element_loss(y, torch.full_like(y, np.log(8.0)), m, "poisson")
    assert good < bad


def test_bits_per_spike_reference_values():
    rng = np.random.default_rng(0)
    rate = rng.uniform(0.2, 3.0, 2000)
    y = rng.poisson(rate)
    assert poisson_bits_per_spike(y, rate, y.mean()) > 0.1           # true rate beats flat
    assert abs(poisson_bits_per_spike(y, np.full_like(rate, y.mean()), y.mean())) < 1e-9


def test_poisson_both_training_reports_bits_per_spike(ds):
    emb = Embedder("tcn", bin_size=0.1, n_states=3, likelihood="auto", mask_mode="both",
                   max_steps=40, time_budget_s=None, log=None,
                   model_kwargs=dict(dilations=(1, 2, 4))).fit_spikes(ds["spike_times"])
    assert emb.likelihood_used == "poisson"
    assert np.isfinite(emb.history["best_val_bps"])
    assert emb.metrics()["likelihood"] == "poisson"


def test_cosmoothing_pca_beats_flat_rates(ds):
    res = cosmooth_score("pca", ds["counts"], log=None)
    assert res["bps"] > 0.05 and res["n_heldout"] == 6
    assert len(res["bps_per_neuron"]) == 6


def test_fa_baseline_and_default_dims(ds):
    fa = Embedder("fa", bin_size=0.1, n_states=3, log=None).fit_spikes(ds["spike_times"])
    assert fa.Z.shape[1] == 30 and np.isfinite(fa.Z).all()
    assert Embedder("gpfa", log=None).embedding_dim == 8
    assert Embedder("tcn", log=None).embedding_dim == 32
    assert Embedder("tcn", embedding_dim=12, log=None).embedding_dim == 12


def test_psid_rows_are_cross_fitted(ds):
    pytest.importorskip("PSID")
    emb = Embedder("pca", bin_size=0.1, n_states=3, log=None).fit_spikes(ds["spike_times"])
    tab = emb.decode(ds["behavior"], ds["times"], circular=["phase"], n_perm=20, psid=True)
    ps = tab[tab.features == "psid"].set_index("target")
    assert set(ps.index) == {"phase", "speed"}                    # classification skipped
    assert ps.loc["phase", "score"] > 0.5


def test_run_writes_dynamics_bundle_and_tphate(tmp_path, ds):
    emb = Embedder("pca", bin_size=0.1, n_states=3, log=None).fit_spikes(ds["spike_times"])
    emb.run(str(tmp_path), behavior=ds["behavior"], behavior_times=ds["times"],
            circular=["phase"], methods=("pca", "tphate"), gif_method="pca", animate=False,
            html=False, n_perm=30, formats=("png",))
    names = set(os.listdir(tmp_path))
    for f in ("pca_topology.png", "pca_flow_field.png", "pca_recurrence.png",
              "pca_rastermap.png", "pca_contingency_state.png", "pca_dynamics_stats.csv",
              "pca_neuron_order.csv"):
        assert f in names, f
    P, idx = emb.projection("tphate")
    assert P.shape == (len(idx), 2) and np.isfinite(P).all()
