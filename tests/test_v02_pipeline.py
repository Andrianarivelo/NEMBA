"""Tests for the v0.2 pipeline: leak-free masked training, fast inference,
time-series-aware decoding, persistence, data adapters and figures."""

import os

import numpy as np
import pytest
import torch

from embedders_backbones import Embedder, list_backbones, synthetic
from embedders_backbones import training as TR
from embedders_backbones import decoding as DEC
from embedders_backbones import data as D
from embedders_backbones.backbones import make_model


# --------------------------------------------------------------------------- #
# masked self-supervision must not leak the answer                            #
# --------------------------------------------------------------------------- #

def test_masked_training_on_white_noise_cannot_beat_chance():
    """White noise is unpredictable from its neighbours, so the masked loss of
    an honest model stays near the noise variance (1 after z-scoring). v0.1
    TCNs saw the masked frames and drove this loss towards 0 (copying)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3000)).astype(np.float32)
    net, hist = TR.train("tcn", X, speed="fast", max_steps=150, eval_every=25, patience=50,
                         log=None, time_budget_s=None,
                         model_kwargs=dict(d_model=32, dilations=(1, 2, 4)))
    assert min(hist["val_loss"]) > 0.9


@pytest.mark.parametrize("name", ["tcn", "embtcn", "rcnn", "ed_tcn", "c2f_tcn"])
def test_reconstruction_ignores_masked_input_values(name):
    """After ``apply_input_mask`` the prediction cannot depend on the values
    hidden under the mask (would be a leak through the receptive field)."""
    torch.manual_seed(0)
    net = make_model(name, num_features=5, embedding_dim=4, max_len=64, d_model=16).eval()
    x = torch.randn(2, 5, 64)
    mask = torch.zeros(2, 64, dtype=torch.bool); mask[:, 20:36] = True
    x2 = x.clone(); x2[:, :, 20:36] += 5.0 * torch.randn(2, 5, 16)
    with torch.no_grad():
        r1 = net(TR.apply_input_mask(x, mask), mask).reconstruction
        r2 = net(TR.apply_input_mask(x2, mask), mask).reconstruction
    assert torch.allclose(r1, r2)


def test_span_mask_is_vectorised_and_covers_requested_ratio():
    g = torch.Generator().manual_seed(0)
    m = TR.make_span_mask(64, 256, 0.4, 16, generator=g)
    assert m.shape == (64, 256) and m.dtype == torch.bool
    assert 0.25 < m.float().mean().item() <= 0.41          # spans may overlap


# --------------------------------------------------------------------------- #
# inference                                                                    #
# --------------------------------------------------------------------------- #

def test_overlap_add_embedding_has_no_seams_for_framewise_model():
    """For a per-frame model (PCA baseline) windowed inference must equal the
    direct projection exactly: any seam / padding artefact would show."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, 1000)).astype(np.float32)
    net, _ = TR.train("pca", X, log=None)
    Z = TR.embed_sequence(net, X, win_len=128)
    with torch.no_grad():
        direct = net.encode(torch.from_numpy(X)[None]).squeeze(0).numpy().T
    assert np.allclose(Z, direct, atol=1e-4)


@pytest.mark.parametrize("name", list_backbones())
def test_every_backbone_trains_and_embeds(name):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 160)).astype(np.float32)
    kw = dict(d_model=16, num_heads=2, hidden_size=16, conv_channels=16, dilations=(1, 2),
              c2f_depth=2, dense_block_layers=(2,), dense_growth=8, multiscale_kernels=(3,),
              attention_heads=2, attention_window=3, stochastic_latent_dim=4, kernel_size=3,
              num_layers=2, num_stages=2, n_groups=2, patch_len=4, tcn_dilations=(1, 2))
    net, hist = TR.train(name, X, embedding_dim=4, max_steps=3, win_len=32, batch=4,
                         eval_every=2, log=None, model_kwargs=kw, time_budget_s=None)
    Z = TR.embed_sequence(net, X, win_len=hist["win_len"])
    assert Z.shape[0] == 160 and np.isfinite(Z).all()


def test_contrastive_and_hybrid_objectives_run():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 600)).astype(np.float32)
    for obj in ("contrastive", "hybrid"):
        _, hist = TR.train("tcn", X, objective=obj, max_steps=6, win_len=64, batch=8,
                           eval_every=3, log=None, time_budget_s=None,
                           model_kwargs=dict(d_model=16, dilations=(1, 2)))
        assert np.isfinite(hist["train_loss"]).all()


# --------------------------------------------------------------------------- #
# decoding                                                                     #
# --------------------------------------------------------------------------- #

def _ar1(T, rho, rng):
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + rng.normal()
    return x


def test_blocked_folds_are_disjoint_with_gap():
    for tr, te in DEC.blocked_folds(1000, n_folds=5, gap=30):
        assert np.intersect1d(tr, te).size == 0
        d = np.abs(tr[:, None] - te[None, :]).min()
        assert d > 30


def test_decoding_finds_linear_signal():
    rng = np.random.default_rng(0)
    Z = np.c_[_ar1(3000, 0.98, rng), _ar1(3000, 0.98, rng), rng.normal(size=(3000, 4))]
    y = 2 * Z[:, 0] - Z[:, 1] + 0.3 * rng.normal(size=3000)
    r = DEC.decode_target(Z, y, n_perm=99)
    assert r["task"] == "regression" and r["score"] > 0.8 and r["p_value"] <= 0.02


def test_circular_shift_null_is_calibrated_for_independent_series():
    """Independent but strongly autocorrelated series: a frame-shuffle null
    would declare them related far too often; the circular-shift null must not."""
    hits = 0
    for s in range(20):
        rng = np.random.default_rng(100 + s)
        Z = np.c_[_ar1(1500, 0.99, rng), _ar1(1500, 0.99, rng)]
        y = _ar1(1500, 0.99, rng)
        hits += DEC.decode_target(Z, y, n_perm=99, seed=s)["p_value"] < 0.05
    assert hits <= 4                                    # nominal 1 of 20


def test_classification_and_circular_tasks():
    rng = np.random.default_rng(0)
    T = 2400
    lab = np.repeat(rng.integers(0, 3, T // 80), 80)
    Z = np.eye(3)[lab] + 0.3 * rng.normal(size=(T, 3))
    r = DEC.decode_target(Z, lab, n_perm=49)
    assert r["task"] == "classification" and r["score"] > 0.9 and abs(r["chance"] - 1 / 3) < 1e-9
    th = np.angle(np.exp(1j * np.cumsum(0.05 * rng.normal(size=T) + 0.02)))
    Zc = np.c_[np.cos(th), np.sin(th)] + 0.1 * rng.normal(size=(T, 2))
    rc = DEC.decode_target(Zc, th, task="circular", n_perm=49)
    assert rc["score"] > 0.9 and rc["mae_deg"] < 15


def test_holm_adjustment():
    adj = DEC.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


# --------------------------------------------------------------------------- #
# Embedder end to end                                                         #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def small_dataset():
    return synthetic.make_dataset(n_neurons=24, duration=240.0, dt=0.1, seed=3)


def test_embedder_decodes_ground_truth_and_roundtrips(tmp_path, small_dataset):
    ds = small_dataset
    emb = Embedder("tcn", speed="fast", max_steps=40, bin_size=0.1, n_states=3, log=None,
                   time_budget_s=None, model_kwargs=dict(dilations=(1, 2, 4)))
    emb.fit_spikes(ds["spike_times"])
    tab = emb.decode(ds["behavior"], ds["times"], circular=["phase"], n_perm=99)
    emb_rows = tab[tab.features == "embedding"].set_index("target")
    assert emb_rows.loc["state", "score"] > 0.6
    assert emb_rows.loc["phase", "score"] > 0.5
    assert set(tab.features) == {"embedding", "raw_pca"}
    p = emb.save(os.path.join(tmp_path, "enc.pt"))
    emb2 = Embedder.load(p, log=None)
    assert np.allclose(emb2.transform(emb.X_input), emb.Z, atol=1e-4)


def test_run_writes_all_outputs(tmp_path, small_dataset):
    ds = small_dataset
    emb = Embedder("pca", bin_size=0.1, log=None).fit_spikes(ds["spike_times"])
    emb.run(str(tmp_path), behavior=ds["behavior"], behavior_times=ds["times"],
            circular=["phase"], methods=("pca",), gif_method="pca", n_perm=60)
    names = set(os.listdir(tmp_path))
    for stem in ("pca_dashboard", "pca_embeddings", "pca_decoding", "pca_states", "pca_training"):
        for ext in ("png", "pdf", "svg"):
            assert f"{stem}.{ext}" in names
    for f in ("pca_decoding.csv", "pca_metrics.csv", "pca_params.json", "pca_explorer.html",
              "pca_trajectory.gif", "pca_embedding.npz", "pca_state_stats.csv"):
        assert f in names


# --------------------------------------------------------------------------- #
# data adapters                                                               #
# --------------------------------------------------------------------------- #

def test_fast_sliding_median_matches_exact():
    from scipy.ndimage import median_filter
    rng = np.random.default_rng(0)
    v = np.cumsum(rng.normal(size=60000)) * 0.01 + rng.normal(size=60000)
    fast = D._sliding_median(v, 3000)
    exact = median_filter(v, size=3000, mode="nearest")
    assert np.median(np.abs(fast - exact)) < 0.1 * v.std()


def test_input_reducer_auto_and_nan_interpolation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 500)).astype(np.float32)
    red = D.InputReducer("auto")
    Y = red.fit_transform(X)
    assert Y.shape == (64, 500) and np.allclose(red.transform(X), Y, atol=1e-4)
    assert D.InputReducer("auto").fit_transform(X[:50]).shape == (50, 500)
    pose = np.vstack([np.linspace(0, 1, 100)] * 2)
    pose[0, 40:45] = np.nan
    Xz, _ = D.features_to_matrix(pose)
    assert np.isfinite(Xz).all() and np.allclose(Xz[0], Xz[1], atol=1e-6)


def test_crf_batched_nll_matches_sequential():
    from embedders_backbones.backbones import LinearChainCRF
    torch.manual_seed(0)
    crf = LinearChainCRF(emb_dim=4, num_tags=3)
    z = torch.randn(50, 4); tags = torch.randint(0, 3, (50,))
    a = crf.nll(z, tags)
    b = crf.nll_batch(z[None], tags[None])
    assert torch.allclose(a, b, atol=1e-4)
    assert (crf.viterbi_np(z.numpy()) == crf.viterbi(z)).all()
