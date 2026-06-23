import torch

import embedders_backbones._tcn_variants as V
from embedders_backbones.backbones import make_model
from embedders_backbones.data import coco_masks_to_matrix


TCN_VARIANTS = [
    "tcn",
    "ed_tcn",
    "dilated_tcn",
    "ms_tcn",
    "ms_tcnpp",
    "c2f_tcn",
    "stcn",
    "c_tcn",
    "dc_tcn",
    "nac_tcn",
    "dwa_tcn",
    "pit_tcn",
]


def test_tcn_variants_preserve_embedding_and_reconstruction_shapes():
    model_kwargs = dict(
        d_model=12,
        num_layers=2,
        num_stages=2,
        dilations=(1, 2),
        c2f_depth=2,
        dense_block_layers=(2,),
        dense_growth=6,
        multiscale_kernels=(3,),
        attention_heads=3,
        attention_window=3,
        stochastic_latent_dim=4,
        kernel_size=3,
    )
    x = torch.randn(2, 6, 32)
    mask = torch.zeros(2, 32, dtype=torch.bool)

    for name in TCN_VARIANTS:
        model = make_model(name, num_features=6, embedding_dim=4, max_len=32, **model_kwargs)
        out = model(x, mask)
        assert out.embeddings.shape == (2, 4, 32)
        assert out.reconstruction.shape == (2, 6, 32)


def test_source_causal_tcn_variants_default_to_causal_mode():
    assert make_model("tcn", num_features=3).cfg.causal is True
    assert make_model("nac_tcn", num_features=3).cfg.causal is True
    assert make_model("tcn", num_features=3, causal=False).cfg.causal is False


def test_stcn_exposes_prior_posterior_latent_hierarchy():
    model = make_model(
        "stcn",
        num_features=5,
        embedding_dim=4,
        d_model=12,
        dilations=(1, 2, 4),
        stochastic_latent_dim=3,
        kernel_size=3,
    )
    out = model(torch.randn(2, 5, 24), torch.zeros(2, 24, dtype=torch.bool))

    assert out.aux_loss is not None
    assert set(model.latent_dists) == {"q", "p"}
    assert len(model.latent_dists["q"]) == 3
    assert len(model.latent_dists["p"]) == 3
    q_mu, q_logvar = model.latent_dists["q"][0]
    p_mu, p_logvar = model.latent_dists["p"][0]
    assert q_mu.shape == p_mu.shape == (2, 3, 24)
    assert q_logvar.shape == p_logvar.shape == (2, 3, 24)


def test_dwa_uses_soft_dtw_alignment_convolution():
    model = make_model(
        "dwa_tcn",
        num_features=5,
        embedding_dim=4,
        d_model=8,
        num_layers=2,
        kernel_size=3,
    )

    assert any(isinstance(module, V.SoftDTWAlignmentConv1d) for module in model.modules())
    out = model(torch.randn(2, 5, 16), torch.zeros(2, 16, dtype=torch.bool))
    assert out.embeddings.shape == (2, 4, 16)


def test_pit_exports_learned_discrete_temporal_architecture():
    model = make_model(
        "pit_tcn",
        num_features=5,
        embedding_dim=4,
        d_model=8,
        dilations=(1, 2),
        kernel_size=5,
    )
    _ = model(torch.randn(2, 5, 16), torch.zeros(2, 16, dtype=torch.bool))
    arch = model.export_architecture()

    assert arch["blocks"]
    first = arch["blocks"][0]["conv1"]
    assert first["learned_dilation"] in first["candidate_dilations"]
    assert first["active_kernel_indices"]
    assert len(first["position_gates"]) == 5


def test_coco_masks_to_matrix_rasterizes_raw_tracks():
    coco = {
        "images": [
            {"id": 1, "height": 10, "width": 10, "frame_index": 0},
            {"id": 2, "height": 10, "width": 10, "frame_index": 1},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "track_id": 7,
                "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
            },
            {
                "id": 2,
                "image_id": 2,
                "track_id": 8,
                "bbox": [6, 6, 2, 2],
            },
        ],
    }

    X, times = coco_masks_to_matrix(
        coco, resize=(8, 8), channel_key="track_id", standardize=False
    )

    assert X.shape == (2 * 8 * 8, 2)
    assert times.tolist() == [0.0, 1.0]
    assert X.sum() > 0
