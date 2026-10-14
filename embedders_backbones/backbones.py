"""Unified factory over all self-supervised sequence backbones.

Every backbone exposes the same interface:
    encode(x[B,C,T], mask[B,T]) -> z[B,E,T]          per-frame embedding
    forward(x, mask) -> out with out.reconstruction[B,C,T]   (SSL head)
so they all train by the shared objectives in ``training`` and embed
identically.

``make_model(name, num_features, ...)`` builds any of them by name.
"""

from __future__ import annotations

import torch.nn as nn

from . import _transformers as T
from . import _bilstm as B
from . import _embtcn as E
from . import _tcn_variants as V
from . import _baselines as P
from ._embtcn import make_span_mask, masked_reconstruction_loss   # re-export
from ._transformers import neuron_graph, DECAutoencoder, target_distribution, LinearChainCRF


# name -> (display label, clustering kind, needs functional graph)
BACKBONES = {
    "pca":         ("PCA baseline (no training)",   "kmeans", False),
    "dtc":         ("Transformer-AE + DeepCluster", "dec",    False),
    "tst":         ("Time-Series Transformer",      "kmeans", False),
    "bilstm":      ("Bi-LSTM",                       "kmeans", False),
    "attn_bilstm": ("Attention Bi-LSTM",             "kmeans", False),
    "stt":         ("Spatiotemporal Transformer",    "kmeans", False),
    "gatr":        ("Graph-Attention Transformer",   "kmeans", True),
    "vit":         ("Vision Transformer",            "kmeans", False),
    "enc_crf":     ("Encoder + CRF",                 "crf",    False),
    "rcnn":        ("Recurrent CNN",                 "kmeans", False),
    "patchtst":    ("PatchTST",                      "kmeans", False),
    "embtcn":      ("EmbTCN-Attention",              "kmeans", False),
    "tcn":          ("Generic Residual TCN",          "kmeans", False),
    "ed_tcn":       ("Encoder-Decoder TCN",           "kmeans", False),
    "dilated_tcn":  ("Dilated/WaveNet TCN",           "kmeans", False),
    "ms_tcn":       ("Multi-Stage TCN",               "kmeans", False),
    "ms_tcnpp":     ("MS-TCN++ Dual-Dilated",         "kmeans", False),
    "c2f_tcn":      ("Coarse-to-Fine TCN",            "kmeans", False),
    "stcn":         ("Stochastic TCN",                "kmeans", False),
    "c_tcn":        ("Concept-wise TCN",              "kmeans", False),
    "dc_tcn":       ("Densely Connected TCN",         "kmeans", False),
    "nac_tcn":      ("Neighborhood Attention TCN",    "kmeans", False),
    "dwa_tcn":      ("Dynamic Alignment TCN",         "kmeans", False),
    "pit_tcn":      ("Pruning-in-Time TCN",           "kmeans", False),
}

# Models that build one token per (channel, frame): cost grows with channels x
# time^2, so they need small batches and benefit most from input PCA.
CHANNEL_TOKEN_MODELS = {"stt", "gatr", "patchtst"}

# Relative CPU cost per training step measured on 60 channels x 256 frames
# (v0.2, 16-thread laptop CPU). Used by the GUI / docs to warn users.
CPU_COST_HINT = {"pca": "instant", "stt": "slow", "gatr": "slow", "dwa_tcn": "slow",
                 "dtc": "medium", "patchtst": "medium", "rcnn": "medium"}


def list_backbones() -> list[str]:
    return list(BACKBONES)


def label(name: str) -> str:
    return BACKBONES.get(name, (name,))[0]


def clustering_kind(name: str) -> str:
    return BACKBONES.get(name, (None, "kmeans"))[1]


def needs_graph(name: str) -> bool:
    return BACKBONES.get(name, (None, None, False))[2]


def _config_class(name: str):
    """Config dataclass of a backbone (``None`` if unknown)."""
    if name in T.MODEL_REGISTRY:
        return T.MODEL_REGISTRY[name][1]
    if name in ("bilstm", "attn_bilstm"):
        return B.NeuralBiLSTMConfig
    if name == "embtcn":
        return E.NeuralEmbTCNConfig
    if name in V.MODEL_REGISTRY:
        return V.TCNVariantConfig
    if name == "pca":
        return P.PCABaselineConfig
    return None


def default_config(name: str):
    """Default config instance of a backbone (for introspection / presets)."""
    cls = _config_class(name)
    if cls is None:
        raise ValueError(f"unknown backbone {name!r}; choose from {list_backbones()}")
    return cls()


def width_overrides(name: str, width: float) -> dict:
    """Scale the hidden width of a backbone by ``width`` (rounded to a multiple
    of 8 so attention heads and GroupNorm groups still divide it)."""
    if width is None or abs(float(width) - 1.0) < 1e-9:
        return {}
    cfg = default_config(name)
    out = {}
    for key in ("d_model", "hidden_size", "conv_channels"):
        if hasattr(cfg, key):
            out[key] = max(16, int(round(getattr(cfg, key) * float(width) / 8.0)) * 8)
    return out


def make_model(name: str, num_features: int, embedding_dim: int = 32,
               max_len: int = 512, **overrides) -> nn.Module:
    """Construct a backbone by name. Unknown override keys are ignored, so one
    ``model_kwargs`` dict can be shared across a comparison of backbones."""
    base = dict(num_features=num_features, embedding_dim=embedding_dim, max_len=max_len)
    if name in T.MODEL_REGISTRY:
        klass, cfg_cls = T.MODEL_REGISTRY[name]
        net = klass(cfg_cls.from_dict({**base, **overrides}))
    elif name == "bilstm":
        net = B.NeuralBiLSTM(B.NeuralBiLSTMConfig.from_dict({**base, **overrides}))
    elif name == "attn_bilstm":
        net = B.NeuralBiLSTM(B.NeuralBiLSTMConfig.from_dict(
            {**base, **overrides, "use_attention": True}))
    elif name == "embtcn":
        net = E.NeuralEmbTCN(E.NeuralEmbTCNConfig.from_dict({**base, **overrides}))
    elif name in V.MODEL_REGISTRY:
        net = V.make_tcn_model(name, num_features, embedding_dim=embedding_dim,
                               max_len=max_len, **overrides)
    elif name == "pca":
        net = P.PCABaseline(P.PCABaselineConfig.from_dict({**base, **overrides}))
    else:
        raise ValueError(f"unknown backbone {name!r}; choose from {list_backbones()}")
    net.channel_tokens = name in CHANNEL_TOKEN_MODELS
    return net
