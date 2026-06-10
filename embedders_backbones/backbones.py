"""Unified factory over all self-supervised sequence backbones.

Every backbone exposes the same interface:
    encode(x[B,C,T], mask[B,T]) -> z[B,E,T]          per-frame embedding
    forward(x, mask) -> out with out.reconstruction[B,C,T]   (SSL head)
so they all train by masked reconstruction and embed identically.

``make_model(name, num_features, ...)`` builds any of them by name.
"""

from __future__ import annotations

import torch.nn as nn

from . import _transformers as T
from . import _bilstm as B
from . import _embtcn as E
from ._embtcn import make_span_mask, masked_reconstruction_loss   # re-export
from ._transformers import neuron_graph, DECAutoencoder, target_distribution, LinearChainCRF


# name -> (display label, clustering kind, needs functional graph)
BACKBONES = {
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
}


def list_backbones() -> list[str]:
    return list(BACKBONES)


def label(name: str) -> str:
    return BACKBONES.get(name, (name,))[0]


def clustering_kind(name: str) -> str:
    return BACKBONES.get(name, (None, "kmeans"))[1]


def needs_graph(name: str) -> bool:
    return BACKBONES.get(name, (None, None, False))[2]


def make_model(name: str, num_features: int, embedding_dim: int = 32,
               max_len: int = 512, **overrides) -> nn.Module:
    """Construct a backbone by name."""
    if name in T.MODEL_REGISTRY:
        klass, cfg_cls = T.MODEL_REGISTRY[name]
        cfg = cfg_cls.from_dict(dict(num_features=num_features, embedding_dim=embedding_dim,
                                     max_len=max_len, **overrides))
        return klass(cfg)
    if name == "bilstm":
        return B.NeuralBiLSTM(B.NeuralBiLSTMConfig(
            num_features=num_features, embedding_dim=embedding_dim, **overrides))
    if name == "attn_bilstm":
        return B.NeuralBiLSTM(B.NeuralBiLSTMConfig(
            num_features=num_features, embedding_dim=embedding_dim, use_attention=True, **overrides))
    if name == "embtcn":
        return E.NeuralEmbTCN(E.NeuralEmbTCNConfig(
            num_features=num_features, embedding_dim=embedding_dim, max_len=max_len, **overrides))
    raise ValueError(f"unknown backbone {name!r}; choose from {list_backbones()}")
