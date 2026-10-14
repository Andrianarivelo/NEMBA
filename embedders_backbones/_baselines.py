"""Closed-form, no-training baselines that share the backbone interface.

A deep self-supervised encoder is only worth its CPU time if it beats these.
``PCABaseline`` is a per-frame linear projection onto the top principal
components of the input (implemented as a fixed 1x1 convolution), so it plugs
into the same embedding / clustering / decoding / saving pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from ._embtcn import NeuralEmbTCNOutput


@dataclass
class PCABaselineConfig:
    num_features: int = 45
    embedding_dim: int = 32
    max_len: int = 1024          # unused, accepted for interface parity
    causal: bool = False

    @classmethod
    def from_dict(cls, payload: dict) -> "PCABaselineConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in payload.items() if k in fields})

    def to_dict(self) -> dict:
        return asdict(self)


class PCABaseline(nn.Module):
    """Top-k principal components of the input, frame by frame (no SGD)."""

    trainable = False

    def __init__(self, cfg: PCABaselineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.k = int(min(cfg.embedding_dim, cfg.num_features))
        self.proj = nn.Conv1d(cfg.num_features, self.k, 1)
        self.register_buffer("explained_variance_ratio", torch.zeros(self.k))
        for p in self.parameters():
            p.requires_grad_(False)

    def fit_numpy(self, X: np.ndarray) -> "PCABaseline":
        """Fit components on X [C, T] and load them into the 1x1 convolution."""
        X = np.asarray(X, dtype=np.float64)
        mu = X.mean(axis=1)
        Xc = (X - mu[:, None]).T                                    # [T, C]
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        W = Vt[: self.k]                                            # [k, C]
        lam = S ** 2
        with torch.no_grad():
            self.proj.weight.copy_(torch.from_numpy(W[:, :, None]).float())
            self.proj.bias.copy_(torch.from_numpy(-(W @ mu)).float())
            self.explained_variance_ratio.copy_(
                torch.from_numpy(lam[: self.k] / max(lam.sum(), 1e-12)).float())
        return self

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.proj(x)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        return NeuralEmbTCNOutput(embeddings=self.encode(x, mask))
