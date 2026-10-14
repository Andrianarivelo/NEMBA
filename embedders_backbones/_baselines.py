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


class FactorAnalysisBaseline(PCABaseline):
    """Factor analysis: like PCA but with a separate noise variance per channel
    (Cunningham & Yu, Nat Neurosci 2014). The embedding is the posterior mean of
    the factors, E[z | x] = (I + L' Psi^-1 L)^-1 L' Psi^-1 (x - mu), which is a
    fixed linear map, so it is also implemented as a 1x1 convolution."""

    def fit_numpy(self, X: np.ndarray) -> "FactorAnalysisBaseline":
        """Fit factors on X [C, T] and load the posterior-mean map."""
        from sklearn.decomposition import FactorAnalysis
        X = np.asarray(X, dtype=np.float64)
        fa = FactorAnalysis(n_components=self.k, random_state=0).fit(X.T)
        Lam = fa.components_.T                                    # [C, k]
        psi_inv = 1.0 / np.maximum(fa.noise_variance_, 1e-6)      # [C]
        A = np.eye(self.k) + (Lam.T * psi_inv) @ Lam              # [k, k]
        W = np.linalg.solve(A, Lam.T * psi_inv)                   # [k, C]
        mu = fa.mean_
        total = (Lam ** 2).sum() + fa.noise_variance_.sum()
        with torch.no_grad():
            self.proj.weight.copy_(torch.from_numpy(W[:, :, None]).float())
            self.proj.bias.copy_(torch.from_numpy(-(W @ mu)).float())
            # share of the total variance carried by each factor
            self.explained_variance_ratio.copy_(torch.from_numpy(
                (Lam ** 2).sum(0) / max(total, 1e-12)).float())
        self.noise_variance_ = fa.noise_variance_
        return self
