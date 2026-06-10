"""Bidirectional-LSTM self-supervised embedding model for population spike data.

Drop-in alternative to ``neural_embtcn.NeuralEmbTCN`` with the same interface so
the same training / embedding / clustering pipeline can use it:

  * input  x : [B, N, T]
  * output Z : [B, E, T]   per-frame embedding
  * decoder  : masked-input reconstruction head for SSL pretraining
  * ``encode(x, mask)`` and ``forward(x, mask)`` mirror the EmbTCN model, and the
    forward returns a ``NeuralEmbTCNOutput`` so downstream code is unchanged.

Architecture: per-frame input projection to ``d_model`` (with a learned mask
token for masked SSL), a stacked bidirectional LSTM over time, then a linear
fusion of the forward+backward hidden states into the ``E``-dim embedding, plus a
small MLP decoder back to the ``N`` input channels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from ._embtcn import NeuralEmbTCNOutput, DualBranchAttention


@dataclass
class NeuralBiLSTMConfig:
    num_features: int = 45            # N input channels
    d_model: int = 96
    embedding_dim: int = 32           # E
    hidden_size: int = 96             # per-direction LSTM hidden size
    num_layers: int = 2
    dropout: float = 0.15
    use_decoder: bool = True
    use_attention: bool = False       # dual-branch temporal+channel attention
    temperature: float = 0.5          # temporal-attention temperature

    @classmethod
    def from_dict(cls, payload: dict) -> "NeuralBiLSTMConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in payload.items() if k in fields})

    def to_dict(self) -> dict:
        return asdict(self)


class NeuralBiLSTM(nn.Module):
    """Bidirectional LSTM backbone + SSL reconstruction head."""

    def __init__(self, cfg: NeuralBiLSTMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Conv1d(cfg.num_features, cfg.d_model, 1)
        self.in_norm = nn.LayerNorm(cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.lstm = nn.LSTM(
            input_size=cfg.d_model, hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers, batch_first=True, bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(cfg.dropout)
        lstm_out = 2 * cfg.hidden_size
        if cfg.use_attention:
            # temporal + channel attention over the BiLSTM hidden sequence;
            # the embedding fuses the raw recurrent state with the attended one
            self.dba = DualBranchAttention(lstm_out, cfg.temperature, cfg.dropout)
            self.fuse = nn.Linear(2 * lstm_out, cfg.embedding_dim)
        else:
            self.fuse = nn.Linear(lstm_out, cfg.embedding_dim)
        if cfg.use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(cfg.embedding_dim, cfg.d_model), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.num_features),
            )

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x).transpose(1, 2)          # [B, T, d_model]
        h = self.in_norm(h)
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        out, _ = self.lstm(h)                           # [B, T, 2*hidden]
        out = self.drop(out)
        if self.cfg.use_attention:
            attn, _, _ = self.dba(out)                  # [B, T, 2*hidden]
            z = self.fuse(torch.cat([out, attn], dim=-1))
        else:
            z = self.fuse(out)                          # [B, T, E]
        return z.transpose(1, 2)                         # [B, E, T]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        z = self.encode(x, mask)
        out = NeuralEmbTCNOutput(embeddings=z)
        if self.cfg.use_decoder:
            out.reconstruction = self.decoder(z.transpose(1, 2)).transpose(1, 2)  # [B, N, T]
        return out
