"""EmbTCN-Attention backbone, reconstituted here and adapted for neural data.

This is a self-supervised per-frame representation model for population spike
activity. It reproduces the architecture of
``MAMIE/src/behavior_segmentation/models/embtcn_attention.py`` (a TCN input
embedding feeding a bidirectional Transformer, plus a dual-branch temporal +
channel attention module and multi-scale fusion), but with defaults tuned for
neural population data and only the self-supervised pieces kept:

  * input  x : [B, N, T]  (N neurons / features, T time bins)
  * output Z : [B, E, T]  per-frame embedding
  * decoder  : masked-input reconstruction head for SSL pretraining
  * fault    : optional per-frame novelty score (unused here)

The supervised classification / MS-TCN refinement heads of the original are
dropped: we discover neural states unsupervised by clustering Z, then test
post-hoc whether those states match behaviour.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class NeuralEmbTCNConfig:
    num_features: int = 45            # N neurons (input channels)
    d_model: int = 96
    embedding_dim: int = 32           # E
    tcn_dilations: tuple = (1, 2, 4, 8, 16)
    kernel_size: int = 5
    num_encoder_layers: int = 2
    num_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.15
    temperature: float = 0.5
    causal: bool = False              # offline -> bidirectional
    max_len: int = 1024
    use_decoder: bool = True
    use_fault_head: bool = False

    @classmethod
    def from_dict(cls, payload: dict) -> "NeuralEmbTCNConfig":
        fields = cls.__dataclass_fields__
        values = {k: v for k, v in payload.items() if k in fields}
        if "tcn_dilations" in values:
            values["tcn_dilations"] = tuple(values["tcn_dilations"])
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


def _num_groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class DilatedResidualBlock(nn.Module):
    """Centered (non-causal) dilated residual 1D conv block. [B, C, T]."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.norm(self.conv(x))))
        return x + self.res(out)


class EmbTCN(nn.Module):
    """TCN input embedding: N -> d_model via stacked dilated residual blocks."""

    def __init__(self, cfg: NeuralEmbTCNConfig) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(cfg.num_features, cfg.d_model, 1)
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(cfg.d_model, cfg.kernel_size, d, cfg.dropout)
            for d in cfg.tcn_dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return h


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class DualBranchAttention(nn.Module):
    """Temporal attention (temperature softmax) + channel SE, fused. [B, T, C]."""

    def __init__(self, d_model: int, temperature: float, dropout: float) -> None:
        super().__init__()
        self.temperature = temperature
        self.w_h = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, 1, bias=False)
        se_hidden = max(d_model // 8, 8)
        self.se = nn.Sequential(nn.Linear(d_model, se_hidden), nn.ReLU(),
                                nn.Linear(se_hidden, d_model))
        self.proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        e = self.v(torch.tanh(self.w_h(h))).squeeze(-1)
        alpha = torch.softmax(e / self.temperature, dim=1)
        h_temp = h * alpha.unsqueeze(-1)
        pooled = h.mean(dim=1)
        s = torch.sigmoid(self.se(pooled))
        h_chan = h * s.unsqueeze(1)
        fused = self.proj(torch.cat([h_temp, h_chan], dim=-1))
        out = self.norm(h + self.drop(fused))
        return out, alpha, s


@dataclass
class NeuralEmbTCNOutput:
    embeddings: torch.Tensor
    reconstruction: torch.Tensor | None = None
    fault: torch.Tensor | None = None
    temporal_weights: torch.Tensor | None = None
    channel_gate: torch.Tensor | None = None


class NeuralEmbTCN(nn.Module):
    """Backbone + SSL reconstruction head for population spike activity."""

    def __init__(self, cfg: NeuralEmbTCNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embtcn = EmbTCN(cfg)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len)
        if cfg.num_encoder_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.num_heads,
                dim_feedforward=cfg.ffn_mult * cfg.d_model, dropout=cfg.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, cfg.num_encoder_layers)
        else:
            self.encoder = nn.Identity()
        self.dba = DualBranchAttention(cfg.d_model, cfg.temperature, cfg.dropout)
        self.fuse = nn.Linear(3 * cfg.d_model, cfg.embedding_dim)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        if cfg.use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(cfg.embedding_dim, cfg.d_model), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.num_features),
            )
        if cfg.use_fault_head:
            self.fault_head = nn.Linear(cfg.embedding_dim, 1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h_tcn = self.embtcn(x)                      # [B, d_model, T]
        h = h_tcn.transpose(1, 2)                   # [B, T, d_model]
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        h = self.pos(h)
        h_enc = self.encoder(h)
        h_dba, alpha, gate = self.dba(h_enc)
        fused = self.fuse(torch.cat([h_tcn.transpose(1, 2), h_enc, h_dba], dim=-1))
        z = fused.transpose(1, 2)                   # [B, E, T]
        self._last_alpha = alpha
        self._last_gate = gate
        return z

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        z = self.encode(x, mask)
        zt = z.transpose(1, 2)                      # [B, T, E]
        out = NeuralEmbTCNOutput(
            embeddings=z,
            temporal_weights=getattr(self, "_last_alpha", None),
            channel_gate=getattr(self, "_last_gate", None),
        )
        if self.cfg.use_decoder:
            out.reconstruction = self.decoder(zt).transpose(1, 2)   # [B, N, T]
        if self.cfg.use_fault_head:
            out.fault = self.fault_head(zt).squeeze(-1)
        return out


# --------------------------------------------------------------------------- #
# SSL helpers                                                                  #
# --------------------------------------------------------------------------- #

def make_span_mask(batch: int, length: int, ratio: float, span: int,
                   device=None, generator=None) -> torch.Tensor:
    """Boolean [B, T] mask of contiguous spans covering ~ratio of frames."""
    mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
    span = max(int(span), 1)
    n_spans = max(int(round(ratio * length / span)), 1)
    for b in range(batch):
        for _ in range(n_spans):
            start = int(torch.randint(0, max(length - span, 1), (1,), generator=generator))
            mask[b, start: start + span] = True
    return mask


def masked_reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
    """MSE over masked frames only. x/x_hat [B, N, T], mask [B, T]."""
    m = mask.unsqueeze(1).float()
    se = ((x - x_hat) ** 2) * m
    return se.sum() / m.sum().clamp_min(1.0) / x.shape[1]
