"""A zoo of self-supervised sequence encoders for population spike activity.

Every model here exposes the *same* interface as ``neural_embtcn.NeuralEmbTCN``
so they all drop into the shared training / embedding / clustering pipeline:

    input  x : [B, N, T]   (N neurons, T time bins; z-scored sqrt-counts)
    encode(x, mask) -> z : [B, E, T]      per-frame embedding
    forward(x, mask) -> NeuralEmbTCNOutput with .reconstruction [B, N, T]

All are pretrained by masked reconstruction (the loss in ``neural_embtcn``),
then per-frame embeddings are clustered into neural states. The eight new
architectures requested:

  * ``TransformerAE``            - transformer auto-encoder (backbone for DEC)
  * ``VisionTransformer``        - 2D space x time patch ViT
  * ``EncoderOnlyTransformer``   - plain encoder (paired with a CRF state head)
  * ``RecurrentCNN``             - CNN front-end + bidirectional GRU (CRNN/RCNN)
  * ``TimeSeriesTransformer``    - per-timestep tokeniser (Zerveas-style TST)
  * ``SpatiotemporalTransformer``- factorised (axial) space/time attention
  * ``PatchTST``                 - channel-independent time-patch transformer
  * ``GraphAttnTransformer``     - functional neuron-graph attention + time

Plus two reusable heads used by the runner:
  * ``ClusteringLayer`` / ``target_distribution`` - Deep Embedded Clustering
  * ``LinearChainCRF``           - structured (Viterbi) state decoder

Brutal-honesty caveat: with only ~3k frames and <=45 channels per animal these
transformers are over-parameterised. We keep them small, regularise hard, and
trust the cross-validated metrics rather than the training loss.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._embtcn import NeuralEmbTCNOutput, PositionalEncoding, _num_groups


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #

def _encoder_layer(d_model: int, nhead: int, ffn_mult: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=ffn_mult * d_model,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )


def _safe_heads(d_model: int, nhead: int) -> int:
    """Largest head count <= nhead that divides d_model."""
    for h in range(min(nhead, d_model), 0, -1):
        if d_model % h == 0:
            return h
    return 1


def _apply_frame_mask(h: torch.Tensor, mask: torch.Tensor | None, token: torch.Tensor) -> torch.Tensor:
    """Replace masked frames [B,T] with a learned token in h [B,T,d]."""
    if mask is None:
        return h
    return torch.where(mask.unsqueeze(-1), token.view(1, 1, -1), h)


# --------------------------------------------------------------------------- #
# 1. Transformer auto-encoder (backbone for Deep Embedded Clustering)         #
# --------------------------------------------------------------------------- #

@dataclass
class TransformerAEConfig:
    num_features: int = 45
    d_model: int = 96
    embedding_dim: int = 32
    num_encoder_layers: int = 3
    num_decoder_layers: int = 2
    num_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.15
    max_len: int = 512
    use_decoder: bool = True

    @classmethod
    def from_dict(cls, p: dict) -> "TransformerAEConfig":
        f = cls.__dataclass_fields__
        return cls(**{k: v for k, v in p.items() if k in f})

    def to_dict(self) -> dict:
        return asdict(self)


class TransformerAE(nn.Module):
    """Symmetric transformer auto-encoder; bottleneck is the per-frame embedding."""

    def __init__(self, cfg: TransformerAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = _safe_heads(cfg.d_model, cfg.num_heads)
        self.input_proj = nn.Conv1d(cfg.num_features, cfg.d_model, 1)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.encoder = nn.TransformerEncoder(
            _encoder_layer(cfg.d_model, h, cfg.ffn_mult, cfg.dropout), cfg.num_encoder_layers)
        self.to_emb = nn.Linear(cfg.d_model, cfg.embedding_dim)
        if cfg.use_decoder:
            self.from_emb = nn.Linear(cfg.embedding_dim, cfg.d_model)
            self.decoder = nn.TransformerEncoder(
                _encoder_layer(cfg.d_model, h, cfg.ffn_mult, cfg.dropout), cfg.num_decoder_layers)
            self.out = nn.Linear(cfg.d_model, cfg.num_features)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x).transpose(1, 2)          # [B,T,d]
        h = _apply_frame_mask(h, mask, self.mask_token)
        h = self.pos(h)
        h = self.encoder(h)
        z = self.to_emb(h)                              # [B,T,E]
        return z.transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        z = self.encode(x, mask)
        out = NeuralEmbTCNOutput(embeddings=z)
        if self.cfg.use_decoder:
            h = self.pos(self.from_emb(z.transpose(1, 2)))
            h = self.decoder(h)
            out.reconstruction = self.out(h).transpose(1, 2)
        return out


# --------------------------------------------------------------------------- #
# 2. Vision Transformer (2D space x time patches)                            #
# --------------------------------------------------------------------------- #

@dataclass
class ViTConfig:
    num_features: int = 45
    d_model: int = 96
    embedding_dim: int = 32
    n_groups: int = 5                # neuron groups (spatial patches)
    patch_len: int = 8              # time patch length
    num_encoder_layers: int = 3
    num_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.15
    max_len: int = 512
    use_decoder: bool = True

    @classmethod
    def from_dict(cls, p: dict) -> "ViTConfig":
        f = cls.__dataclass_fields__
        return cls(**{k: v for k, v in p.items() if k in f})

    def to_dict(self) -> dict:
        return asdict(self)


class VisionTransformer(nn.Module):
    """Treat the [N,T] activity window as an image; tokenise a 2D grid of
    (neuron-group x time-patch) blocks, run a transformer over the grid with
    learned 2D positional embeddings, and reconstruct masked patches."""

    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.G = cfg.n_groups
        self.P = cfg.patch_len
        self.npg = int(math.ceil(cfg.num_features / self.G))      # neurons / group
        self.Npad = self.npg * self.G                             # padded neuron count
        self.patch_dim = self.npg * self.P
        self.max_tp = int(math.ceil(cfg.max_len / self.P))        # max time-patches
        h = _safe_heads(cfg.d_model, cfg.num_heads)

        self.embed = nn.Linear(self.patch_dim, cfg.d_model)
        self.row_pos = nn.Parameter(torch.zeros(1, self.G, 1, cfg.d_model))
        self.col_pos = nn.Parameter(torch.zeros(1, 1, self.max_tp, cfg.d_model))
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.row_pos, std=0.02)
        nn.init.normal_(self.col_pos, std=0.02)
        self.encoder = nn.TransformerEncoder(
            _encoder_layer(cfg.d_model, h, cfg.ffn_mult, cfg.dropout), cfg.num_encoder_layers)
        self.to_emb = nn.Linear(cfg.d_model, cfg.embedding_dim)
        if cfg.use_decoder:
            self.unembed = nn.Linear(cfg.d_model, self.patch_dim)

    def _patchify(self, x: torch.Tensor):
        """[B,N,T] -> tokens [B, G*ntp, patch_dim], grid dims (G, ntp)."""
        B, N, T = x.shape
        Tpad = int(math.ceil(T / self.P)) * self.P
        ntp = Tpad // self.P
        xp = x
        if N < self.Npad:
            xp = F.pad(xp, (0, 0, 0, self.Npad - N))
        if Tpad > T:
            xp = F.pad(xp, (0, Tpad - T))
        # [B, G, npg, ntp, P]
        xp = xp.view(B, self.G, self.npg, ntp, self.P)
        # tokens ordered (g, tp): [B, G, ntp, npg*P]
        tok = xp.permute(0, 1, 3, 2, 4).reshape(B, self.G, ntp, self.patch_dim)
        return tok, ntp, T, N

    def _grid_pos(self, ntp: int) -> torch.Tensor:
        return (self.row_pos + self.col_pos[:, :, :ntp]).reshape(1, self.G * ntp, -1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, T = x.shape
        tok, ntp, T0, N0 = self._patchify(x)
        h = self.embed(tok.reshape(B, self.G * ntp, self.patch_dim))
        if mask is not None:
            # frame mask [B,T] -> time-patch mask [B,ntp] (>=50% frames masked)
            mt = F.pad(mask.float(), (0, ntp * self.P - T)).view(B, ntp, self.P).mean(-1) >= 0.5
            mtg = mt.unsqueeze(1).expand(B, self.G, ntp).reshape(B, self.G * ntp)
            h = torch.where(mtg.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        h = h + self._grid_pos(ntp)
        h = self.encoder(h)
        h = h.view(B, self.G, ntp, -1)
        z_tp = self.to_emb(h.mean(dim=1))               # pool groups -> [B,ntp,E]
        z = z_tp.repeat_interleave(self.P, dim=1)[:, :T]  # broadcast to frames
        return z.transpose(1, 2)                         # [B,E,T]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        B, N, T = x.shape
        tok, ntp, T0, N0 = self._patchify(x)
        h = self.embed(tok.reshape(B, self.G * ntp, self.patch_dim))
        if mask is not None:
            mt = F.pad(mask.float(), (0, ntp * self.P - T)).view(B, ntp, self.P).mean(-1) >= 0.5
            mtg = mt.unsqueeze(1).expand(B, self.G, ntp).reshape(B, self.G * ntp)
            h = torch.where(mtg.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        h = h + self._grid_pos(ntp)
        henc = self.encoder(h)
        z_tp = self.to_emb(henc.view(B, self.G, ntp, -1).mean(dim=1))
        z = z_tp.repeat_interleave(self.P, dim=1)[:, :T].transpose(1, 2)
        out = NeuralEmbTCNOutput(embeddings=z)
        if self.cfg.use_decoder:
            rec = self.unembed(henc)                    # [B, G*ntp, patch_dim]
            rec = rec.view(B, self.G, ntp, self.npg, self.P)
            rec = rec.permute(0, 1, 3, 2, 4).reshape(B, self.Npad, ntp * self.P)
            out.reconstruction = rec[:, :N, :T]
        return out


# --------------------------------------------------------------------------- #
# 3. Encoder-only transformer (paired with the CRF state head)               #
# --------------------------------------------------------------------------- #

@dataclass
class EncoderOnlyConfig:
    num_features: int = 45
    d_model: int = 96
    embedding_dim: int = 32
    num_encoder_layers: int = 3
    num_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.15
    max_len: int = 512
    kernel_size: int = 5
    use_decoder: bool = True

    @classmethod
    def from_dict(cls, p: dict) -> "EncoderOnlyConfig":
        f = cls.__dataclass_fields__
        return cls(**{k: v for k, v in p.items() if k in f})

    def to_dict(self) -> dict:
        return asdict(self)


class EncoderOnlyTransformer(nn.Module):
    """Conv tokeniser + sinusoidal-pos transformer encoder. The discrete state
    sequence is refined post-hoc by ``LinearChainCRF`` (handled in the runner)."""

    def __init__(self, cfg: EncoderOnlyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = _safe_heads(cfg.d_model, cfg.num_heads)
        pad = (cfg.kernel_size - 1) // 2
        self.tokeniser = nn.Sequential(
            nn.Conv1d(cfg.num_features, cfg.d_model, cfg.kernel_size, padding=pad),
            nn.GroupNorm(_num_groups(cfg.d_model), cfg.d_model), nn.GELU())
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.encoder = nn.TransformerEncoder(
            _encoder_layer(cfg.d_model, h, cfg.ffn_mult, cfg.dropout), cfg.num_encoder_layers)
        self.to_emb = nn.Linear(cfg.d_model, cfg.embedding_dim)
        if cfg.use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(cfg.embedding_dim, cfg.d_model), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.num_features))

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.tokeniser(x).transpose(1, 2)
        h = _apply_frame_mask(h, mask, self.mask_token)
        h = self.encoder(self.pos(h))
        return self.to_emb(h).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        z = self.encode(x, mask)
        out = NeuralEmbTCNOutput(embeddings=z)
        if self.cfg.use_decoder:
            out.reconstruction = self.decoder(z.transpose(1, 2)).transpose(1, 2)
        return out


# --------------------------------------------------------------------------- #
# 4. Recurrent CNN (CNN front-end + bidirectional GRU)                       #
# --------------------------------------------------------------------------- #

@dataclass
class RecurrentCNNConfig:
    num_features: int = 45
    d_model: int = 96
    embedding_dim: int = 32
    hidden_size: int = 96
    num_layers: int = 2
    conv_channels: int = 96
    kernel_size: int = 5
    n_conv: int = 2
    dropout: float = 0.15
    use_decoder: bool = True

    @classmethod
    def from_dict(cls, p: dict) -> "RecurrentCNNConfig":
        f = cls.__dataclass_fields__
        return cls(**{k: v for k, v in p.items() if k in f})

    def to_dict(self) -> dict:
        return asdict(self)


class RecurrentCNN(nn.Module):
    """Stacked dilated 1D-CNN feature extractor feeding a bidirectional GRU."""

    def __init__(self, cfg: RecurrentCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers = []
        cin = cfg.num_features
        for i in range(cfg.n_conv):
            dil = 2 ** i
            pad = (cfg.kernel_size - 1) // 2 * dil
            layers += [nn.Conv1d(cin, cfg.conv_channels, cfg.kernel_size, padding=pad, dilation=dil),
                       nn.GroupNorm(_num_groups(cfg.conv_channels), cfg.conv_channels),
                       nn.GELU(), nn.Dropout(cfg.dropout)]
            cin = cfg.conv_channels
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(cfg.conv_channels, cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.gru = nn.GRU(cfg.d_model, cfg.hidden_size, cfg.num_layers, batch_first=True,
                          bidirectional=True, dropout=cfg.dropout if cfg.num_layers > 1 else 0.0)
        self.to_emb = nn.Linear(2 * cfg.hidden_size, cfg.embedding_dim)
        if cfg.use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(cfg.embedding_dim, cfg.d_model), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.num_features))

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj(self.cnn(x).transpose(1, 2))      # [B,T,d]
        h = _apply_frame_mask(h, mask, self.mask_token)
        out, _ = self.gru(h)
        return self.to_emb(out).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        z = self.encode(x, mask)
        out = NeuralEmbTCNOutput(embeddings=z)
        if self.cfg.use_decoder:
            out.reconstruction = self.decoder(z.transpose(1, 2)).transpose(1, 2)
        return out


# --------------------------------------------------------------------------- #
# 5. Time-Series Transformer (Zerveas-style per-timestep tokeniser)          #
# --------------------------------------------------------------------------- #

@dataclass
class TSTConfig:
    num_features: int = 45
    d_model: int = 96
    embedding_dim: int = 32
    num_encoder_layers: int = 3
    num_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.15
    max_len: int = 512
    use_decoder: bool = True

    @classmethod
    def from_dict(cls, p: dict) -> "TSTConfig":
        f = cls.__dataclass_fields__
        return cls(**{k: v for k, v in p.items() if k in f})

    def to_dict(self) -> dict:
        return asdict(self)


class TimeSeriesTransformer(nn.Module):
    """Per-timestep linear tokeniser + learnable positional embedding + encoder,
    pretrained by value-level masked reconstruction (Zerveas et al. 2021)."""

    def __init__(self, cfg: TSTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = _safe_heads(cfg.d_model, cfg.num_heads)
        self.tokeniser = nn.Linear(cfg.num_features, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.encoder = nn.TransformerEncoder(
            _encoder_layer(cfg.d_model, h, cfg.ffn_mult, cfg.dropout), cfg.num_encoder_layers)
        self.to_emb = nn.Linear(cfg.d_model, cfg.embedding_dim)
        if cfg.use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.num_features))

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, T = x.shape
        h = self.tokeniser(x.transpose(1, 2))           # [B,T,d]
        h = _apply_frame_mask(h, mask, self.mask_token)
        h = self.norm(h + self.pos_emb[:, :T])
        h = self.encoder(h)
        return self.to_emb(h).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        B, N, T = x.shape
        h = self.tokeniser(x.transpose(1, 2))
        h = _apply_frame_mask(h, mask, self.mask_token)
        h = self.encoder(self.norm(h + self.pos_emb[:, :T]))
        z = self.to_emb(h).transpose(1, 2)
        out = NeuralEmbTCNOutput(embeddings=z)
        if self.cfg.use_decoder:
            out.reconstruction = self.decoder(h).transpose(1, 2)
        return out


# --------------------------------------------------------------------------- #
# 6. Spatiotemporal transformer (factorised / axial attention)               #
# --------------------------------------------------------------------------- #

@dataclass
class SpatiotemporalConfig:
    num_features: int = 45
    d_model: int = 48
    embedding_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    ffn_mult: int = 2
    dropout: float = 0.15
    max_len: int = 512
    use_decoder: bool = True

    @classmethod
    def from_dict(cls, p: dict) -> "SpatiotemporalConfig":
        f = cls.__dataclass_fields__
        return cls(**{k: v for k, v in p.items() if k in f})

    def to_dict(self) -> dict:
        return asdict(self)


class _AxialBlock(nn.Module):
    """One temporal-attention pass (over T, per neuron) then one spatial pass
    (over N, per frame)."""

    def __init__(self, d: int, nhead: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        h = _safe_heads(d, nhead)
        self.t_attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.t_norm = nn.LayerNorm(d)
        self.s_attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.s_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_mult * d, d))
        self.f_norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:   # h [B,N,T,d]
        B, N, T, d = h.shape
        ht = h.reshape(B * N, T, d)
        a, _ = self.t_attn(self.t_norm(ht), self.t_norm(ht), self.t_norm(ht))
        ht = ht + a
        h = ht.reshape(B, N, T, d)
        hs = h.permute(0, 2, 1, 3).reshape(B * T, N, d)
        a, _ = self.s_attn(self.s_norm(hs), self.s_norm(hs), self.s_norm(hs))
        hs = hs + a
        h = hs.reshape(B, T, N, d).permute(0, 2, 1, 3)
        h = h + self.ffn(self.f_norm(h))
        return h


class SpatiotemporalTransformer(nn.Module):
    """Embed every (neuron, frame) entry, then alternate attention along the
    time axis and the neuron axis; pool neurons for the per-frame embedding."""

    def __init__(self, cfg: SpatiotemporalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.val_proj = nn.Linear(1, d)
        self.space_emb = nn.Parameter(torch.zeros(1, cfg.num_features, 1, d))
        self.time_pe = PositionalEncoding(d, cfg.max_len)
        self.mask_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.space_emb, std=0.02)
        self.blocks = nn.ModuleList([
            _AxialBlock(d, cfg.num_heads, cfg.ffn_mult, cfg.dropout) for _ in range(cfg.num_layers)])
        self.to_emb = nn.Linear(d, cfg.embedding_dim)
        if cfg.use_decoder:
            self.out = nn.Linear(d, 1)

    def _embed(self, x: torch.Tensor, mask: torch.Tensor | None):
        B, N, T = x.shape
        h = self.val_proj(x.unsqueeze(-1))              # [B,N,T,d]
        h = h + self.space_emb[:, :N]
        # temporal positional encoding broadcast over neurons
        pe = self.time_pe(torch.zeros(1, T, h.shape[-1], device=h.device))
        h = h + pe.unsqueeze(1)
        if mask is not None:
            m = mask.view(B, 1, T, 1)
            h = torch.where(m, self.mask_token.view(1, 1, 1, -1), h)
        for blk in self.blocks:
            h = blk(h)
        return h

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self._embed(x, mask)                         # [B,N,T,d]
        z = self.to_emb(h.mean(dim=1))                   # pool neurons -> [B,T,E]
        return z.transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        h = self._embed(x, mask)
        z = self.to_emb(h.mean(dim=1)).transpose(1, 2)
        out = NeuralEmbTCNOutput(embeddings=z)
        if self.cfg.use_decoder:
            out.reconstruction = self.out(h).squeeze(-1)   # [B,N,T]
        return out


# --------------------------------------------------------------------------- #
# 7. PatchTST (channel-independent time-patch transformer)                    #
# --------------------------------------------------------------------------- #

@dataclass
class PatchTSTConfig:
    num_features: int = 45
    d_model: int = 64
    embedding_dim: int = 32
    patch_len: int = 8
    num_encoder_layers: int = 3
    num_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.15
    max_len: int = 512
    use_decoder: bool = True

    @classmethod
    def from_dict(cls, p: dict) -> "PatchTSTConfig":
        f = cls.__dataclass_fields__
        return cls(**{k: v for k, v in p.items() if k in f})

    def to_dict(self) -> dict:
        return asdict(self)


class PatchTST(nn.Module):
    """Each neuron is patched into non-overlapping time patches and encoded by a
    transformer with weights *shared across channels* (channel independence)."""

    def __init__(self, cfg: PatchTSTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.P = cfg.patch_len
        self.max_np = int(math.ceil(cfg.max_len / self.P))
        h = _safe_heads(cfg.d_model, cfg.num_heads)
        self.embed = nn.Linear(self.P, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_np, cfg.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.encoder = nn.TransformerEncoder(
            _encoder_layer(cfg.d_model, h, cfg.ffn_mult, cfg.dropout), cfg.num_encoder_layers)
        self.to_emb = nn.Linear(cfg.d_model, cfg.embedding_dim)
        if cfg.use_decoder:
            self.unembed = nn.Linear(cfg.d_model, self.P)

    def _patch(self, x: torch.Tensor):
        B, N, T = x.shape
        Tpad = int(math.ceil(T / self.P)) * self.P
        if Tpad > T:
            x = F.pad(x, (0, Tpad - T))
        npatch = Tpad // self.P
        p = x.view(B, N, npatch, self.P)                # [B,N,np,P]
        return p, npatch, T

    def _run(self, x: torch.Tensor, mask: torch.Tensor | None):
        B, N, T = x.shape
        p, npatch, T0 = self._patch(x)
        tok = self.embed(p)                             # [B,N,np,d]
        tok = tok.reshape(B * N, npatch, -1)
        if mask is not None:
            mp = F.pad(mask.float(), (0, npatch * self.P - T)).view(B, npatch, self.P).mean(-1) >= 0.5
            mp = mp.unsqueeze(1).expand(B, N, npatch).reshape(B * N, npatch)
            tok = torch.where(mp.unsqueeze(-1), self.mask_token.view(1, 1, -1), tok)
        tok = tok + self.pos_emb[:, :npatch]
        henc = self.encoder(tok).reshape(B, N, npatch, -1)
        return henc, npatch, T0

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, T = x.shape
        henc, npatch, T0 = self._run(x, mask)
        z_p = self.to_emb(henc.mean(dim=1))             # pool channels -> [B,np,E]
        z = z_p.repeat_interleave(self.P, dim=1)[:, :T]
        return z.transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        B, N, T = x.shape
        henc, npatch, T0 = self._run(x, mask)
        z_p = self.to_emb(henc.mean(dim=1))
        z = z_p.repeat_interleave(self.P, dim=1)[:, :T].transpose(1, 2)
        out = NeuralEmbTCNOutput(embeddings=z)
        if self.cfg.use_decoder:
            rec = self.unembed(henc).reshape(B, N, npatch * self.P)
            out.reconstruction = rec[:, :, :T]
        return out


# --------------------------------------------------------------------------- #
# 8. Graph-attention transformer (functional neuron graph)                    #
# --------------------------------------------------------------------------- #

@dataclass
class GraphAttnConfig:
    num_features: int = 45
    d_model: int = 48
    embedding_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    ffn_mult: int = 2
    dropout: float = 0.15
    max_len: int = 512
    use_decoder: bool = True

    @classmethod
    def from_dict(cls, p: dict) -> "GraphAttnConfig":
        f = cls.__dataclass_fields__
        return cls(**{k: v for k, v in p.items() if k in f})

    def to_dict(self) -> dict:
        return asdict(self)


def neuron_graph(X: np.ndarray, top_k: int = 8) -> np.ndarray:
    """Symmetric k-NN functional-connectivity mask [N,N] from |correlation| of
    the population activity X [N,T]. Diagonal always on (self-loops)."""
    N = X.shape[0]
    C = np.corrcoef(X)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    A = np.zeros((N, N), dtype=bool)
    k = int(min(top_k, max(N - 1, 1)))
    for i in range(N):
        nb = np.argsort(np.abs(C[i]))[::-1][:k]
        A[i, nb] = True
    A = A | A.T
    np.fill_diagonal(A, True)
    return A


class _GraphBlock(nn.Module):
    """Graph-masked spatial attention over neurons, then temporal attention."""

    def __init__(self, d: int, nhead: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        h = _safe_heads(d, nhead)
        self.g_attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.g_norm = nn.LayerNorm(d)
        self.t_attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.t_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_mult * d, d))
        self.f_norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, N, T, d = h.shape
        # spatial: attend over neurons, restricted to graph edges
        hs = h.permute(0, 2, 1, 3).reshape(B * T, N, d)
        hn = self.g_norm(hs)
        a, _ = self.g_attn(hn, hn, hn, attn_mask=attn_mask)
        hs = hs + a
        h = hs.reshape(B, T, N, d).permute(0, 2, 1, 3)
        # temporal: attend over time per neuron
        ht = h.reshape(B * N, T, d)
        hn = self.t_norm(ht)
        a, _ = self.t_attn(hn, hn, hn)
        ht = ht + a
        h = ht.reshape(B, N, T, d)
        h = h + self.ffn(self.f_norm(h))
        return h


class GraphAttnTransformer(nn.Module):
    """Functional neuron-graph attention interleaved with temporal attention.

    Call ``set_graph(A)`` with a boolean [N,N] adjacency before use; otherwise a
    fully-connected graph is assumed."""

    def __init__(self, cfg: GraphAttnConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.val_proj = nn.Linear(1, d)
        self.space_emb = nn.Parameter(torch.zeros(1, cfg.num_features, 1, d))
        self.time_pe = PositionalEncoding(d, cfg.max_len)
        self.mask_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.space_emb, std=0.02)
        self.blocks = nn.ModuleList([
            _GraphBlock(d, cfg.num_heads, cfg.ffn_mult, cfg.dropout) for _ in range(cfg.num_layers)])
        self.to_emb = nn.Linear(d, cfg.embedding_dim)
        if cfg.use_decoder:
            self.out = nn.Linear(d, 1)
        # default: fully connected (no masking)
        self.register_buffer("attn_bias", torch.zeros(cfg.num_features, cfg.num_features))

    def set_graph(self, A: np.ndarray) -> None:
        """Store additive attention bias: 0 on edges, -inf off edges."""
        At = torch.from_numpy(np.asarray(A, dtype=bool))
        bias = torch.where(At, torch.zeros_like(At, dtype=torch.float32),
                           torch.full_like(At, float("-inf"), dtype=torch.float32))
        self.attn_bias = bias.to(self.attn_bias.device)

    def _embed(self, x: torch.Tensor, mask: torch.Tensor | None):
        B, N, T = x.shape
        h = self.val_proj(x.unsqueeze(-1)) + self.space_emb[:, :N]
        pe = self.time_pe(torch.zeros(1, T, h.shape[-1], device=h.device))
        h = h + pe.unsqueeze(1)
        if mask is not None:
            h = torch.where(mask.view(B, 1, T, 1), self.mask_token.view(1, 1, 1, -1), h)
        amask = self.attn_bias[:N, :N]
        for blk in self.blocks:
            h = blk(h, amask)
        return h

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self._embed(x, mask)
        return self.to_emb(h.mean(dim=1)).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralEmbTCNOutput:
        h = self._embed(x, mask)
        z = self.to_emb(h.mean(dim=1)).transpose(1, 2)
        out = NeuralEmbTCNOutput(embeddings=z)
        if self.cfg.use_decoder:
            out.reconstruction = self.out(h).squeeze(-1)
        return out


# --------------------------------------------------------------------------- #
# Deep Embedded Clustering head                                                #
# --------------------------------------------------------------------------- #

class ClusteringLayer(nn.Module):
    """Student-t soft cluster assignment (DEC). q_ij over K learnable centroids."""

    def __init__(self, n_clusters: int, emb_dim: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.centroids = nn.Parameter(torch.randn(n_clusters, emb_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:    # z [M,E] -> q [M,K]
        d2 = torch.cdist(z, self.centroids) ** 2
        q = (1.0 + d2 / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        return q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """DEC target P: sharpen Q by normalising squared assignments by cluster freq."""
    w = q ** 2 / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)


class DECAutoencoder(nn.Module):
    """Small MLP auto-encoder whose bottleneck is clustering-friendly (DEC).
    Operates on per-frame transformer features H [M, F]."""

    def __init__(self, in_dim: int, bottleneck: int = 16, hidden: int = 64,
                 n_clusters: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck))
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, in_dim))
        self.cluster = ClusteringLayer(n_clusters, bottleneck)

    def forward(self, h: torch.Tensor):
        z = self.encoder(h)
        return z, self.decoder(z), self.cluster(z)


# --------------------------------------------------------------------------- #
# Linear-chain CRF (structured state decoder)                                 #
# --------------------------------------------------------------------------- #

class LinearChainCRF(nn.Module):
    """Linear-chain CRF over a single sequence. Emissions come from a linear head
    on the per-frame embeddings; transitions are learned. Trained to fit the
    KMeans pseudo-labels, then Viterbi-decoded for a temporally coherent state
    sequence (the transformer analogue of the HMM's forward-backward)."""

    def __init__(self, emb_dim: int, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.emit = nn.Linear(emb_dim, num_tags)
        self.trans = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        self.start = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end = nn.Parameter(torch.randn(num_tags) * 0.1)

    def _scores(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        T = emissions.shape[0]
        score = self.start[tags[0]] + emissions[0, tags[0]]
        for t in range(1, T):
            score = score + self.trans[tags[t - 1], tags[t]] + emissions[t, tags[t]]
        return score + self.end[tags[-1]]

    def _log_partition(self, emissions: torch.Tensor) -> torch.Tensor:
        T = emissions.shape[0]
        alpha = self.start + emissions[0]               # [K]
        for t in range(1, T):
            alpha = torch.logsumexp(
                alpha.unsqueeze(1) + self.trans + emissions[t].unsqueeze(0), dim=0)
        return torch.logsumexp(alpha + self.end, dim=0)

    def nll(self, z: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        emissions = self.emit(z)                        # [T,K]
        return self._log_partition(emissions) - self._scores(emissions, tags)

    @torch.no_grad()
    def viterbi(self, z: torch.Tensor) -> np.ndarray:
        emissions = self.emit(z)
        T = emissions.shape[0]
        score = self.start + emissions[0]
        backptr = []
        for t in range(1, T):
            m = score.unsqueeze(1) + self.trans         # [K,K]
            best, idx = m.max(dim=0)
            score = best + emissions[t]
            backptr.append(idx)
        score = score + self.end
        best_last = int(score.argmax())
        path = [best_last]
        for idx in reversed(backptr):
            best_last = int(idx[best_last])
            path.append(best_last)
        return np.asarray(path[::-1], dtype=int)


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #

MODEL_REGISTRY = {
    "dtc": (TransformerAE, TransformerAEConfig),
    "vit": (VisionTransformer, ViTConfig),
    "enc_crf": (EncoderOnlyTransformer, EncoderOnlyConfig),
    "rcnn": (RecurrentCNN, RecurrentCNNConfig),
    "tst": (TimeSeriesTransformer, TSTConfig),
    "stt": (SpatiotemporalTransformer, SpatiotemporalConfig),
    "patchtst": (PatchTST, PatchTSTConfig),
    "gatr": (GraphAttnTransformer, GraphAttnConfig),
}

MODEL_LABELS = {
    "dtc": "Transformer-AE + DeepCluster",
    "vit": "Vision Transformer",
    "enc_crf": "Encoder + CRF",
    "rcnn": "Recurrent CNN",
    "tst": "Time-Series Transformer",
    "stt": "Spatiotemporal Transformer",
    "patchtst": "PatchTST",
    "gatr": "Graph-Attn Transformer",
}
