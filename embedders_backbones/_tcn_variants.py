"""Temporal convolutional backbone family for NEMBA.

The classes here keep NEMBA's backbone contract:

    encode(x[B, C, T], mask[B, T] | None) -> z[B, E, T]
    forward(x, mask) -> output with ``reconstruction[B, C, T]``

The variants are compact PyTorch adaptations of the architecture families
commonly used around TCNs: generic residual TCNs, Lea-style ED/Dilated TCNs,
multi-stage refiners, C2F/U-Net decoders, stochastic latent TCNs,
concept-wise filters, dense/multiscale TCNs, local attention hybrids, dynamic
soft-DTW alignment filters, and PIT-style learnable temporal masks.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


@dataclass
class TCNVariantConfig:
    num_features: int = 45
    d_model: int = 96
    embedding_dim: int = 32
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    num_layers: int = 5
    num_stages: int = 3
    dropout: float = 0.15
    causal: bool = False
    max_len: int = 1024
    use_decoder: bool = True
    concept_filters: int = 1
    dense_block_layers: tuple[int, ...] = (3, 3, 3)
    dense_growth: int = 32
    multiscale_kernels: tuple[int, ...] = (3, 5, 7)
    c2f_depth: int = 4
    stochastic_latent_dim: int = 16
    stochastic_kl_weight: float = 1e-4
    stochastic_temporal_kl_weight: float = 0.0
    stochastic_dense_z: bool = True
    stochastic_dynamic_prior: bool = True
    stochastic_precision_update: bool = True
    stochastic_eval_posterior: bool = True
    pit_l1: float = 1e-5
    pit_threshold: float = 0.5
    attention_heads: int = 4
    attention_window: int = 7
    alignment_dilations: tuple[int, ...] = (1, 2, 4)
    dwa_softmax_gamma: float = 0.1

    @classmethod
    def from_dict(cls, payload: dict) -> "TCNVariantConfig":
        fields = cls.__dataclass_fields__
        values = {k: v for k, v in payload.items() if k in fields}
        for key in (
            "dilations",
            "dense_block_layers",
            "multiscale_kernels",
            "alignment_dilations",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TCNVariantOutput:
    embeddings: torch.Tensor
    reconstruction: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _channel_max_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.abs().amax(dim=1, keepdim=True) + 1e-5)


def _spatial_dropout(x: torch.Tensor, drop: nn.Module) -> torch.Tensor:
    return drop(x)


class SamePadConv1d(nn.Module):
    """Length-preserving Conv1d with causal or centered padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        causal: bool = False,
        bias: bool = True,
        weight_normed: bool = False,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.causal = bool(causal)
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            dilation=self.dilation,
            bias=bias,
            groups=groups,
        )
        self.conv = weight_norm(conv) if weight_normed else conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = (self.kernel_size - 1) * self.dilation
        if self.causal:
            left, right = total, 0
        else:
            left = total // 2
            right = total - left
        return self.conv(F.pad(x, (left, right)))


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        causal: bool = False,
        dropout: float = 0.0,
        norm: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.conv = SamePadConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, causal=causal
        )
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels) if norm else nn.Identity()
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class ReconstructionHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, out_features: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.transpose(1, 2)).transpose(1, 2)


class TCNModelBase(nn.Module):
    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.use_decoder:
            self.decoder = ReconstructionHead(
                cfg.embedding_dim, cfg.d_model, cfg.num_features, cfg.dropout
            )
        self._last_aux_loss: torch.Tensor | None = None

    def _output(self, z: torch.Tensor) -> TCNVariantOutput:
        out = TCNVariantOutput(embeddings=z, aux_loss=self._last_aux_loss)
        if self.cfg.use_decoder:
            out.reconstruction = self.decoder(z)
        return out

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> TCNVariantOutput:
        z = self.encode(x, mask)
        return self._output(z)


class ResidualTemporalBlock(nn.Module):
    """Bai et al. style residual block with two dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        *,
        causal: bool,
        weight_normed: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = SamePadConv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            causal=causal,
            weight_normed=weight_normed,
        )
        self.conv2 = SamePadConv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            causal=causal,
            weight_normed=weight_normed,
        )
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.conv1(x)))
        h = self.drop(self.act(self.conv2(h)))
        return self.act(h + self.down(x))


class GenericResidualTCN(TCNModelBase):
    """Generic residual/dilated TCN from Bai, Kolter & Koltun."""

    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__(cfg)
        channels = [cfg.d_model] * len(cfg.dilations)
        blocks: list[nn.Module] = []
        in_ch = cfg.num_features
        for out_ch, dilation in zip(channels, cfg.dilations):
            blocks.append(
                ResidualTemporalBlock(
                    in_ch,
                    out_ch,
                    cfg.kernel_size,
                    dilation,
                    cfg.dropout,
                    causal=cfg.causal,
                    weight_normed=True,
                )
            )
            in_ch = out_ch
        self.network = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(cfg.d_model, cfg.embedding_dim, 1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self._last_aux_loss = None
        return self.proj(self.network(x))


class WaveNetResidualBlock(nn.Module):
    """Lea/WaveNet-style gated dilated residual block with a skip output."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, causal: bool) -> None:
        super().__init__()
        self.filter_conv = SamePadConv1d(channels, channels, kernel_size, dilation=dilation, causal=causal)
        self.gate_conv = SamePadConv1d(channels, channels, kernel_size, dilation=dilation, causal=causal)
        self.out = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        h = self.drop(self.out(h))
        return x + h, h


class LeaDilatedTCN(TCNModelBase):
    """Dilated TCN adapted from Lea et al.'s WaveNet-style action model."""

    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__(cfg)
        k0 = 2 if cfg.causal else 3
        self.input_proj = SamePadConv1d(cfg.num_features, cfg.d_model, k0, causal=cfg.causal)
        blocks = []
        for _ in range(cfg.num_stages):
            for dilation in cfg.dilations:
                blocks.append(WaveNetResidualBlock(cfg.d_model, k0, dilation, cfg.dropout, cfg.causal))
        self.blocks = nn.ModuleList(blocks)
        self.tail = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(cfg.d_model, cfg.d_model, 1),
            nn.ReLU(),
            nn.Conv1d(cfg.d_model, cfg.embedding_dim, 1),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self._last_aux_loss = None
        h = self.input_proj(x)
        skips = []
        for block in self.blocks:
            h, skip = block(h)
            skips.append(skip)
        h = torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(max(len(skips), 1))
        return self.tail(h)


class EDConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float, causal: bool) -> None:
        super().__init__()
        self.conv = SamePadConv1d(in_ch, out_ch, kernel_size, causal=causal)
        self.drop = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _spatial_dropout(self.conv(x), self.drop)
        return _channel_max_norm(F.relu(h))


class EncoderDecoderTCN(TCNModelBase):
    """Encoder-Decoder TCN with long filters, pooling and upsampling."""

    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__(cfg)
        depth = max(1, cfg.num_layers)
        enc: list[nn.Module] = []
        in_ch = cfg.num_features
        for _ in range(depth):
            enc.append(EDConv(in_ch, cfg.d_model, cfg.kernel_size, cfg.dropout, cfg.causal))
            in_ch = cfg.d_model
        self.encoder = nn.ModuleList(enc)
        self.decoder_layers = nn.ModuleList(
            [EDConv(cfg.d_model, cfg.d_model, cfg.kernel_size, cfg.dropout, cfg.causal) for _ in range(depth)]
        )
        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        self.proj = nn.Conv1d(cfg.d_model, cfg.embedding_dim, 1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self._last_aux_loss = None
        h = x
        lengths = []
        for layer in self.encoder:
            lengths.append(h.shape[-1])
            h = layer(h)
            h = self.pool(h)
        for target_len, layer in zip(reversed(lengths), self.decoder_layers):
            h = F.interpolate(h, size=target_len, mode="nearest")
            h = layer(h)
        return self.proj(h)


class StageResidualLayer(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float, causal: bool) -> None:
        super().__init__()
        self.conv = SamePadConv1d(channels, channels, 3, dilation=dilation, causal=causal)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mix(F.relu(self.conv(x))))


class SingleStageTCN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        channels: int,
        out_ch: int,
        dilations: tuple[int, ...],
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(in_ch, channels, 1)
        self.layers = nn.ModuleList(
            [StageResidualLayer(channels, d, dropout, causal) for d in dilations]
        )
        self.out_proj = nn.Conv1d(channels, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.out_proj(h)


class MultiStageTCN(TCNModelBase):
    """MS-TCN-style repeated dilated stages for iterative refinement."""

    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__(cfg)
        stages = [
            SingleStageTCN(
                cfg.num_features,
                cfg.d_model,
                cfg.embedding_dim,
                cfg.dilations,
                cfg.dropout,
                cfg.causal,
            )
        ]
        for _ in range(max(cfg.num_stages - 1, 0)):
            stages.append(
                SingleStageTCN(
                    cfg.embedding_dim,
                    cfg.d_model,
                    cfg.embedding_dim,
                    cfg.dilations,
                    cfg.dropout,
                    cfg.causal,
                )
            )
        self.stages = nn.ModuleList(stages)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self._last_aux_loss = None
        h = self.stages[0](x)
        for stage in self.stages[1:]:
            h = stage(F.softmax(h, dim=1))
        return h


class DualDilatedResidualLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        small_dilation: int,
        large_dilation: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        self.small = SamePadConv1d(channels, channels, 3, dilation=small_dilation, causal=causal)
        self.large = SamePadConv1d(channels, channels, 3, dilation=large_dilation, causal=causal)
        self.mix = nn.Conv1d(2 * channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([F.relu(self.small(x)), F.relu(self.large(x))], dim=1)
        return x + self.drop(self.mix(h))


class DualDilatedStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        channels: int,
        out_ch: int,
        dilations: tuple[int, ...],
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_ch, channels, 1)
        reverse = tuple(reversed(dilations))
        self.layers = nn.ModuleList(
            [
                DualDilatedResidualLayer(channels, small, large, dropout, causal)
                for small, large in zip(dilations, reverse)
            ]
        )
        self.out_proj = nn.Conv1d(channels, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.out_proj(F.gelu(h))


class MultiStageTCNPlusPlus(TCNModelBase):
    """MS-TCN++-inspired stages with paired small/large dilation layers."""

    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__(cfg)
        self.stage1 = DualDilatedStage(
            cfg.num_features,
            cfg.d_model,
            cfg.embedding_dim,
            cfg.dilations,
            cfg.dropout,
            cfg.causal,
        )
        self.refiners = nn.ModuleList(
            [
                DualDilatedStage(
                    cfg.embedding_dim,
                    cfg.d_model,
                    cfg.embedding_dim,
                    cfg.dilations,
                    cfg.dropout,
                    cfg.causal,
                )
                for _ in range(max(cfg.num_stages - 1, 0))
            ]
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self._last_aux_loss = None
        h = self.stage1(x)
        for stage in self.refiners:
            h = stage(F.softmax(h, dim=1))
        return h


class DoubleConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(in_ch, out_ch, kernel_size, dropout=dropout, activation="relu"),
            ConvNormAct(out_ch, out_ch, kernel_size, dropout=dropout, activation="relu"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalPyramidPooling(nn.Module):
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.scales = (2, 3, 5, 6)
        self.proj = nn.Conv1d(in_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        pooled = []
        for scale in self.scales:
            k = min(scale, max(length, 1))
            y = F.max_pool1d(x, kernel_size=k, stride=k, ceil_mode=True)
            y = F.interpolate(self.proj(y), size=length, mode="linear", align_corners=False)
            pooled.append(y)
        return torch.cat([*pooled, x], dim=1)


class CoarseToFineTCN(TCNModelBase):
    """C2F-TCN-style temporal U-Net with pyramid bottleneck and decoder skips."""

    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__(cfg)
        depth = max(1, cfg.c2f_depth)
        self.inc = DoubleConv1d(cfg.num_features, cfg.d_model, 5, cfg.dropout)
        self.downs = nn.ModuleList(
            [DoubleConv1d(cfg.d_model, cfg.d_model, 5, cfg.dropout) for _ in range(depth)]
        )
        self.tpp = TemporalPyramidPooling(cfg.d_model)
        self.ups = nn.ModuleList()
        in_ch = cfg.d_model + 4 + cfg.d_model
        for i in range(depth):
            self.ups.append(DoubleConv1d(in_ch if i == 0 else 2 * cfg.d_model, cfg.d_model, 5, cfg.dropout))
        self.proj = nn.Conv1d(cfg.d_model, cfg.embedding_dim, 1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self._last_aux_loss = None
        h = self.inc(x)
        skips = [h]
        for down in self.downs:
            h = F.max_pool1d(h, 2, ceil_mode=True)
            h = down(h)
            skips.append(h)
        h = self.tpp(h)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = up(torch.cat([skip, h], dim=1))
        return self.proj(h)


class GaussianParamHead(nn.Module):
    """Small temporal head returning diagonal Normal parameters."""

    def __init__(self, in_ch: int, latent_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Conv1d(hidden, latent_dim, 1)
        self.logvar = nn.Conv1d(hidden, latent_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h).clamp(-8.0, 4.0)


def _normal_kl(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        p_logvar - q_logvar
        + (q_logvar.exp() + (q_mu - p_mu).pow(2)) / p_logvar.exp().clamp_min(1e-8)
        - 1.0
    )


def _precision_weighted_normal(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_prec = torch.exp(-q_logvar)
    p_prec = torch.exp(-p_logvar)
    precision = (q_prec + p_prec).clamp_min(1e-8)
    mu = (q_mu * q_prec + p_mu * p_prec) / precision
    logvar = torch.log(1.0 / precision)
    return mu, logvar.clamp(-8.0, 4.0)


class StochasticTCN(TCNModelBase):
    """STCN-style TCN with hierarchical prior/posterior Gaussian latents."""

    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__(cfg)
        self.input_proj = nn.Conv1d(cfg.num_features, cfg.d_model, 1)
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    cfg.d_model,
                    cfg.d_model,
                    cfg.kernel_size,
                    d,
                    cfg.dropout,
                    causal=cfg.causal,
                    weight_normed=False,
                )
                for d in cfg.dilations
            ]
        )
        n_layers = len(cfg.dilations)
        latent = cfg.stochastic_latent_dim
        self.posterior_heads = nn.ModuleList()
        self.prior_heads = nn.ModuleList()
        for idx in range(n_layers):
            extra = latent if idx < n_layers - 1 else 0
            self.posterior_heads.append(
                GaussianParamHead(cfg.d_model + extra, latent, cfg.d_model, cfg.dropout)
            )
            prior_extra = latent if idx < n_layers - 1 else 0
            self.prior_heads.append(
                GaussianParamHead(cfg.d_model + prior_extra, latent, cfg.d_model, cfg.dropout)
            )
        self.inject = nn.ModuleList([nn.Conv1d(latent, cfg.d_model, 1) for _ in range(n_layers)])
        z_channels = latent * n_layers if cfg.stochastic_dense_z else latent
        self.proj = nn.Conv1d(cfg.d_model + z_channels, cfg.embedding_dim, 1)
        self.latent_dists: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    @staticmethod
    def _sample(mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
        if not training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        deterministic = []
        for block in self.blocks:
            h = block(h)
            deterministic.append(h)

        q_dists: list[tuple[torch.Tensor, torch.Tensor]] = [None] * len(deterministic)  # type: ignore[list-item]
        p_dists: list[tuple[torch.Tensor, torch.Tensor]] = [None] * len(deterministic)  # type: ignore[list-item]
        samples: list[torch.Tensor] = [None] * len(deterministic)  # type: ignore[list-item]
        kls = []
        temporal_kls = []
        upper_sample: torch.Tensor | None = None

        # Top-down ladder, matching STCN's default hierarchical latent flow.
        for idx in reversed(range(len(deterministic))):
            det = deterministic[idx]
            prior_inputs = [det]
            posterior_inputs = [det]
            if upper_sample is not None:
                prior_inputs.append(upper_sample)
                posterior_inputs.append(upper_sample)

            if self.cfg.stochastic_dynamic_prior:
                p_mu, p_logvar = self.prior_heads[idx](torch.cat(prior_inputs, dim=1))
            else:
                p_mu = torch.zeros(
                    det.shape[0],
                    self.cfg.stochastic_latent_dim,
                    det.shape[-1],
                    device=det.device,
                    dtype=det.dtype,
                )
                p_logvar = torch.zeros_like(p_mu)

            q_mu_raw, q_logvar_raw = self.posterior_heads[idx](torch.cat(posterior_inputs, dim=1))
            if self.cfg.stochastic_precision_update:
                q_mu, q_logvar = _precision_weighted_normal(q_mu_raw, q_logvar_raw, p_mu, p_logvar)
            else:
                q_mu, q_logvar = q_mu_raw, q_logvar_raw

            # Training samples the posterior (ELBO). At inference the embedding
            # reads the posterior mean by default (the representation of the
            # observed data, as in a VAE encoder); set
            # stochastic_eval_posterior=False to read the prior mean instead.
            use_posterior = self.training or self.cfg.stochastic_eval_posterior
            sample_mu, sample_logvar = (q_mu, q_logvar) if use_posterior else (p_mu, p_logvar)
            z = self._sample(sample_mu, sample_logvar, self.training)

            q_dists[idx] = (q_mu, q_logvar)
            p_dists[idx] = (p_mu, p_logvar)
            samples[idx] = z
            kls.append(_normal_kl(q_mu, q_logvar, p_mu, p_logvar).mean())
            if self.cfg.stochastic_temporal_kl_weight > 0.0 and q_mu.shape[-1] > 1:
                prev_mu = F.pad(q_mu[..., :-1], (1, 0))
                prev_logvar = F.pad(q_logvar[..., :-1], (1, 0))
                temporal_kls.append(_normal_kl(q_mu, q_logvar, prev_mu.detach(), prev_logvar.detach()).mean())

            upper_sample = z
            deterministic[idx] = det + self.inject[idx](z)

        h = deterministic[-1]
        z_features = torch.cat(samples, dim=1) if self.cfg.stochastic_dense_z else samples[0]
        kl = torch.stack(kls).sum() if kls else h.new_tensor(0.0)
        aux = self.cfg.stochastic_kl_weight * kl
        if temporal_kls:
            aux = aux + self.cfg.stochastic_temporal_kl_weight * torch.stack(temporal_kls).sum()
        self._last_aux_loss = aux
        self.latent_dists = {"q": q_dists, "p": p_dists}
        return self.proj(F.gelu(torch.cat([h, z_features], dim=1)))


class ConceptWiseTemporalConv(nn.Module):
    """Shared temporal filter applied independently to every concept/channel."""

    def __init__(self, concept_filters: int, kernel_size: int, dilation: int, dropout: float, causal: bool) -> None:
        super().__init__()
        self.concept_filters = int(concept_filters)
        self.conv = SamePadConv1d(
            1, self.concept_filters, kernel_size, dilation=dilation, causal=causal
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        h = self.conv(x.reshape(batch * channels, 1, length))
        h = self.drop(F.gelu(h))
        return h.reshape(batch, channels * self.concept_filters, length)


class ConceptWiseBlock(nn.Module):
    def __init__(self, channels: int, concept_filters: int, kernel_size: int, dilation: int, dropout: float, causal: bool) -> None:
        super().__init__()
        self.ctc = ConceptWiseTemporalConv(concept_filters, kernel_size, dilation, dropout, causal)
        self.mix = nn.Conv1d(channels * concept_filters, channels, 1)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.norm(self.mix(self.ctc(x)))))
        return x + h


class ConceptWiseTCN(TCNModelBase):
    """C-TCN-style concept-wise temporal filtering before channel mixing."""

    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__(cfg)
        cf = max(1, cfg.concept_filters)
        self.first = ConceptWiseTemporalConv(cf, cfg.kernel_size, 1, cfg.dropout, cfg.causal)
        self.input_mix = nn.Conv1d(cfg.num_features * cf, cfg.d_model, 1)
        self.blocks = nn.ModuleList(
            [ConceptWiseBlock(cfg.d_model, cf, cfg.kernel_size, d, cfg.dropout, cfg.causal) for d in cfg.dilations]
        )
        self.proj = nn.Conv1d(cfg.d_model, cfg.embedding_dim, 1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self._last_aux_loss = None
        h = F.gelu(self.input_mix(self.first(x)))
        for block in self.blocks:
            h = block(h)
        return self.proj(h)


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.net(x.mean(dim=-1)).unsqueeze(-1)
        return x * gate


class DenseMultiScaleLayer(nn.Module):
    def __init__(
        self,
        in_ch: int,
        growth: int,
        kernels: tuple[int, ...],
        dilation: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        kernels = tuple(kernels) or (3,)
        branch_ch = max(growth // len(kernels), 1)
        self.branches = nn.ModuleList(
            [
                ConvNormAct(
                    in_ch,
                    branch_ch,
                    k,
                    dilation=dilation,
                    causal=causal,
                    dropout=dropout,
                    activation="gelu",
                )
                for k in kernels
            ]
        )
        self.out_ch = branch_ch * len(kernels)
        self.se = SqueezeExcite1d(in_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(features, dim=1)
        x = self.se(x)
        return self.drop(torch.cat([branch(x) for branch in self.branches], dim=1))


class DenseTemporalBlock(nn.Module):
    def __init__(
        self,
        layers: int,
        in_ch: int,
        growth: int,
        kernels: tuple[int, ...],
        dilations: tuple[int, ...],
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        mods = []
        current = in_ch
        for idx in range(layers):
            mod = DenseMultiScaleLayer(current, growth, kernels, dilations[idx % len(dilations)], dropout, causal)
            mods.append(mod)
            current += mod.out_ch
        self.layers = nn.ModuleList(mods)
        self.out_ch = current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(features))
        return torch.cat(features, dim=1)


class DenseConnectedTCN(TCNModelBase):
    """DC-TCN-style dense temporal blocks with multiscale kernels and SE gates."""

    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__(cfg)
        self.transition0 = nn.Sequential(
            nn.Conv1d(cfg.num_features, cfg.d_model, 1),
            nn.GroupNorm(_num_groups(cfg.d_model), cfg.d_model),
            nn.GELU(),
        )
        blocks = []
        transitions = []
        current = cfg.d_model
        for idx, n_layers in enumerate(cfg.dense_block_layers):
            block = DenseTemporalBlock(
                n_layers,
                current,
                cfg.dense_growth,
                cfg.multiscale_kernels,
                cfg.dilations,
                cfg.dropout,
                cfg.causal,
            )
            blocks.append(block)
            current = block.out_ch
            if idx != len(cfg.dense_block_layers) - 1:
                transitions.append(
                    nn.Sequential(
                        nn.Conv1d(current, cfg.d_model, 1),
                        nn.GroupNorm(_num_groups(cfg.d_model), cfg.d_model),
                        nn.GELU(),
                    )
                )
                current = cfg.d_model
        self.blocks = nn.ModuleList(blocks)
        self.transitions = nn.ModuleList(transitions)
        self.final_norm = nn.GroupNorm(_num_groups(current), current)
        self.proj = nn.Conv1d(current, cfg.embedding_dim, 1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self._last_aux_loss = None
        h = self.transition0(x)
        for idx, block in enumerate(self.blocks):
            h = block(h)
            if idx < len(self.transitions):
                h = self.transitions[idx](h)
        return self.proj(F.gelu(self.final_norm(h)))


class LocalNeighborhoodAttention1d(nn.Module):
    """Small dependency-free local/dilated attention for NAC-style blocks."""

    def __init__(self, channels: int, heads: int, window: int, dilation: int, causal: bool) -> None:
        super().__init__()
        heads = max(1, min(heads, channels))
        while channels % heads != 0:
            heads -= 1
        self.heads = heads
        self.dim = channels // heads
        self.window = max(1, int(window))
        self.dilation = max(1, int(dilation))
        self.causal = causal
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    @staticmethod
    def _shift(x: torch.Tensor, offset: int) -> torch.Tensor:
        length = x.shape[-1]
        if offset > 0:
            return F.pad(x, (offset, 0))[..., :length]
        if offset < 0:
            return F.pad(x, (0, -offset))[..., -offset:]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q.reshape(batch, self.heads, self.dim, length)
        k = k.reshape(batch, self.heads, self.dim, length)
        v = v.reshape(batch, self.heads, self.dim, length)
        if self.causal:
            offsets = [i * self.dilation for i in range(self.window)]
        else:
            half = self.window // 2
            offsets = [(i - half) * self.dilation for i in range(self.window)]
        k_win = torch.stack([self._shift(k, off) for off in offsets], dim=-1)
        v_win = torch.stack([self._shift(v, off) for off in offsets], dim=-1)
        score = (q.unsqueeze(-1) * k_win).sum(dim=2) / math.sqrt(self.dim)
        attn = torch.softmax(score, dim=-1)
        out = (attn.unsqueeze(2) * v_win).sum(dim=-1)
        out = out.reshape(batch, channels, length)
        return self.proj(out)


class NACBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, heads: int, window: int, causal: bool) -> None:
        super().__init__()
        self.conv = ConvNormAct(
            channels, channels, kernel_size, dilation=dilation, causal=causal, dropout=dropout
        )
        self.attn = LocalNeighborhoodAttention1d(channels, heads, window, dilation, causal)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.attn(h)
        h = self.drop(self.mix(F.gelu(self.norm(h))))
        return x + h


class NACTCN(TCNModelBase):
    """NAC-TCN-style causal/dilated local attention combined with convolutions."""

    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__(cfg)
        self.input_proj = nn.Conv1d(cfg.num_features, cfg.d_model, 1)
        self.blocks = nn.ModuleList(
            [
                NACBlock(
                    cfg.d_model,
                    cfg.kernel_size,
                    d,
                    cfg.dropout,
                    cfg.attention_heads,
                    cfg.attention_window,
                    cfg.causal,
                )
                for d in cfg.dilations
            ]
        )
        self.proj = nn.Conv1d(cfg.d_model, cfg.embedding_dim, 1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self._last_aux_loss = None
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.proj(F.gelu(h))


class SoftDTWAlignmentConv1d(nn.Module):
    """Dynamic Weight Alignment convolution using differentiable DTW paths."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dropout: float,
        causal: bool,
        gamma: float,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = int(kernel_size)
        self.causal = causal
        self.gamma = max(float(gamma), 1e-4)
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        total = self.kernel_size - 1
        if self.causal:
            return F.pad(x, (total, 0))
        left = total // 2
        return F.pad(x, (left, total - left))

    def _softmax_previous(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([a, b, c], dim=0)
        return self.gamma * torch.logsumexp(stacked / self.gamma, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows = self._pad(x).unfold(dimension=2, size=self.kernel_size, step=1)
        # Local similarity between each input time in the window and each weight time.
        score = torch.einsum("bcti,ocj->botij", windows, self.weight)
        batch, out_ch, length, k, _ = score.shape
        neg_inf = torch.full(
            (batch, out_ch, length),
            -torch.finfo(score.dtype).max / 4,
            device=score.device,
            dtype=score.dtype,
        )
        rows = [[neg_inf for _ in range(k + 1)] for _ in range(k + 1)]
        rows[0][0] = torch.zeros_like(neg_inf)
        for i in range(1, k + 1):
            for j in range(1, k + 1):
                prev = self._softmax_previous(rows[i - 1][j], rows[i][j - 1], rows[i - 1][j - 1])
                rows[i][j] = score[..., i - 1, j - 1] + prev
        out = rows[k][k] + self.bias.view(1, out_ch, 1)
        return self.drop(out)


class AlignmentResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float, causal: bool, gamma: float) -> None:
        super().__init__()
        self.align = SoftDTWAlignmentConv1d(channels, channels, kernel_size, dropout, causal, gamma)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(F.gelu(self.norm(self.align(x))))


class DynamicWeightAlignmentTCN(TCNModelBase):
    """DWA TCN using soft-DTW dynamic alignment inside temporal convolutions."""

    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__(cfg)
        self.input_proj = nn.Conv1d(cfg.num_features, cfg.d_model, 1)
        self.blocks = nn.ModuleList(
            [
                AlignmentResidualBlock(
                    cfg.d_model,
                    cfg.kernel_size,
                    cfg.dropout,
                    cfg.causal,
                    cfg.dwa_softmax_gamma,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.proj = nn.Conv1d(cfg.d_model, cfg.embedding_dim, 1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self._last_aux_loss = None
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.proj(F.gelu(h))


class PITConv1d(nn.Module):
    """PIT Conv1d with learnable time-axis masks and exportable dilation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, causal: bool) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.base_dilation = int(dilation)
        self.causal = causal
        self.conv = nn.Conv1d(in_ch, out_ch, self.kernel_size, dilation=self.base_dilation)
        self.candidate_dilations = tuple(
            2 ** i for i in range(max(1, int(math.floor(math.log2(max(self.kernel_size - 1, 1)))) + 1))
        )
        dilation_init = torch.zeros(len(self.candidate_dilations))
        dilation_init[0] = 2.0
        self.dilation_logits = nn.Parameter(dilation_init)
        self.position_logits = nn.Parameter(torch.full((self.kernel_size,), 2.0))

    def _padding(self) -> tuple[int, int]:
        total = (self.kernel_size - 1) * self.base_dilation
        if self.causal:
            return total, 0
        left = total // 2
        return left, total - left

    def dilation_probs(self) -> torch.Tensor:
        return torch.softmax(self.dilation_logits, dim=0)

    def position_gates(self) -> torch.Tensor:
        pos = torch.arange(self.kernel_size, device=self.position_logits.device)
        masks = []
        for dilation in self.candidate_dilations:
            masks.append((pos % dilation == 0).float())
        dilation_mask = torch.stack(masks, dim=0)
        expected_dilation_mask = torch.matmul(self.dilation_probs(), dilation_mask)
        return torch.sigmoid(self.position_logits) * expected_dilation_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv.weight * self.position_gates().view(1, 1, -1)
        return F.conv1d(F.pad(x, self._padding()), weight, self.conv.bias, dilation=self.base_dilation)

    def regularization(self) -> torch.Tensor:
        return self.position_gates().mean()

    def export_architecture(self, threshold: float = 0.5) -> dict:
        with torch.no_grad():
            probs = self.dilation_probs().detach().cpu()
            learned_idx = int(probs.argmax().item())
            learned_dilation = int(self.candidate_dilations[learned_idx] * self.base_dilation)
            gates = self.position_gates().detach().cpu()
            active = [idx for idx, value in enumerate(gates.tolist()) if value >= threshold]
            if 0 not in active:
                active.insert(0, 0)
            return {
                "base_dilation": int(self.base_dilation),
                "candidate_dilations": [int(d * self.base_dilation) for d in self.candidate_dilations],
                "candidate_probabilities": [float(v) for v in probs.tolist()],
                "learned_dilation": learned_dilation,
                "active_kernel_indices": active,
                "effective_kernel_size": len(active),
                "position_gates": [float(v) for v in gates.tolist()],
            }


class PITResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, causal: bool) -> None:
        super().__init__()
        self.conv1 = PITConv1d(channels, channels, kernel_size, dilation, causal)
        self.conv2 = PITConv1d(channels, channels, kernel_size, dilation, causal)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.conv1(x)))
        h = self.drop(self.mix(F.gelu(self.norm(self.conv2(h)))))
        return x + h

    def regularization(self) -> torch.Tensor:
        return self.conv1.regularization() + self.conv2.regularization()

    def export_architecture(self, threshold: float = 0.5) -> dict:
        return {
            "conv1": self.conv1.export_architecture(threshold),
            "conv2": self.conv2.export_architecture(threshold),
        }


class PITTCN(TCNModelBase):
    """PIT-style TCN with learnable time masks and discrete architecture export."""

    def __init__(self, cfg: TCNVariantConfig) -> None:
        super().__init__(cfg)
        self.input_proj = nn.Conv1d(cfg.num_features, cfg.d_model, 1)
        self.blocks = nn.ModuleList(
            [PITResidualBlock(cfg.d_model, cfg.kernel_size, d, cfg.dropout, cfg.causal) for d in cfg.dilations]
        )
        self.proj = nn.Conv1d(cfg.d_model, cfg.embedding_dim, 1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        regs = []
        for block in self.blocks:
            h = block(h)
            regs.append(block.regularization())
        reg = torch.stack(regs).sum() if regs else h.new_tensor(0.0)
        self._last_aux_loss = self.cfg.pit_l1 * reg
        return self.proj(F.gelu(h))

    def export_architecture(self, threshold: float | None = None) -> dict:
        threshold = self.cfg.pit_threshold if threshold is None else threshold
        return {
            "threshold": float(threshold),
            "blocks": [block.export_architecture(threshold) for block in self.blocks],
        }


MODEL_REGISTRY = {
    "tcn": GenericResidualTCN,
    "ed_tcn": EncoderDecoderTCN,
    "dilated_tcn": LeaDilatedTCN,
    "ms_tcn": MultiStageTCN,
    "ms_tcnpp": MultiStageTCNPlusPlus,
    "c2f_tcn": CoarseToFineTCN,
    "stcn": StochasticTCN,
    "c_tcn": ConceptWiseTCN,
    "dc_tcn": DenseConnectedTCN,
    "nac_tcn": NACTCN,
    "dwa_tcn": DynamicWeightAlignmentTCN,
    "pit_tcn": PITTCN,
}


def make_tcn_model(
    name: str,
    num_features: int,
    embedding_dim: int = 32,
    max_len: int = 512,
    **overrides,
) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown TCN variant {name!r}")
    if name in {"tcn", "nac_tcn"} and "causal" not in overrides:
        overrides["causal"] = True
    cfg = TCNVariantConfig.from_dict(
        dict(num_features=num_features, embedding_dim=embedding_dim, max_len=max_len, **overrides)
    )
    return MODEL_REGISTRY[name](cfg)
