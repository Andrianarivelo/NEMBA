# TCN Variant Audit

NEMBA exposes these TCN-family backbones through the same self-supervised
masked-reconstruction interface as the existing models:

```text
encode(x[B, C, T], mask[B, T] | None) -> z[B, E, T]
forward(x, mask) -> reconstruction[B, C, T]
```

This means every paper-specific classification, segmentation, sequence
likelihood, smoothing, or architecture-search head is replaced by NEMBA's common
embedding projection and reconstruction head. The table below separates the
source architecture from the NEMBA adaptation.

The source taxonomy is cross-checked against the Emergent Mind TCN review:
<https://www.emergentmind.com/topics/temporal-convolutional-networks>

| key | citation | source-defining architecture | audit result in NEMBA |
|---|---|---|---|
| `tcn` | [Bai, Kolter & Koltun, 2018](https://arxiv.org/abs/1803.01271) | Causal, length-preserving residual TCN blocks with dilated convolutions, weight norm, dropout, and residual 1x1 matching. | Core faithful. `GenericResidualTCN` uses two weight-normalized dilated convs per residual block, residual projection, dropout, and defaults to `causal=True` unless explicitly overridden. |
| `ed_tcn` | [Lea et al., 2016](https://arxiv.org/abs/1611.05267) | Encoder-decoder TCN: temporal conv, spatial dropout, ReLU/channel normalization, max-pooling encoder, upsampling decoder. | Core faithful. `EncoderDecoderTCN` keeps the conv/pool/upsample/channel-normalized backbone and replaces the framewise softmax classifier with an embedding projection. |
| `dilated_tcn` | [Lea et al., 2016](https://arxiv.org/abs/1611.05267) | WaveNet-inspired dilated temporal blocks, exponentially increasing dilations, residual update, and skip aggregation. | Core faithful adaptation. `LeaDilatedTCN` uses gated dilated residual blocks and aggregates skip outputs before the embedding projection. The supervised temporal softmax output is replaced. |
| `ms_tcn` | [Abu Farha & Gall, 2019](https://arxiv.org/abs/1903.01945) | Multi-stage temporal segmentation: first stage predicts framewise probabilities; later stages refine the previous stage using dilated residual layers. | Latent faithful adaptation. `MultiStageTCN` now emits embedding logits from stage 1 and feeds softmax-normalized latent predictions into refinement stages, preserving the staged refinement pattern without labels. |
| `ms_tcnpp` | [Li et al., 2020](https://arxiv.org/abs/2006.09220) | MS-TCN extension with dual-dilated layers combining small and large receptive fields, plus separate generation/refinement-stage design. | Latent faithful adaptation. `MultiStageTCNPlusPlus` uses staged latent refinement and dual-dilated small/large branches. Classification and smoothing losses are replaced by masked reconstruction. |
| `c2f_tcn` | [Singhania, Rahaman & Yao, 2021](https://arxiv.org/abs/2105.10859) | Coarse-to-fine temporal encoder-decoder/U-Net with temporal pyramid pooling and multiple decoder predictions at increasing resolutions. | Backbone faithful adaptation. `CoarseToFineTCN` keeps the temporal U-Net, decoder skips, and pyramid bottleneck. Auxiliary decoder classifiers, feature augmentation, and action loss are not included. |
| `stcn` | [Aksan & Hilliges, 2019](https://arxiv.org/abs/1902.06568) | Stochastic TCN with a hierarchy of latent variables at multiple temporal scales, trained as a generative latent-variable model. | Source-faithful embedding adaptation. `StochasticTCN` now builds dynamic priors and approximate posteriors at each deterministic level, uses precision-weighted posterior updates, samples from posterior during training and prior at eval, stores `latent_dists["q"/"p"]`, and adds KL(q\|\|p) as the ELBO auxiliary term. The output likelihood is NEMBA's masked reconstruction head. |
| `c_tcn` | [Li et al., 2019](https://arxiv.org/abs/1908.09442) | Concept-wise temporal convolution: apply temporal filters independently to each latent concept/channel with shared filter parameters, then mix. | Core faithful. `ConceptWiseTCN` applies shared per-channel temporal filters and 1x1 mixing, allowing deeper temporal filtering while limiting early concept recombination. |
| `dc_tcn` | [Ma et al., 2020](https://arxiv.org/abs/2009.14233) | Dense temporal blocks for denser receptive-field coverage, multiscale temporal kernels, transitions, and squeeze-excitation attention. | Core faithful adaptation. `DenseConnectedTCN` keeps dense concatenation, multiscale kernel branches, transition projections, and SE-style gates. The lip-reading classifier is replaced. |
| `nac_tcn` | [Mehta & Yang, 2023](https://arxiv.org/abs/2312.07507) | TCN blocks combining causal dilated convolution with causal dilated neighborhood attention for lower-cost temporal attention. | Core faithful adaptation. `NACTCN` defaults to `causal=True` and implements dependency-free local/dilated attention instead of requiring `natten`. |
| `dwa_tcn` | [Iwana & Uchida, 2017](https://arxiv.org/abs/1712.06530) | Dynamic Weight Alignment uses DTW-style dynamic programming to align convolution weights to temporal input windows. | Source-faithful differentiable variant. `DynamicWeightAlignmentTCN` uses `SoftDTWAlignmentConv1d`, which performs dynamic programming over input-window and weight-time positions inside each convolution. A softmax temperature keeps the alignment differentiable for gradient training. |
| `pit_tcn` | [Risso et al., 2022](https://arxiv.org/abs/2203.14768) | Pruning In Time learns dilation/temporal structure using differentiable binary masks over the time axis, then derives efficient hardware-friendly TCNs. | Source-faithful embedding adaptation. `PITTCN` learns time-axis dilation candidates and position gates with an L1 cost and exposes `export_architecture()` with discrete learned dilation, active kernel taps, gate values, and candidate probabilities for every PIT convolution. |

## Strict Versus Embedding Mode

- `tcn` and `nac_tcn` default to strict causal mode because causality is central
  to their cited architectures.
- Other variants default to acausal/offline mode because the original action
  segmentation/localization implementations commonly use full temporal context
  and NEMBA's target use case is post-hoc embedding extraction.
- Any TCN can be forced into causal or acausal mode:

```python
from embedders_backbones import Embedder

emb = Embedder(
    backbone="ed_tcn",
    model_kwargs={"causal": True, "d_model": 96, "dilations": (1, 2, 4, 8, 16)},
)
```
