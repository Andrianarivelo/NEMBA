# NEMBA data advisor

## What the data looks like

* **spike times (sorted units)**: 60 spike trains (list of spike-time arrays)
* 60 channels x 12,000 frames (600 s at 20 Hz); time axis 1 (given confidence)
* values: counts; median count/bin 0.196, median rate 3.92 Hz, Fano factor 1.27
* dominant timescale: 33 frames (1.65 s); spectral peak 0.137 Hz (x197 median power)
* slow drift: 0 % of variance
* dimensionality: participation ratio 4.9; 4 / 6 PCs for 80 / 90 % variance
* loop (ring) persistence: 0.30 -> ring-like manifold
* cluster structure: silhouette 0.38 at K=3 (weak)
* behaviour `phase`: circular
* behaviour `speed`: regression
* behaviour `state`: classification, 3 levels

## Recommended pipeline

```python
from embedders_backbones import Embedder, compare_embedders
emb = Embedder('tcn', speed='fast', objective='masked', likelihood='gaussian', mask_mode='time', n_states=None, input_pca='auto', embedding_dim=12, smooth_sigma=8.0, state_model='auto', bin_size=0.05, win_len=256)
emb.fit_spikes(data)
emb.run('out', behavior=behavior, methods=('pca', 'umap', 'tsne'))
# honest check against no-training baselines:
cmp = compare_embedders(data, input='spikes', backbones=['pca', 'fa', 'tcn'], speed='fast')
```

| setting | value |
|---|---|
| `backbone` | `'tcn'` |
| `speed` | `'fast'` |
| `objective` | `'masked'` |
| `likelihood` | `'gaussian'` |
| `mask_mode` | `'time'` |
| `n_states` | `None` |
| `input_pca` | `'auto'` |
| `embedding_dim` | `12` |
| `smooth_sigma` | `8.0` |
| `state_model` | `'auto'` |
| `bin_size` | `0.05` |
| `win_len` | `256` |
| compare with | pca, fa, tcn |
| projections | pca, umap, tsne |
| colour by | time, state, phase, speed, state, circular coordinate (ring angle) |
| figures | dashboard, explorer, decoding, persistence barcode, flow field, trajectory GIF, recurrence plot, Rastermap raster |

## Why

| topic | decision | evidence |
|---|---|---|
| binning | bin_size = 0.05 s | smallest bin with >= 0.1 spikes per bin for the median neuron (3.9 Hz) and >= 1000 frames |
| backbone | tcn (masked, fast) | fast on CPU; no loss vs v0.1 in the benchmark; compare with pca / fa |
| objective | gaussian likelihood + time masks (default) | no objective variant beat the default in the NEMBA objective benchmark (6 paired fits, all Holm-corrected p > 0.4); Poisson + neuron masks remain an option for state recovery |
| embedding_dim | 12 | twice the 6 components needed for 90 % of the (smoothed) variance |
| smoothing | smooth_sigma = 8 frames | half the autocorrelation time (33 frames) (capped at 8 frames): smooths noise without blurring the dominant timescale |
| window | win_len = 256 | windows must span several autocorrelation times (33 frames) |
| model choice | rank backbones by co-smoothing (bits/spike) | label-free: no selection bias from choosing the best decoder on the same labels |
| representation | persistence barcode, flow field, cyclic colours | loop persistence 0.30 >= 0.18: ring-like manifold (head-direction ring, Chaudhuri 2019; grid torus, Gardner 2022) |
| representation | Rastermap-sorted raster | 60 neurons: sorting by activity similarity reveals sequences and assemblies (Stringer et al. 2025) |
| decoding | linear, blocked CV, 200 circular shifts | linear first (interpretable); n_perm >= 20 x targets so Holm-corrected p can reach 0.05 |

_These are rule-based suggestions from measured properties, not guarantees. Always check the recommended no-training baseline._
