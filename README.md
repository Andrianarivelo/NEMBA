# embedders_backbones

Standalone, self-supervised **sequence encoders** for any population time series,
with a unified API and embedding visualisation. Pick a backbone, fit it to your
data, and get per-frame embeddings, discrete states, PCA/t-SNE/UMAP projections,
a trajectory GIF, and manifold metrics.

## Backbones

| key | architecture |
|---|---|
| `dtc` | Transformer auto-encoder + Deep Embedded Clustering |
| `tst` | Time-Series Transformer |
| `bilstm` / `attn_bilstm` | Bi-LSTM (+ optional attention) |
| `stt` | Spatiotemporal (axial) Transformer |
| `gatr` | Graph-Attention Transformer |
| `vit` | Vision Transformer (space x time patches) |
| `enc_crf` | Encoder + linear-chain CRF |
| `rcnn` | Recurrent CNN |
| `patchtst` | PatchTST |
| `embtcn` | EmbTCN-Attention |
| `tcn` | Generic residual/dilated TCN |
| `ed_tcn` | Encoder-Decoder TCN |
| `dilated_tcn` | Lea/WaveNet-style Dilated TCN |
| `ms_tcn` / `ms_tcnpp` | Multi-Stage TCN / MS-TCN++ dual-dilated TCN |
| `c2f_tcn` | Coarse-to-Fine temporal U-Net TCN |
| `stcn` | Stochastic latent TCN |
| `c_tcn` | Concept-wise TCN |
| `dc_tcn` | Densely Connected multiscale TCN |
| `nac_tcn` | Neighborhood Attention + Convolution TCN |
| `dwa_tcn` | Dynamic Weight Alignment TCN |
| `pit_tcn` | Pruning-in-Time TCN |

All share the same masked-reconstruction SSL and the same `encode -> [B,E,T]`
interface, so they are interchangeable.
The TCN variants were audited against the Emergent Mind TCN review and the
papers below; see `TCN_VARIANTS.md` for the detailed compliance notes. Because
NEMBA trains self-supervised embeddings, paper-specific classification,
segmentation, generative-likelihood, and architecture-search heads are replaced
by the common masked-reconstruction head.

### TCN variant citations

| key | cited architecture | NEMBA compliance |
|---|---|---|
| `tcn` | [Bai, Kolter & Koltun, 2018](https://arxiv.org/abs/1803.01271) | Faithful residual/dilated TCN core; defaults to causal mode unless overridden. |
| `ed_tcn` | [Lea et al., 2016](https://arxiv.org/abs/1611.05267) | Faithful encoder-pool / upsample-decoder backbone; supervised softmax head replaced. |
| `dilated_tcn` | [Lea et al., 2016](https://arxiv.org/abs/1611.05267) | Faithful dilated residual + skip-aggregation backbone; supervised softmax head replaced. |
| `ms_tcn` | [Abu Farha & Gall, 2019](https://arxiv.org/abs/1903.01945) | Latent version of staged prediction refinement with dilated residual layers. |
| `ms_tcnpp` | [Li et al., 2020](https://arxiv.org/abs/2006.09220) | Latent staged refinement with dual-dilated small/large receptive-field layers. |
| `c2f_tcn` | [Singhania, Rahaman & Yao, 2021](https://arxiv.org/abs/2105.10859) | Coarse-to-fine temporal U-Net backbone with pyramid bottleneck; task losses replaced. |
| `stcn` | [Aksan & Hilliges, 2019](https://arxiv.org/abs/1902.06568) | Hierarchical prior/posterior Gaussian latent TCN with KL auxiliary ELBO term; task-specific likelihood head replaced. |
| `c_tcn` | [Li et al., 2019](https://arxiv.org/abs/1908.09442) | Faithful concept-wise temporal filtering with shared filters before channel mixing. |
| `dc_tcn` | [Ma et al., 2020](https://arxiv.org/abs/2009.14233) | Faithful dense/multiscale temporal block idea with squeeze-excitation gates. |
| `nac_tcn` | [Mehta & Yang, 2023](https://arxiv.org/abs/2312.07507) | Causal convolution + dilated local attention adaptation without `natten` dependency. |
| `dwa_tcn` | [Iwana & Uchida, 2017](https://arxiv.org/abs/1712.06530) | Differentiable soft-DTW alignment convolution inside TCN residual blocks. |
| `pit_tcn` | [Risso et al., 2022](https://arxiv.org/abs/2203.14768) | Learnable time-axis masks with L1 cost and `export_architecture()` for discrete dilation/pruning metadata. |

Review source: [Emergent Mind, Temporal Convolutional Networks](https://www.emergentmind.com/topics/temporal-convolutional-networks).
Local PDF copies of the cited arXiv papers are archived in
[`docs/papers`](docs/papers/README.md).

## Inputs (any modality -> channels x time)

| input type | pass | adapter |
|---|---|---|
| spike trains | list of spike-time arrays (s) | `fit_spikes` |
| pose (DeepLabCut) | `[features, time]` matrix | `fit_features` |
| segmentation mask features | `[features, time]` matrix | `fit_features` |
| COCO segmentation masks | raw COCO `images`/`annotations` JSON | `fit_coco_masks` |
| fiber photometry | continuous signal(s) (+ optional dF/F) | `fit_fiber` |
| any time series | `[channels, time]` or `[time, channels]` | `fit_features` / `fit_matrix` |

## Run it

**GUI (one click):** open `main.py` in VSCode and press Run (F5), or
`python main.py`. Pick a file, input type and backbone, press Run; the embedding
is shown in the window and saved. Leave the file empty for a synthetic demo.

**CLI:**
```bash
python main.py --list
python main.py --backbone dtc --input spikes --file spikes.npy --out out
python main.py --backbone tst --input fiber_photometry --file photometry.npy --fs 30 --dff
python main.py --backbone ed_tcn --input coco_mask --file masks.json --mask-size 64
python main.py --compare tcn,ed_tcn,dilated_tcn,c2f_tcn,pit_tcn --input coco_mask --file masks.json --mask-channel-key track_id
python main.py                      # no args -> GUI
```

**Library (single backbone):**
```python
from embedders_backbones import Embedder
emb = Embedder(backbone="dtc", epochs=120).fit_spikes(spike_times)  # n_states auto by silhouette
emb.plot_embeddings("emb.png")          # PCA/t-SNE/UMAP
emb.plot_state_sequence("states.png")   # state raster (+ optional behaviour)
emb.animate("traj.gif", method="umap")  # trajectory animation
emb.run("out_dir")                      # everything + metrics.csv
print(emb.metrics())                    # silhouette, circularity, diameter, PR, ...
```

**Direct COCO masks:**
```python
from embedders_backbones import Embedder

emb = Embedder(backbone="c2f_tcn", epochs=120).fit_coco_masks(
    "masks.json",
    resize=(64, 64),          # raw binary mask raster, flattened as pixels x time
    channel_key="track_id",   # optional: keep tracked animals/objects separate
    fps=30,
)
Z = emb.embedding()
```

`fit_coco_masks` rasterizes polygons and uncompressed RLE directly. Compressed
COCO RLE is supported when `pycocotools` is installed.

TCN hyperparameters can be passed through `model_kwargs`, for example:
```python
emb = Embedder(
    backbone="pit_tcn",
    model_kwargs={"d_model": 64, "dilations": (1, 2, 4, 8), "causal": False},
)
```

**Compare several backbones on the same data:**
```python
from embedders_backbones import compare_embedders
cmp = compare_embedders(spike_times, input="spikes",
                        backbones=["dtc", "tst", "bilstm", "stt", "gatr"], epochs=80)
cmp.run("compare_out")        # metrics table + figures
print(cmp.metrics)            # one row per backbone (silhouette, circularity, ...)
cmp.best("silhouette")        # winning backbone
```
or from the CLI: `python main.py --compare dtc,tst,bilstm --input spikes --file spikes.npy`,
or in the GUI with the **Compare** button.

`compare_out/` holds `comparison_metrics.csv`, `compare_embeddings.png` (each
backbone's projection side by side), **`compare_state_sequences.png`** (the state
sequences time-aligned, one row per backbone), and `compare_silhouette.png` /
`compare_circularity.png` bar charts.

### Number of states
If you don't pass `n_states`, K is **chosen automatically by silhouette** over
`[k_min, k_max]` (default 2-10). Pass `n_states=8` to fix it. The chosen K and its
silhouette are recorded in the metrics (`k_selected_by`, `silhouette_at_k`).

## Install (optional)
```bash
pip install -e "D:\embedders_backbones[full]"   # full = + umap-learn, ripser
pip install -e ".[coco]"                         # optional compressed COCO RLE support
```
Runs on GPU automatically when CUDA is available (`device="auto"`). No install is
needed to run `main.py` - it adds the package to the path itself. Tkinter (GUI)
ships with Python.

## Outputs of `run(outdir)`
`<bb>_embeddings.png` (projections), `<bb>_states.png` (state raster + optional
behaviour), `<bb>_trajectory.gif`, `<bb>_metrics.csv/.json`, `<bb>_embedding.npz`
(`Z`, `states`, `times`).
