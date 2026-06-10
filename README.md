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

All share the same masked-reconstruction SSL and the same `encode -> [B,E,T]`
interface, so they are interchangeable.

## Inputs (any modality -> channels x time)

| input type | pass | adapter |
|---|---|---|
| spike trains | list of spike-time arrays (s) | `fit_spikes` |
| pose (DeepLabCut) | `[features, time]` matrix | `fit_features` |
| segmentation masks | `[features, time]` matrix | `fit_features` |
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
python main.py                      # no args -> GUI
```

**Library:**
```python
from embedders_backbones import Embedder
emb = Embedder(backbone="dtc", n_states=8, epochs=120).fit_spikes(spike_times)
emb.plot_embeddings("emb.png")          # PCA/t-SNE/UMAP
emb.animate("traj.gif", method="umap")  # trajectory animation
emb.run("out_dir")                      # everything + metrics.csv
print(emb.metrics())                    # circularity, diameter, PR, speed, ...
```

## Install (optional)
```bash
pip install -e "D:\embedders_backbones[full]"   # full = + umap-learn, ripser
```
Runs on GPU automatically when CUDA is available (`device="auto"`). No install is
needed to run `main.py` - it adds the package to the path itself. Tkinter (GUI)
ships with Python.

## Outputs of `run(outdir)`
`<bb>_embeddings.png` (projections), `<bb>_states.png` (state raster + optional
behaviour), `<bb>_trajectory.gif`, `<bb>_metrics.csv/.json`, `<bb>_embedding.npz`
(`Z`, `states`, `times`).
