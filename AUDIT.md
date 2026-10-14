# NEMBA audit (v0.1.0 -> v0.2.0)

Audit of commit `a017c65` (v0.1.0) and what v0.2.0 changes. Every claim below
was checked by running code on this repository; the scripts are named so the
evidence can be reproduced.

## 1. Correctness defects

| # | severity | finding | evidence | fix in v0.2 |
|---|---|---|---|---|
| 1 | critical | **Masked self-supervision leaked the answer.** All 12 TCN variants ignored `mask` in `encode`, so masked frames were visible to the encoder. EmbTCN applied the mask token *after* a dilated TCN stack (receptive field about +/-62 frames), so masked content reached every neighbouring token. RCNN and Encoder+CRF leaked partially (convolution before the mask). | Training on **white noise** (unpredictable, honest masked loss is about 1.0): v0.1 EmbTCN reached 0.049, TCN 0.155, C2F-TCN 0.505 (copying the hidden frames). TST (correct masking) stayed at 0.96. | The shared training loop zeroes masked frames in the *input* (`training.apply_input_mask`) for every backbone. Regression test `test_masked_training_on_white_noise_cannot_beat_chance` and a per-backbone invariance test. |
| 2 | high | **Seams in inference.** `embed_sequence` used non-overlapping windows with a zero-padded tail, so every `win_len` frames the context jumped (artefacts aligned with window edges), batch size 1. | Code reading; with a per-frame model v0.2 inference now equals the direct projection to 1e-4 (`test_overlap_add_embedding_has_no_seams_for_framewise_model`). | Batched overlap-add with a positive Hann taper (ramp for causal models). 10x faster (0.20 s -> 0.02 s for TCN on 6000 frames). |
| 3 | high | **GUI not thread-safe.** Tk widgets (`Text`, canvas) were updated from worker threads, which Tk does not support (random freezes / `RuntimeError`). | Code reading (`_msg`, `_draw` called in `threading.Thread` targets). | Queue + `root.after` polling; figures are built off-thread on bare `Figure` objects, embedded on the main thread. |
| 4 | medium | STCN read out the **prior** mean at evaluation, not the posterior mean. For representation extraction the posterior (what the data says) is the right readout, as in a VAE encoder. | Code reading. | `stochastic_eval_posterior=True` (default); the prior readout is still available. |
| 5 | medium | Randomised-PCA `fit_transform` returns the approximate U*S, which differs from `transform` on the same data (new in v0.2 code, caught by a test before release). | `test_input_reducer_auto_and_nan_interpolation`. | Fit, then project. |
| 6 | low | `spikes_to_matrix` silently drops silent neurons and re-orders rows by firing rate, with no way to map channels back to neurons. | Code reading. | `return_index=True`; `Embedder.channel_index`. |
| 7 | low | NaN in pose data were replaced by the row mean (artificial jumps). | Code reading. | Linear interpolation of NaN gaps (`data.interpolate_nans`). |

## 2. Speed on CPU (no GPU)

Measured on a 16-thread laptop CPU, torch 2.14 CPU build, 60 channels x 6000 frames.

| item | v0.1 | v0.2 |
|---|---|---|
| `import embedders_backbones` | 16.3 s (eager `import umap` pulls numba + TensorFlow) | 2.5 s (lazy optional imports; the rest is torch) |
| inference, TCN, whole recording | 0.20 s | 0.02 s (batched overlap-add) |
| STT / GATR training | 25-27 s per epoch (3 steps); about 50 min for the default 120 epochs | fused attention (`need_weights=False`), single LayerNorm per block, automatic batch cap at 2^16 tokens per step, wall-clock budget per preset |
| training budget | epochs: cost grows linearly with recording length | fixed step budget + early stopping on a held-out block; cost independent of recording length |
| dF/F baseline | Python loop, O(T x window): hours for 30 min at 1 kHz | C median filter on block medians: seconds |
| CRF state head | Python loop over all T frames, 80 times | batched 256-frame chunks + NumPy Viterbi |
| span masks | one random draw per span in a Python loop | one vectorised draw |
| wide inputs (64x64 mask pixels = 4096 channels) | every backbone slow; channel-token models infeasible | `input_pca="auto"`: 64 randomised-PCA components when > 128 channels |

Honest caveat: for a *short* recording with a cheap backbone, v0.1's default
(80 epochs of 3 small steps) was already cheap; v0.2's `fast` preset trains on a
comparable number of frames and is not faster there. The gains are for long
recordings, attention-heavy backbones, wide inputs, inference, import and
preprocessing. See `benchmarks/results/` for the measured comparison.

## 3. Scientific design issues

1. **No decoding, no held-out evaluation, no null.** v0.1 reported silhouette,
   circularity and similar geometry metrics only. Geometry metrics do not tell
   you whether the embedding carries behaviourally relevant information, and
   silhouette is biased towards compact convex clusters and small K.
   *v0.2:* `decoding.py` with blocked cross-validation and a purge gap,
   circular-shift permutation null, Holm correction, and a raw-input (PCA)
   baseline decoded by the same pipeline (`delta_vs_raw`).
2. **Random K-fold would have been wrong.** Neighbouring frames are
   autocorrelated and embeddings are smoothed (sigma = 4 frames), so random
   folds place near-duplicates of each test frame in the training set. Blocked
   folds with a gap are used instead; a frame-shuffle null would also be
   anti-conservative, the circular-shift null keeps both autocorrelations
   (`test_circular_shift_null_is_calibrated_for_independent_series`).
3. **Auto-K by silhouette is unreliable on continuous manifolds.** On a ring
   the silhouette keeps rising with K, so the selected K is the upper bound.
   *v0.2:* the whole score curve is saved, the silhouette is read with the
   Kaufman & Rousseeuw (1990) thresholds (`cluster_structure`), and warnings are
   logged when there is no substantial structure or K hits `k_max`.
4. **No honest baseline.** A 23-backbone zoo without a no-training reference
   invites over-interpretation. *v0.2:* `pca` backbone (closed form, instant).
   On the synthetic ground truth it matches or beats every trained backbone on
   linear decoding, which is the expected result when the latent structure is
   (close to) linearly embedded in the population. Deep encoders earn their
   cost on data with nonlinear structure, so always compare against it.
5. **Reconstruction objectives are not aimed at behaviour.** Masked
   reconstruction rewards whatever is predictable from context. v0.2 adds a
   time-contrastive objective (CEBRA-time style) and a hybrid. Measured trade-off
   on the synthetic data: contrastive training lowers linear decoding of the
   fast-varying speed variable (R^2 0.58 vs 0.78 for masked, TCN, one seed) because
   it favours slow features. Masked therefore remains the default.
6. **Smoothing is a strong, hidden prior.** Gaussian smoothing (sigma = 4 frames)
   raises state stability and slow-variable decoding but blurs fast transients;
   it is now recorded in `params.json` and used consistently for the baseline.

## 4. Alternatives worth knowing (not implemented here)

* **CEBRA** (Schneider, Lee & Mathis, 2023, *Nature* 617:360-368): contrastive
  embeddings, with behaviour-supervised variants (PMID 37138088). `objective="contrastive"`
  approximates CEBRA-time only; the supervised variants are not included because
  they would leak labels into a later decoding test unless strictly cross-fitted.
* **LFADS / latent dynamical models** (Pandarinath et al., 2018, *Nature Methods*
  15:805-815, PMID 30224673) model the dynamics explicitly and give smoother single-trial
  trajectories.
* **GPFA** (Yu et al., 2009, *J Neurophysiol* 102:614-635, PMID 19357332) is the classic
  linear-Gaussian baseline for spike-count latents.
* **HMM / switching linear dynamical systems** give discrete states with explicit
  temporal persistence, which is often more appropriate than clustering a
  continuous embedding frame by frame.

## 5. What was not changed

The TCN/transformer architectures themselves were left as audited in
`TCN_VARIANTS.md` (apart from the attention-call and STCN readout fixes). The
v0.1 `Embedder` calls (`Embedder("dtc", n_states=8, epochs=120)`,
`fit_*`, `run`, `metrics`, `compare_embedders`) still work; `epochs` is converted
to an equivalent step budget. `Embedder.projection` now returns `(P, frame_index)`.
