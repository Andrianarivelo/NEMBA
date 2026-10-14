# Is the embedder integration optimal? A critical review and a roadmap

Short answer: **the plumbing is good, the science is not yet optimal.** Every
backbone shares one interface, one leak-free training loop, one decoder and a
no-training baseline, which is the right architecture. But the *choices inside*
that architecture (loss, masking, model selection, state model, session handling,
which methods are offered at all) lag behind what the field has converged on.
This document lists the evidence, the literature, and a prioritised plan.

All references were checked against PubMed (journal articles, with DOI) or the
proceedings / preprint servers (machine-learning venues).

---

## 1. What our own data says

From `benchmarks/results/` (synthetic recordings with known phase, speed and
three behavioural states; 3 seeds; one common evaluator):

* **No trained backbone beats PCA on held-out linear decoding.** PCA with no
  training decodes phase, speed and state at least as well as every deep model.
* **The objective decides what survives, not the architecture.** v0.1's leaky
  TCN family (which could copy masked frames) decoded the fast-varying speed
  better than the leak-free v0.2 models: an input-copying model behaves like a
  denoising autoencoder and keeps fast fluctuations, while honest masked
  prediction from temporal context discards what neighbouring frames cannot
  predict. Time-contrastive training discards even more (slow-feature bias).
* **Unsupervised states are fragile.** The adjusted Rand index between K-means
  states and the true states swings from about 0.02 to 0.98 for the *same*
  backbone across seeds.

## 2. Structural weaknesses (and what the literature does instead)

| # | current NEMBA choice | why it is sub-optimal | evidence / precedent |
|---|---|---|---|
| 1 | Gaussian MSE on z-scored (sqrt) data for every modality | spike counts are Poisson-like; MSE on z-scores weights a 0.1 Hz neuron like a 20 Hz one and ignores count statistics | LFADS, NDT and the Neural Latents Benchmark all use a Poisson likelihood for spikes [LFADS, NDT, NLB] |
| 2 | masking whole time spans only | the model only learns temporal interpolation, never "predict neuron i from the others at the same time" | multi-task masking (neurons, time, regions) improves population models [MtM]; held-out-neuron prediction is the standard evaluation [NLB] |
| 3 | no label-free model selection; best backbone picked by decoding on the same data | 24 backbones x objectives x presets is a garden of forking paths; selection on the evaluation data inflates the winner | co-smoothing (held-out neurons, bits per spike) is the accepted unsupervised criterion [NLB]; AutoLFADS tunes models on it [AutoLFADS] |
| 4 | states = K-means on single frames (plus DEC / CRF) | no dwell-time model, so jitter becomes spurious transitions; K by silhouette fails on continuous manifolds | AR-HMM / switching linear dynamical systems [rSLDS]; keypoint-MoSeq shows frame-wise clustering mistakes jitter for transitions [keypoint-MoSeq]; VAME and B-SOiD for pose [VAME, B-SOiD] |
| 5 | one recording at a time; `transform` needs identical channels | neurons change across days and animals, so embeddings cannot be compared or reused | aligned latent dynamics are stable across years [Gallego]; CEBRA multi-session and its consistency score [CEBRA]; MARBLE gives consistent cross-animal representations [MARBLE] |
| 6 | home-made CEBRA-time approximation | untested defaults (temperature, negatives drawn inside windows) | reference implementation is pip-installable with a scikit-learn API [CEBRA] |
| 7 | 24 backbones, 12 of them action-segmentation TCNs | designed for supervised video segmentation, not latent population dynamics; no evidence they help here; maintenance and multiple-comparison cost | the neural-latent literature converges on a few families: linear (PCA, FA, GPFA), sequential VAEs (LFADS), masked transformers (NDT), contrastive (CEBRA) [Cunningham & Yu, NLB] |
| 8 | behaviour enters only after the embedding | behaviourally relevant dynamics can be a small subspace that unsupervised models miss | PSID (linear) and DPAD (nonlinear) prioritise behaviour-relevant dynamics [PSID, DPAD]; TNDM, pi-VAE [pi-VAE] |

Counter-arguments worth keeping in mind:
* **Bigger is not automatically better.** Pretrained "foundation" models
  (NDT2/3, POYO/POYO+, MtM) need GPUs and large multi-session corpora, and a 2025
  preprint reports that data heterogeneity limits the benefit of scaling
  pretraining [Jiang 2025]. For a single lab on a CPU, linear and small models
  remain strong baselines (our benchmark agrees).
* **2-D pictures distort.** Projections from tens of dimensions to 2 inevitably
  distort distances and cluster structure [Chari & Pachter]; every visual must be
  paired with a quantitative metric computed in the full space (NEMBA does this
  for decoding and silhouette; it should do it for every new visual too).
* **Supervised contrastive embeddings are circular for decoding.** CEBRA-behavior
  trained on the same labels it is then scored on will look excellent by
  construction; it must be cross-fitted (train the embedding on training folds only).

## 3. Methods worth integrating

| method | what it adds | CPU | how to integrate | priority |
|---|---|---|---|---|
| **Poisson likelihood + neuron masking** (NDT / MtM style) | count-aware objective, cross-neuron structure | yes | new `likelihood="poisson"` head and `mask="time|neuron|both"` in `training.py`; keep raw counts for the loss | **1** |
| **Co-smoothing (bits per spike)** [NLB] | label-free model selection and early stopping | yes | hold out 10-20 % of neurons; predict them from the others' latents with a Poisson GLM | **1** |
| **Gaussian HMM / AR-HMM on the embedding** [rSLDS, keypoint-MoSeq] | states with dwell times; K by held-out likelihood | yes | numpy EM (diagonal Gaussians, sticky prior) or `hmmlearn`; AR-HMM for pose | **1** |
| **CEBRA** [CEBRA] | reference contrastive embeddings, multi-session, consistency score | yes (small `offset10-model`) | optional backbone `cebra` wrapping `cebra.CEBRA(...).fit/transform`; `consistency_score` in Compare | **1** |
| **Factor analysis** [Cunningham & Yu] | the other classic linear baseline (per-neuron noise) | instant | `sklearn.decomposition.FactorAnalysis`, same pattern as `pca` | **1** |
| **PSID** [PSID] / DPAD [DPAD] | behaviour-prioritised latents; PSID is linear and fast | PSID yes | optional `PSID` package; cross-fitted inside decoding folds | 2 |
| **GPFA** [GPFA] | smooth single-trial latents, classic spike baseline | moderate | optional via `elephant` | 2 |
| **Cross-session alignment** [Gallego] | compare / reuse embeddings across days and animals | yes | CCA / Procrustes alignment of latent spaces + aligned decoding | 2 |
| **Decoders: gradient boosting, Bayesian place decoder** [Glaser] | stronger non-linear decoder; interpretable spike decoder | yes | `HistGradientBoostingRegressor`; Poisson naive-Bayes on tuning curves | 2 |
| **Manifold latent models** [mGPLVM, SPUD] | ring / torus latents (head direction, grid cells) read out directly | moderate | unsupervised circular coordinate from the ring (SPUD-style) as a readout | 2 |
| LFADS / AutoLFADS [LFADS, AutoLFADS] | strongest single-trial dynamics | slow on CPU | external (`lfads-torch`) import of its latents | 3 |
| MARBLE [MARBLE], DSA [DSA] | compare dynamics (flow fields), not just geometry | moderate | external, import latents / flow fields | 3 |
| NDT2/3, POYO/POYO+, MtM, POSSM | pretrained multi-session models | GPU | out of scope for training; possibly import pretrained encoders later | 3 |

## 4. Visualisations worth adding

| visual | what it shows | source / precedent | cost |
|---|---|---|---|
| **Rastermap panel** | neurons sorted along a 1-D axis of similar activity, aligned with states and behaviour | Rastermap [Rastermap] (`pip install rastermap`, CPU) | low |
| **Persistence barcode / diagram** | loops (H1) and voids: ring or torus topology, not just a "circularity" number | grid-cell torus [Gardner], head-direction ring [SPUD] (ripser already used) | low |
| **Flow field / phase portrait** | average velocity of the latent state on a grid: attractors, rotations, transitions | flow-field representations [MARBLE] | low |
| **Cross-model consistency matrix** | R² between embeddings after linear alignment, for backbones, seeds or sessions | CEBRA consistency [CEBRA] | low |
| **State-behaviour contingency** | discovered states vs annotated behaviours, with NMI / ARI | keypoint-MoSeq evaluations [keypoint-MoSeq] | low |
| **Co-smoothing per neuron** | which neurons the embedding explains | NLB [NLB] | low (after item 1) |
| **Recurrence plot** | similarity of the latent state over time: repeated sequences, slow drift | standard dynamical-systems tool | low |
| **T-PHATE projection** | time-aware manifold layout for trajectories | T-PHATE [T-PHATE], PHATE [PHATE] | medium (new dependency) |
| **3-D explorer view + linked traces** | rotate a 3-D embedding; brushing a time window highlights it in the traces | CEBRA uses 3-D outputs [CEBRA] | medium |

## 5. Recommended plan

1. **Objective (tier 1):** Poisson head for counts, neuron + time masking,
   co-smoothing evaluation, and co-smoothing-based early stopping and backbone
   selection. Re-run the benchmark: does it close the gap to PCA?
2. **States (tier 1):** sticky Gaussian HMM (and AR-HMM for pose) with K chosen by
   held-out likelihood; contingency and dwell-time figures.
3. **Methods (tier 1):** `fa` baseline, optional `cebra` backbone, consistency matrix.
4. **Visuals (tier 1):** Rastermap, barcodes, flow field, recurrence plot.
5. **Prune:** label the 12 action-segmentation TCN variants "experimental" and make
   the default comparison `pca, fa, tcn, tst, cebra`.
6. **Tier 2:** PSID, GPFA, cross-session alignment, gradient-boosting and Bayesian
   decoders, circular readout, T-PHATE, 3-D explorer.

## 6. Implementation status (after this review)

| item | status | where | measured outcome |
|---|---|---|---|
| Poisson likelihood, neuron / both masking | done | `training.py` (`likelihood`, `mask_mode`) | no significant gain over the default on synthetic data (`benchmarks/results/objectives_*`); leaning towards better state recovery |
| co-smoothing (bits/spike) | done | `cosmoothing.py`, `Embedder.cosmooth()` | PCA / FA score highest on the synthetic benchmark |
| sticky HMM / AR-HMM states, K by held-out likelihood | done | `states.py`, `state_model=` | see `benchmarks/results/states_validation.*` |
| factor analysis, GPFA (native), CEBRA wrapper | done | `_baselines.py`, `gpfa.py`, `integrations.py` | GPFA recovers simulated latents (R^2 > 0.99) and timescales (within 6 %) |
| PSID (cross-fitted) decoding rows | done | `integrations.py`, `decode(psid=True)` | R^2 0.94 on a simulated behaviour-specific subspace |
| gradient boosting, Bayesian count decoder | done | `decoding.py` | gbm R^2 0.82 where linear fails on a non-linear target |
| cross-session alignment, consistency | done | `align.py`, `Comparison.consistency()` | CCA robust to anisotropic stretch where Procrustes fails |
| persistence barcode, flow field, recurrence, Rastermap, contingency | done | `viz_dynamics.py`, `Embedder.run(dynamics=True)` | each figure carries a null reference |
| T-PHATE projection, 3-D explorer | done | `manifold.py`, explorer `view=3d` | native re-implementation; emphasises slow structure (separates states: 2-D silhouette 0.49 vs 0.02 for PCA) and blurs the faster ring |
| data advisor (profile -> recommendation) | done | `advisor.py`, `Embedder.auto`, `--advise` | 80/80 on 4 checks over 8 synthetic kinds x 10 seeds |
| LFADS, MARBLE, DSA, pretrained foundation models | not done | external | GPU-oriented; import their latents |

## References

Journal articles (verified in PubMed; DOI links):
* [AutoLFADS] Keshtkaran MR et al. A large-scale neural network training framework for generalized estimation of single-trial population dynamics. *Nat Methods* 19:1572-1577 (2022). [doi:10.1038/s41592-022-01675-0](https://doi.org/10.1038/s41592-022-01675-0)
* [B-SOiD] Hsu AI, Yttri EA. B-SOiD, an open-source unsupervised algorithm for identification and fast prediction of behaviors. *Nat Commun* 12:5188 (2021). [doi:10.1038/s41467-021-25420-x](https://doi.org/10.1038/s41467-021-25420-x)
* [CEBRA] Schneider S, Lee JH, Mathis MW. Learnable latent embeddings for joint behavioural and neural analysis. *Nature* 617:360-368 (2023). [doi:10.1038/s41586-023-06031-6](https://doi.org/10.1038/s41586-023-06031-6)
* [Chari & Pachter] Chari T, Pachter L. The specious art of single-cell genomics. *PLoS Comput Biol* 19:e1011288 (2023). [doi:10.1371/journal.pcbi.1011288](https://doi.org/10.1371/journal.pcbi.1011288)
* [Cunningham & Yu] Cunningham JP, Yu BM. Dimensionality reduction for large-scale neural recordings. *Nat Neurosci* 17:1500-1509 (2014). [doi:10.1038/nn.3776](https://doi.org/10.1038/nn.3776)
* [DPAD] Sani OG, Pesaran B, Shanechi MM. Dissociative and prioritized modeling of behaviorally relevant neural dynamics using recurrent neural networks. *Nat Neurosci* 27:2033-2045 (2024). [doi:10.1038/s41593-024-01731-2](https://doi.org/10.1038/s41593-024-01731-2)
* [Gallego] Gallego JA et al. Long-term stability of cortical population dynamics underlying consistent behavior. *Nat Neurosci* 23:260-270 (2020). [doi:10.1038/s41593-019-0555-4](https://doi.org/10.1038/s41593-019-0555-4)
* [Gardner] Gardner RJ et al. Toroidal topology of population activity in grid cells. *Nature* 602:123-128 (2022). [doi:10.1038/s41586-021-04268-7](https://doi.org/10.1038/s41586-021-04268-7)
* [Glaser] Glaser JI et al. Machine learning for neural decoding. *eNeuro* 7(4) (2020). [doi:10.1523/ENEURO.0506-19.2020](https://doi.org/10.1523/ENEURO.0506-19.2020)
* [GPFA] Yu BM et al. Gaussian-process factor analysis for low-dimensional single-trial analysis of neural population activity. *J Neurophysiol* 102:614-635 (2009). [doi:10.1152/jn.90941.2008](https://doi.org/10.1152/jn.90941.2008) (already scored models by predicting each neuron from all the others, the idea behind co-smoothing)
* [keypoint-MoSeq] Weinreb C et al. Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics. *Nat Methods* 21:1329-1339 (2024). [doi:10.1038/s41592-024-02318-2](https://doi.org/10.1038/s41592-024-02318-2)
* [LFADS] Pandarinath C et al. Inferring single-trial neural population dynamics using sequential auto-encoders. *Nat Methods* 15:805-815 (2018). [doi:10.1038/s41592-018-0109-9](https://doi.org/10.1038/s41592-018-0109-9)
* [MARBLE] Gosztolai A et al. MARBLE: interpretable representations of neural population dynamics using geometric deep learning. *Nat Methods* 22:612-620 (2025). [doi:10.1038/s41592-024-02582-2](https://doi.org/10.1038/s41592-024-02582-2)
* [PHATE] Moon KR et al. Visualizing structure and transitions in high-dimensional biological data. *Nat Biotechnol* 37:1482-1492 (2019). [doi:10.1038/s41587-019-0336-3](https://doi.org/10.1038/s41587-019-0336-3)
* [PSID] Sani OG et al. Modeling behaviorally relevant neural dynamics enabled by preferential subspace identification. *Nat Neurosci* 24:140-149 (2021). [doi:10.1038/s41593-020-00733-0](https://doi.org/10.1038/s41593-020-00733-0)
* [Rastermap] Stringer C et al. Rastermap: a discovery method for neural population recordings. *Nat Neurosci* 28:201-212 (2025). [doi:10.1038/s41593-024-01783-4](https://doi.org/10.1038/s41593-024-01783-4)
* [SPUD] Chaudhuri R et al. The intrinsic attractor manifold and population dynamics of a canonical cognitive circuit across waking and sleep. *Nat Neurosci* 22:1512-1520 (2019). [doi:10.1038/s41593-019-0460-x](https://doi.org/10.1038/s41593-019-0460-x)
* [T-PHATE] Busch EL et al. Multi-view manifold learning of human brain-state trajectories. *Nat Comput Sci* 3:240-253 (2023). [doi:10.1038/s43588-023-00419-0](https://doi.org/10.1038/s43588-023-00419-0)
* [VAME] Luxem K et al. Identifying behavioral structure from deep variational embeddings of animal motion. *Commun Biol* 5:1267 (2022). [doi:10.1038/s42003-022-04080-7](https://doi.org/10.1038/s42003-022-04080-7)

Machine-learning venues and preprints (verified on proceedings / preprint servers):
* [DSA] Ostrow M, Eisen A, Kozachkov L, Fiete I. Beyond geometry: comparing the temporal structure of computation in neural circuits with dynamical similarity analysis. *NeurIPS* (2023). [arXiv:2306.10168](https://arxiv.org/abs/2306.10168)
* [Jiang 2025] Jiang et al. Data heterogeneity limits the scaling effect of pretraining neural data transformers. Preprint (2025), listed in [awesome-neurofm](https://github.com/mazabou/awesome-neurofm).
* [mGPLVM] Jensen KT, Kao TC, Tripodi M, Hennequin G. Manifold GPLVMs for discovering non-Euclidean latent structure in neural data. *NeurIPS* (2020). [arXiv:2006.07429](https://arxiv.org/abs/2006.07429)
* [MtM] Zhang Y et al. Towards a "universal translator" for neural dynamics at single-cell, single-spike resolution. *NeurIPS* (2024). [arXiv:2407.14668](https://arxiv.org/abs/2407.14668)
* [NDT] Ye J, Pandarinath C. Representation learning for neural population activity with Neural Data Transformers. *Neurons, Behavior, Data analysis, and Theory* (2021). [arXiv:2108.01210](https://arxiv.org/abs/2108.01210)
* [NDT2] Ye J, Collinger J, Wehbe L, Gaunt R. Neural Data Transformer 2: multi-context pretraining for neural spiking activity. *NeurIPS* (2023). [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe51de4e7baf52e743b679e3bdba7905-Abstract.html)
* [NDT3] Ye J et al. A generalist intracortical motor decoder. Preprint (2025).
* [NLB] Pei F et al. Neural Latents Benchmark '21: evaluating latent variable models of neural population activity. *NeurIPS Datasets and Benchmarks* (2021). [arXiv:2109.04463](https://arxiv.org/abs/2109.04463)
* [pi-VAE] Zhou D, Wei XX. Learning identifiable and interpretable latent models of high-dimensional neural activity using pi-VAE. *NeurIPS* (2020). [arXiv:2011.04798](https://arxiv.org/abs/2011.04798)
* [POSSM] Generalizable, real-time neural decoding with hybrid state-space models. *NeurIPS* (2025).
* [POYO] Azabou M et al. A unified, scalable framework for neural population decoding. *NeurIPS* (2023). [arXiv:2310.16046](https://arxiv.org/abs/2310.16046)
* [POYO+] Multi-session, multi-task neural decoding from distinct cell-types and brain regions. *ICLR* (2025).
* [rSLDS] Linderman SW et al. Bayesian learning and inference in recurrent switching linear dynamical systems. *AISTATS*, PMLR 54:914-922 (2017). [proceedings](https://proceedings.mlr.press/v54/linderman17a.html)
