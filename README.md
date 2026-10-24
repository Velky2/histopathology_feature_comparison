# Learned features vs. classical texture descriptors on colorectal histopathology — code and protocol

Companion repository for *"Does the Learned-Feature Advantage over Classical Texture Descriptors Survive Controls for Invariance, Dimensionality and Label Regime? A Controlled Study on Colorectal Histopathology"* (WVC 2026).

## Data
- **NCT-CRC-HE-100K** (Macenko-normalised distribution) and **CRC-VAL-HE-7K**, Kather, Halama & Marx, Zenodo, doi:10.5281/zenodo.1214456. Not redistributed here.
- **Canonical order:** class folders in the order `ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM`, and files sorted by name inside each folder. Every index file in `splits/` refers to this order.
- Class counts: 100K = `[10407, 10566, 11512, 11557, 8896, 13536, 8763, 10446, 14317]`; 7K = `[1338, 847, 339, 634, 1035, 592, 741, 421, 1233]`.

## Splits and seeds (`splits/`)
| file | content |
|---|---|
| `idx_train_pool_80k.npy` / `idx_eval_interno_20k.npy` | Internal split, **patch-level**, stratified by class: `train_test_split(test_size=0.20, stratify=y, random_state=20260816)`. The public distribution has no slide or patient IDs, so patches from the same slide fall on both sides. |
| `subamostra_f{1,5,10,25}_s{0..4}.npy` | Labelled subsamples used to fit the probes: `train_test_split(train_size=f, stratify=y_pool, random_state=seed)`. 100% = the whole pool. |
| `B1_1_subamostra_interna_500porclasse.npy` | Subsample used by the post-review colour intervention (seed 20261022). |

Other seeds: SimCLR pre-training 0, 1, 2 (full recipe) and 0, 1 (ablation). Paired bootstrap seed 20260818. kNN-monitor carve seed 0.

## Protocol and pre-commitment (`protocol/`)
- `precommit_7k.json`: the 7K evaluation spec (arms, conditions, fractions, seeds, probe, statistics), recorded 2026-08-17 22:13 before the final evaluation run. `spec_hash = df4b2937407f9b87` is the SHA-256 prefix of the JSON-serialised **arm list only**. Before the final run, CRC-VAL-HE-7K had been scored by the Phikon sanity check and by two runs of the same pipeline that crashed at the reporting stage. The 7K was never used for model, arm, hyper-parameter or checkpoint selection.
- Dated internal protocol and weekly logs, including every deviation: Wilcoxon replaced by the paired bootstrap, SSL pre-trained on 80K, H3-a demoted, CLASSIC-MAX comparisons post hoc.
- `adendo_posaceite.md` + `.sha256` + `adendo_posaceite_errata.md`: post-acceptance exploratory analyses and their reading rules, plus the errata (re-analyses without new predictions were computed before the addendum and are descriptive).
- The protocol is pre-specified (dated document plus hash commitment). It was **not** filed in a public registry.

## Code (`code/`)
- Classical descriptors: `descritores.py`, a verbatim copy of the extraction code. Bit-exact against the cached features.
- Probe, split, bootstrap, Holm: `common.py`.
- SimCLR pre-training (revision-1 recipe): `simclr_2x2_treino.py`.
- Feature extraction: `extrai_gpu_camera_ready.py`.
- Original notebooks: `wvc_semana3_4_1.ipynb` (SimCLR), `wvc_semana5_6.ipynb` (descriptors, probes, 7K evaluation), `wvc_semana7_reanalise.ipynb` (statistics without refitting).
- Post-review analyses: `b0_reanalise.py`, `b1_*.py`, `b2_1_probes_2x2.py`.

## Environment and hardware
`environment.yml`: Python 3.13.2, PyTorch 2.10.0 (cu128), torchvision 0.25.0, transformers 4.49.0, scikit-learn 1.7.2, scikit-image 0.26.0, NumPy 2.3.3, SciPy 1.16.2.
- SimCLR pre-training: 1× NVIDIA Tesla T4 (16 GB), 8 data-loader workers, AMP and gradient checkpointing, batch 256.
- 7K feature extraction: 1× NVIDIA GeForce RTX 4090 (24 GB).
- Descriptors and probes: CPU nodes (8–16 cores), SLURM cluster at CNPEM.

## Reproducing the tables
1. Extract features (`wvc_semana5_6.ipynb` §7–§11, GPU only for Phikon and SimCLR).
2. Run the single 7K evaluation (§12). Predictions are stored as `preds_7k/ARM__COND__fPCT__sSEED.npy`.
3. Run the statistics with `wvc_semana7_reanalise.ipynb` or `code/b0_reanalise.py`.
