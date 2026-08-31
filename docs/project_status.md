# Project Roadmap: Pressure Estimation Extension

## Completed

- Base toolbox setup with `uv` package management
- Multi-GPU distributed training setup
- Neckflix loader rebuilt on the external zarr cache (lazy, metadata-only
  construction, participant/posture/perspective filters, per-window label
  normalisation)
- Batch-dict contract end to end: loader dicts survive models, losses and
  evaluation, keyed by canonical channel/signal name (see
  `docs/architecture.md`; the original design spec lives in git history)
- All seven unsupervised methods running on Neckflix, scored per trace
  (POS/ICA/PBV also repaired after NumPy 2 removed the APIs they used)
- PhysMamba running on Neckflix, predicting ABP + CVP + ECG together, verified
  at production resolution on GPU (untuned learning check: ABP waveform Pearson
  0.76 on a held-out subject after 8 epochs over 600 windows)
- `MultiSignalTrainer` + `MODEL_REGISTRY`: adding a model is a builder function
  and a registry line, not a new trainer
- Pure-PyTorch Mamba fallback, so PhysMamba is runnable and testable without a
  `mamba_ssm` wheel *and* on CPU where one is installed (`PortableMamba`; the
  fused kernels are CUDA-only and otherwise make the model CPU-unusable)
- `mamba-ssm` builds natively on Windows: `vendor/mamba-ssm` carries the three
  MSVC fixes upstream lacks, `triton-windows` replaces the Linux-only `triton`,
  and `uv sync` does the rest. 8.6x faster and 3.7x smaller than the fallback at
  128x128x128

## In Progress

- Neckflix zarr cache generation (`rgb128`; participants still being added)
- PhysHydra model development — still on the legacy tuple contract, not yet a
  `DictModel`

## To Do

- Move the remaining architectures onto the dict contract (PhysNet, PhysFormer,
  RhythmFormer, iBVPNet, FactorizePhys, EfficientPhys, TS-CAN, BigSmall,
  PhysHydra); each is a `forward_video` signature change plus a registry line
- Pressure-specific metrics: systolic/diastolic detection, clinical agreement
  bands, morphology — the current report is waveform correlation plus the
  inherited HR metrics
- Denormalised (mmHg) reporting using the `label_stats` already carried in every
  batch and saved output
- IR/depth channels: the contract and loader already support them; no cache has
  been generated with them yet
- Full LOSO sweeps on HPC (`.slurm_scripts/Neckflix_PhysMamba_LOSO.slurm`)

---

Last updated: 2026-08-31
