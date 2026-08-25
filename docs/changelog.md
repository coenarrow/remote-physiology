# Changelog

This file tracks development milestones. As of 2026-08-25 all branches (HPC, MacOS, phys-hydra) are unified into `main`, which has intentionally diverged from upstream rPPG-Toolbox (upstream fixes are cherry-picked only).

**Format:** Each entry records the date, the branch or development context, and a summary of changes. Entries are listed in reverse chronological order (most recent first).

---

## 2026-08-25 — Branch: `main`

Unified branches, wired local Neckflix subset, replaced the vendored mamba fork.

### Changes
- Merged HPC and MacOS branches; absorbed phys-hydra history; deleted all branches except `main`
- Reduced drift vs upstream; upstream/main fully merged at b7500b8 (intentional divergence begins here)
- Added local physhydra configs targeting the Neckflix subset cache (P030-P039)
- Added `mamba-ssm-macos` (darwin) and newest official `mamba-ssm` + `causal-conv1d` (linux) as platform-marked deps
- **Removed vendored `tools/mamba` fork.** PhysMamba/PhysHydra now use vanilla mamba_ssm via `neural_methods/model/mamba_compat.py`: `make_mamba(..., bimamba=True)` returns a composition-based `BiMamba` (two vanilla blocks, one time-reversed, summed). NOT parameter-compatible with fork-trained checkpoints; fresh training required. Separate in/out projections per direction (small param increase vs fork's shared projections)
- MPS support: trainers select cuda -> mps -> cpu; parallel (Hillis-Steele) selective scan with analytical-backward autograd for non-CUDA mamba_ssm (~340x faster than CPU loop at debug res; full 128x128 res trains at ~19 s/step, 18 GB on M1 Pro)
- Added tests/test_mamba_compat.py (BiMamba semantics + parallel-scan-vs-reference)

## 2026-02-12 — Branch: `HPC`

Completed DDP migration for remaining models and added multi-GPU SLURM infrastructure.

### Changes
- Migrated FactorizePhys trainer to DDP with UBFC-rPPG validation config
- Migrated BigSmallTrainer to DDP, fixing pre-existing bugs in the process
- Migrated iBVPNet trainer to DDP with UBFC-rPPG validation config
- Added 2-GPU SLURM scripts for all models (EfficientPhys, FactorizePhys, PhysFormer, PhysMamba, PhysNet, RhythmFormer, TS_CAN, iBVPNet)
- Fixed SLURM port conflicts with dynamic master_port allocation
- Updated CLAUDE.md with GPU scheduling policy for development runs

---

## 2026-02-11 — Branch: `HPC`

Migrated all 7 main model trainers to DDP and validated the full UBFC-rPPG pipeline end-to-end.

### Changes
- Migrated EfficientPhys, TS_CAN, PhysNet, PhysMamba, PhysFormer, and RhythmFormer trainers to DDP
- Added UBFC-rPPG DeepPhys validation config and SLURM scripts (DeepPhys was already DDP-ready)
- Created `.configs/` directory with UBFC-rPPG validation YAML configs for all 7 models
- Validated full pipeline (preprocess, train, validate, model selection, test, metrics, Bland-Altman plots) for each model on V100 and A100 GPUs

---

## 2026-02-09 — Branch: `HPC`

Merged upstream changes from the original rPPG-Toolbox repository.

### Changes
- Merged upstream/main to bring in LADH and SUMS dataset support (upstream PR #405)

---

## 2025-11-06 to 2025-11-19 — Branch: `HPC`

Integrated the PhysHydra model and iterated on the Neckflix data loader and training pipeline.

### Changes
- Added PhysHydra model and trainer to the repository
- Registered PhysHydra as a model option in main.py with a global NUM_WORKERS argument
- Updated Neckflix loader with begin/end parameters, data_format support, squeeze fix, and unnormalised trace handling
- Fixed memory leakage in training loop and added config file saving to output directories
- Cleaned up loss function naming and removed debug memory printouts
- Added uv.lock to .gitignore

---

## 2025-09-05 to 2025-09-08 — Branch: `HPC`

Initial HPC branch setup with Neckflix dataset integration and environment configuration.

### Changes
- Created the HPC branch from the MacOS development branch
- Added NeckflixLoader with base HDF5 loading functionality for multimodal data (RGB/IR/Depth + ABP/CVP/ECG)
- Registered Neckflix dataset in main.py with default train/test splits
- Added initial experiment configs for Neckflix
- Configured HPC environment: pyproject.toml, adjusted Python version and requirements, updated setup.sh
- Set up .gitignore for venv and cache directories
- Reorganized test configs out of root directory

---

## Pre-HPC — Upstream contributions (selected)

Notable upstream changes present in the fork prior to HPC branch creation.

### Changes
- PhysFormerTrainer update (upstream PR #404, 2025-09-01)
- PhysDrive dataset support (upstream PR #393, 2025-05-25)
- Retinaface removal and documentation housekeeping (upstream PR #384, 2025-04-15)
- FactorizePhys model addition (upstream PR #371, 2025-03-08)
- RhythmFormer and PhysMamba fixes (upstream PR #369, 2025-03-01)
- Test-time chunk length fixes (upstream PR #362, 2025-02-23)
