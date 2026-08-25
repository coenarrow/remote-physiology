# Changelog

This file tracks development milestones on the HPC branch. The main branch tracks the upstream rPPG-Toolbox fork and is kept stable. All active development occurs on HPC and is periodically merged.

**Format:** Each entry records the date, the branch or development context, and a summary of changes. Entries are listed in reverse chronological order (most recent first).

---

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
