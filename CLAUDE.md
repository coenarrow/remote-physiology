# rPPG-Toolbox Fork: Pressure Estimation Extension

## Project Mission
Extending the rPPG-Toolbox to support cardiovascular pressure waveform estimation (ABP, CVP) using the multimodal Neckflix dataset. This involves dataset integration, novel model development, custom metrics, and large-scale HPC validation.

## Quick Start
- **Main entry points**: `main.py` (original), `neckflix_main.py` (Neckflix-specific)
- **Run with**: `uv run python <script>.py --config_file <path/to/config.yaml>`
- **SLURM scripts**: `.slurm_scripts/` for HPC job submission
- **Configs**: `configs/train_configs/`, `configs/infer_configs/`, `physhydra_configs/`

## Repository Context
- **Base**: Fork of [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)
- **Original purpose**: Camera-based heart rate estimation via rPPG
- **Our extension**: Pressure waveform estimation (ABP, CVP) with multimodal inputs
- **Environment**: HPC cluster with SLURM, multi-GPU (H100) training via `uv`

## Current Development Phase
- Integrating Neckflix dataset (multimodal: RGB/IR/Depth + ABP/CVP/ECG traces)
- Developing/extending models for pressure estimation
- Defining custom metrics for pressure waveform validation
- Setting up SLURM workflows (LOSO validation, hyperparameter sweeps, distributed training)

## Codebase Structure

### Core Components
- **`config.py`**: Central configuration management (parameters, paths, model settings)
- **`main.py`**: Original toolbox entry point (supervised/unsupervised methods)
- **`neckflix_main.py`**: Neckflix-specific entry point with custom data handling

### Data Pipeline
- **`dataset/`**: Data loaders for various datasets
  - `data_loader/`: Dataset-specific loaders (BaseLoader pattern)
  - Neckflix integration: Custom loader for HDF5 multimodal data
- **Preprocessing**: Configured via YAML (`DO_CHUNK`, `CHUNK_LENGTH`, `DO_CROP_FACE`, etc.)
- **Output**: Cached preprocessed data in `CACHED_PATH` as `.npy` files

### Model Development
- **`neural_methods/model/`**: Neural network architectures
  - Existing: PhysMamba, PhysFormer, PhysHydra, TS_CAN, DeepPhys, etc.
  - Pattern: Each model is a standalone `.py` file with forward pass logic
- **`neural_methods/trainer/`**: Training/validation/testing routines
  - Pattern: Each model has corresponding `<Model>Trainer.py`
  - Required methods: `__init__`, `train()`, `valid()`, `test()`, `save_model()`
- **`neural_methods/loss/`**: Loss functions for training

### Evaluation
- **`evaluation/metrics.py`**: Metrics computation (MAE, RMSE, MAPE, Pearson, SNR, BA)
- **`evaluation/post_process.py`**: Signal post-processing for evaluation
- **`evaluation/BlandAltmanPy.py`**: Bland-Altman analysis

### Unsupervised Methods
- **`unsupervised_methods/`**: Traditional signal processing (ICA, POS, CHROM, GREEN, etc.)

### Configuration
- **`configs/train_configs/`**: Training experiment configs
- **`configs/infer_configs/`**: Inference/testing configs
- **`physhydra_configs/`**: PhysHydra-specific configs (pressure estimation)

### HPC & Execution

- **`.slurm_scripts/`**: SLURM batch scripts for cluster jobs (multi-GPU via `torch.distributed.run`)
- Partitions, GPU selection, and job submission: see the [running-hpc-jobs](.claude/skills/running-hpc-jobs/SKILL.md) skill

## Development Workflows

### Adding a New Dataset

**Pattern**: Follow `dataset/data_loader/BaseLoader.py` interface

1. **Create loader**: `dataset/data_loader/<Name>Loader.py`
2. **Implement required methods**:
   - `preprocess_dataset(config_preprocess)`: Convert raw data to preprocessed format
   - `read_video(video_file)`: Load video frames
   - `read_wave(bvp_file)`: Load the physiological trace
3. **Optional overrides**: `__len__`, `__getitem__`, `save`, `load` (generally not recommended)
4. **Register** in `main.py`'s `LOADER_REGISTRY`
5. **Create YAML configs**: Define preprocessing and training parameters

**Neckflix does NOT follow this pattern.** Its cache is an external input (zarr
stores written by `ghcr.io/coenarrow/neckflix`), so `NeckflixDataset` is a lazy
`torch.utils.data.Dataset` over that cache with no preprocessing step of its
own, and it emits nested dicts rather than the tuple contract. It is reached
through `neckflix_main.py`, not `main.py`. See **The Neckflix Batch-Dict
Contract** below and `docs/architecture.md`.

**A second zarr-cached dataset** needs only a `channel_map`:

```python
class MyDataset(BaseZarrDataset):
    @property
    def channel_map(self):
        return {"R": ("rgb", 0), "G": ("rgb", 1), "B": ("rgb", 2)}
```

### Adding a New Neural Model

**Pattern**: Model definition + Trainer implementation + Config file

1. **Define model**: Create `neural_methods/model/<ModelName>.py`
   - Inherit from `torch.nn.Module`
   - Implement `forward()` method
   - See existing models (PhysMamba, PhysFormer) as templates

2. **Create trainer**: Create `neural_methods/trainer/<ModelName>Trainer.py`
   - Required methods:
     ```python
     def __init__(self, config, data_loader)
     def train(self, data_loader)
     def valid(self, data_loader)
     def test(self, data_loader)
     def save_model(index)
     ```
   - Handle loss computation, optimization, logging

3. **Register in main.py**: Add logic in `train_and_test()` and `test()` functions

4. **Create config**: New YAML in `configs/train_configs/` or `configs/infer_configs/`

**For pressure estimation models**:
- Consider multi-output architectures (ABP + CVP simultaneous prediction)
- Adapt loss functions for waveform regression vs heart rate estimation
- Handle different sampling rates between modalities

### Adding New Metrics

**Pattern**: Extend `evaluation/metrics.py`

1. **Define metric function**: Add to `evaluation/metrics.py`
   - Input: predictions, ground truth (both as numpy arrays)
   - Output: metric value(s)

2. **Register metric**: Update metric dispatcher in evaluation code

3. **Update config**: Add metric name to `METRICS` list in YAML configs

**For pressure waveforms**:
- Consider waveform morphology (systolic/diastolic detection)
- Clinical relevance (pressure ranges, agreement metrics)
- Signal quality (correlation, spectral analysis)
- Bland-Altman for pressure agreement (existing BA code can be adapted)

### SLURM Job Management

**Use the [running-hpc-jobs](.claude/skills/running-hpc-jobs/SKILL.md) skill.** It covers partition and GPU
selection, `.slurm` script conventions and log naming, multi-GPU distributed training, LOSO job
arrays, interactive sessions, monitoring, and troubleshooting.

## Key Files and Conventions

### Configuration Files (YAML)

**Structure**: All experiments controlled via YAML configs

**Key parameters**:
- **`TOOLBOX_MODE`**: `train_and_test` or `only_test`
- **`TRAIN/VALID/TEST DATA`**:
  - `DATA_PATH`: Raw data location
  - `CACHED_PATH`: Preprocessed data output
  - `BEGIN` & `END`: Data split percentages (e.g., 0.0-0.8 for 80% train)
  - `DO_CHUNK`: Split videos into chunks
  - `CHUNK_LENGTH`: Frames per chunk
  - `DO_CROP_FACE`: Enable face detection
  - `BACKEND`: Face detection method (HC=Haar Cascade, Y5F=YOLO5Face)
- **`MODEL`**: Architecture selection and hyperparameters
- **`METRICS`**: List of evaluation metrics ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'BA']
- **`LOG.PATH`**: Output directory for runs (default: `runs/exp`)

**For Neckflix/Pressure estimation**: see `configs/neckflix/`. Keys the zarr
loader reads (all four data blocks carry them):

- `PREPROCESS.CHANNELS` -- ordered camera channels, subset of `R,G,B,I,D`
- `PREPROCESS.TRACES` -- signals to predict, e.g. `['ABP','CVP','ECG']`
- `PREPROCESS.CHUNK_LENGTH` / `CHUNK_STRIDE` -- window size and stride in frames
  (stride `0` means "no overlap")
- `PREPROCESS.RESIZE.H/W` -- what the model sees; resizing happens
  consumer-side, so it need not match the cache resolution
- `PREPROCESS.DATA_TYPE` -- `Raw` / `Standardized` / `DiffNormalized`, applied
  consumer-side and concatenated along channels if several are listed
- `NECKFLIX.LABEL_NORM` -- `zscore` or `minmax`, per window
- `NECKFLIX.ALLOW_MISSING` / `MIN_CHANNELS` / `MIN_LABELS` -- keep recordings
  that lack some traces (the normal case: trace coverage varies per recording)
- `NECKFLIX.POSTURES` / `PERSPECTIVES` / `LIGHT` / `SESSIONS` / `PARTICIPANTS` --
  attribute include filters; `[]` means no filter
- `TRAIN.LOSS` -- `negpearson` or `mse`, the base of the masked multi-signal loss

`DO_PREPROCESS`, `BEGIN`/`END`, `FILE_LIST_PATH` and the face-detection block do
nothing for Neckflix.

### Python Entry Points

**`main.py`**: Original toolbox entry
- Supports all standard datasets (UBFC-rPPG, PURE, SCAMPS, etc.)
- Both supervised and unsupervised methods

**`neckflix_main.py`**: Neckflix entry (zarr cache, batch-dict contract)
- LOSO splits by participant filter (`--test_participants P015`), no percentage slicing
- DDP optional: used under `torch.distributed.run`, skipped otherwise
- `--limit_windows N` subsamples evenly for smoke runs
- Modes: `train_and_test`, `only_test`, `unsupervised_method`

**Usage**:
```bash
# Single GPU
uv run python neckflix_main.py --config_file path/to/config.yaml

# Multi-GPU distributed
uv run python -m torch.distributed.run --nproc_per_node=4 \
    neckflix_main.py --config_file path/to/config.yaml

# With participant selection (LOSO)
uv run python neckflix_main.py --config_file path/to/config.yaml \
    --test_participants P001
```

### Directory Conventions

**Outputs**:
- **`runs/`**: Training runs, logs, model checkpoints
- **`logs/`**: SLURM job outputs (*.out, *.err files)
- **`model_outputs/`**: Saved model predictions
- **`CACHED_PATH/DataFileLists/`**: CSV files listing train/val/test splits

**Code organization**:
- **Models**: One file per architecture in `neural_methods/model/`
- **Trainers**: One file per model in `neural_methods/trainer/`
- **Losses**: Shared loss functions in `neural_methods/loss/`
- **Data loaders**: One file per dataset in `dataset/data_loader/`

### Naming Conventions

**Models**: PascalCase (e.g., `PhysMamba.py`, `PhysFormer.py`)
**Trainers**: `<Model>Trainer.py` (e.g., `PhysMambaTrainer.py`)
**Configs**: `<DATASET>_<TASK>_<MODEL>.yaml` or semantic names
**SLURM scripts**: `<Dataset>_<Model>_<Options>.slurm` (e.g., `UBFC-rPPG_DeepPhys_2GPU.slurm`)

### Git Workflow

**Recent commits show**:
- Active development on memory optimization and loss functions
- Neckflix loader integration
- Config management improvements
- PhysMamba trainer updates

**Branch**: Currently on `main`
**Untracked files to consider**: Check if development files should be committed or .gitignored

## Common Tasks and Patterns

### Running an Experiment

**IMPORTANT**: We are on a **login node** - NEVER run computational tasks directly. Always use SLURM.

See the [running-hpc-jobs](.claude/skills/running-hpc-jobs/SKILL.md) skill for `sbatch` submission, `salloc`
interactive sessions, and job monitoring.

### Preprocessing Data

**First time only**: Set in YAML config:
```yaml
TRAIN:
  DO_PREPROCESS: True
  DATA_PATH: /path/to/raw/data
  CACHED_PATH: /path/to/preprocessed/output
```

**Subsequent runs**: Turn off preprocessing:
```yaml
TRAIN:
  DO_PREPROCESS: False  # Load from CACHED_PATH instead
```

**Note**: Preprocessing happens once and caches to `.npy` files for efficiency

### Model Development Workflow

1. **Create model**: `neural_methods/model/MyModel.py`
2. **Create trainer**: `neural_methods/trainer/MyModelTrainer.py`
3. **Update main.py**: Add model to train_and_test() function
4. **Create config**: New YAML file with model parameters
5. **Test with salloc**: Interactive session with small dataset/short run to verify
6. **Submit to SLURM**: Full training run via sbatch

### Extending Existing Models

**Pattern**: Modify architecture while reusing trainer logic
- Copy existing model file as template
- Modify architecture (layers, attention, etc.)
- Adjust trainer if loss/optimization changes
- Update config with new hyperparameters

**Example**: Adapting PhysMamba for multi-output (ABP + CVP)
- Modify output head to produce multiple waveforms
- Adjust loss to handle multiple targets
- Update config to specify output types

### Debugging Distributed Training

Port conflicts, GPU count mismatches, missing CUDA modules, and OOM errors are covered in the
[running-hpc-jobs](.claude/skills/running-hpc-jobs/SKILL.md) skill.

### Experiment Tracking

**Outputs**:
- Model checkpoints: Saved to `LOG.PATH` (default: `runs/exp`)
- Training curves: Loss plots auto-saved if `PLOT_LOSSES_AND_LR: True`
- Predictions: Saved to `TEST.OUTPUT_SAVE_DIR` if specified
- Metrics: Printed to stdout and SLURM logs

**File lists**: Check `CACHED_PATH/DataFileLists/` for train/val/test splits used

### The Neckflix Batch-Dict Contract

Everything downstream of the Neckflix loader passes nested dicts keyed by
canonical channel and signal names, so any tensor is identifiable by its key at
any point. `neural_methods/batch.py` owns the key names and the shape moves.

**Per sample** (dataset `__getitem__`; `default_collate` adds the batch axis):

```python
{"frames":       {ch:  (1, T, H, W) float32},   # raw pixels, zero-filled where absent
 "labels":       {sig: (T,)         float32},   # per-window normalised
 "label_stats":  {sig: {stat: ()    float32}},  # physical units, for exact inversion
 "channel_mask": {ch:  ()           bool},      # True = real data, not zero fill
 "label_mask":   {sig: ()           bool},
 "metadata":     {"recording_id": str, "camera_id": str, "start_frame": int}}
```

**Through a model** -- the same dict comes back with `predictions` added:

```python
out = model(batch)
out["predictions"]                  # {signal: (B, T)}
out["frames"] is batch["frames"]    # nothing is dropped in transit
```

**Rules**
- Channel and signal *order* lives on the model (`model.channels`,
  `model.traces`); dict iteration order is never load-bearing.
- **Use einops for every reshape** (`rearrange` / `reduce` / `einsum`), not
  `view` / `permute` / `reshape`.
- Frame preprocessing is consumer-side: the loader emits raw pixels and
  `neural_methods/frame_transforms.py` applies `DATA_TYPE` + resize, carried by
  the model.
- `label_mask` is load-bearing, not an edge case: trace coverage genuinely
  varies per recording, and `MaskedMultiSignalLoss` makes an absent signal
  contribute exactly 0 rather than NaN.

**Reading the cache directly** (for inspection):

```python
import zarr
root = zarr.open_group("/path/to/cache/P015_S01_R3_0_D.zarr", mode="r")
dict(root.attrs)                      # recording, participant "015", posture, ...
root["1"]["rgb"]["video"]["frames"]   # (C, T, H, W) uint8
root["1"]["rgb"]["abp"]["data"]       # (T,) float64, physical units
```

### Running Neckflix experiments

```bash
# All seven unsupervised methods (CPU), scored per trace
uv run python neckflix_main.py --config_file configs/neckflix/NECKFLIX_UNSUPERVISED.yaml

# PhysMamba, one LOSO fold
uv run python neckflix_main.py --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml --test_participants P015

# Multi-GPU
uv run python -m torch.distributed.run --nproc_per_node=4 neckflix_main.py --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml --test_participants P015

# Enumerate LOSO folds (metadata-only; safe on a login node)
uv run python tools/list_neckflix_folds.py --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml --prefix P

# Summarise a finished run (per-signal, physical units; --by adds participant etc.)
uv run python tools/summarise_neckflix_outputs.py runs/neckflix_physmamba --by signal participant --csv windows.csv

# Smoke run before submitting anything
uv run python neckflix_main.py --limit_windows 8 --test_participants P015 --config_file configs/neckflix/NECKFLIX_PHYSMAMBA_SMOKE.yaml
```

**Adding a model to the Neckflix pipeline**: no new trainer needed.
`MultiSignalTrainer` serves every dict-contract model.
1. Make the architecture a `DictModel` implementing
   `forward_video(video) -> (B, S, T)`, taking its width from
   `self.in_channels` / `self.out_signals`. A per-frame 2-D backbone needs no
   change at all -- wrap it in `SignalDictWrapper(..., input_mode='frames2d')`.
2. Add a builder to `MODEL_REGISTRY` in
   `neural_methods/trainer/MultiSignalTrainer.py`.
3. Point a config at it with `MODEL.NAME`.

## Dataset Locations

| Dataset | Path | Notes |
|---------|------|-------|
| Neckflix (raw) | `/group/pgh004/carrow/repo/Neckflix/dataset` | Raw captures; input to the external preprocessor |
| Neckflix (zarr cache) | set by `CACHED_PATH` | One `*.zarr` per recording; what this repo reads |
| PURE | `/group/pgh004/carrow/zipped_datasets/PURE` | Standard rPPG dataset |
| UBFC-rPPG | `/group/pgh004/carrow/zipped_datasets/UBFC-rPPG` | Standard rPPG dataset |

Additional datasets will be added to `/group/pgh004/carrow/zipped_datasets/` over time.

## Project-Specific Notes

### Neckflix Dataset Characteristics

**What this repo reads**: a zarr cache, one store per recording, written by the
external preprocessor (`ghcr.io/coenarrow/neckflix` >= 1.0.0) and pointed at by
`CACHED_PATH`. This repo never writes it. Stores are admitted only if their root
attrs carry `complete: true` and `tool_version >= 1.0.0`; anything else is
skipped with a warning (pre-1.0.0 frames were delta-encoded and would decode to
garbage).

**Store layout**: `{recording}.zarr` -> perspective (`"1"`/`"2"`) -> stream
(`rgb`/`ir`/`depth`) -> `video/frames` `(C, T, H, W)` plus `{abp,cvp,ecg}/data`
`(T,)` in physical units, index-aligned to the frames. Root attrs carry
`participant` **unprefixed** (`"015"`, not `P015`).

**Modalities**: RGB, IR, Depth. **Traces**: ABP, CVP, ECG.

**Key differences from standard rPPG datasets**:
- Multimodal camera inputs (not just RGB)
- Pressure waveforms instead of/in addition to PPG
- **Trace coverage varies per recording** -- some have ABP+CVP, some CVP+ECG,
  some all three. `ALLOW_MISSING` + `label_mask` is the normal configuration.
- Two camera perspectives per recording, treated as independent samples
- Splits are participant filters, not `BEGIN`/`END` percentages

### Pressure Estimation vs Heart Rate

**Key differences**:
- **Output**: Full waveform prediction (ABP/CVP) vs single HR value
- **Loss functions**: Waveform regression loss vs rate estimation loss
- **Metrics**: Morphology, pressure ranges, clinical agreement (TBD)
- **Temporal resolution**: Higher sampling rate for waveform details

### Known Issues and Considerations

**Memory management**:
- Recent commits address memory leaks and GPU memory issues
- Monitor memory usage, especially with multi-GPU training
- May need to adjust chunk sizes for different GPU types (V100 vs A100 vs H100)

**Config management**:
- Multiple config systems in use (YAML + Hydra)
- Hydra configs live in `physhydra_configs/`
- Config files now saved to output directories (recent change)

**Data synchronization**:
- Critical for multimodal inputs and physiological traces
- Ensure temporal alignment in preprocessing
- Different sampling rates require interpolation

### Related Repositories

**Neckflix** (`/mmfs1/data/group/pgh004/carrow/repo/Neckflix`):
- Preprocessing pipeline for raw Kinect Azure data
- HDF5 data generation scripts
- Configuration examples for different modalities and traces
- Utilities for frame processing and trace filtering

**Cross-reference**:
- Preprocessing logic from Neckflix may inform rPPG-Toolbox loader
- Config patterns (YAML structure) similar between repos
- Both use `uv` for package management

## Resources and References

### Documentation

**rPPG-Toolbox**:
- Original repo: https://github.com/ubicomplab/rPPG-Toolbox
- Paper: [rPPG-Toolbox: Deep Remote PPG Toolbox](https://arxiv.org/abs/2210.00716)
- README: `/mmfs1/data/group/pgh004/carrow/repo/rPPG-Toolbox/README.md`

**Key papers for implemented models**:
- PhysMamba: [PhysMamba: Efficient Remote Physiological Measurement with SlowFast Temporal Difference Mamba](https://doi.org/10.48550/arXiv.2409.12031)
- PhysFormer: [PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_PhysFormer_Facial_Video-Based_Physiological_Measurement_With_Temporal_Difference_Transformer_CVPR_2022_paper.pdf)
- BigSmall: [BigSmall: Efficient Multi-Task Learning for Disparate Spatial and Temporal Physiological Measurements](https://arxiv.org/abs/2303.11573)
- See README.md for complete algorithm list

### HPC Environment

SLURM cluster. Partitions, modules, GPU resources, and cluster paths are documented in the
[running-hpc-jobs](.claude/skills/running-hpc-jobs/SKILL.md) skill.

### Python Environment

**Package manager**: `uv` (fast Python package manager)
- Install dependencies: `uv sync` (the older `setup.sh uv` predates `pyproject.toml`)
- Run scripts: `uv run python <script.py>`
- Virtual env: `.venv/` (auto-managed by uv)

**Platform split for the Mamba kernels** — `mamba-ssm` publishes Linux wheels
only, and its CUDA sources do not compile with MSVC as published:

| | source of `mamba-ssm` | `causal-conv1d` | `triton` |
|---|---|---|---|
| Linux | PyPI (prebuilt wheel via its `setup.py`) | PyPI sdist | PyPI |
| Windows | `vendor/mamba-ssm` (patched, compiled here) | PyPI sdist | `triton-windows` |
| macOS | `mamba-ssm-macos` | — | — |

`tools/vendor_mamba_windows.py` regenerates the vendored tree and documents the
three patches; `--check` verifies it still matches upstream. Do not hand-edit
`vendor/mamba-ssm`. `override-dependencies` in `pyproject.toml` confines
`triton` to Linux — without it nothing installs on Windows at all, since
`triton` has neither a Windows wheel nor an sdist.

Where no `mamba_ssm` is installed, `mamba_compat.MambaRef` stands in: a
pure-PyTorch selective-SSM block, parameter-compatible with the real one, that
trains but materialises the hidden state. Correctness path, not a speed one.

**Key dependencies**:
- PyTorch (with CUDA support)
- `hdf5plugin` (for Neckflix HDF5 data)
- See `requirements.txt` and `pyproject.toml`

### Common Datasets (Reference)

While our focus is Neckflix, the toolbox supports:
- SCAMPS (synthetic data)
- UBFC-rPPG
- PURE
- BP4D+
- UBFC-Phys
- MMPD
- iBVP
- PhysDrive

See README.md for dataset structure and citations.

### Useful Commands

**SLURM, interactive sessions, and job monitoring**: see the [running-hpc-jobs](.claude/skills/running-hpc-jobs/SKILL.md) skill.

**Git**:
```bash
git log --oneline -10        # Recent commits
git status                   # Current changes
git diff                     # View modifications
```

---

## Quick Reference

**Start new experiment**: Create YAML config → Test with `salloc` → Submit with `sbatch`

**Add new model**: Model file + Trainer file + Update main.py + Create config

**Debug job**: Check `logs/<job>.err` → Test with `salloc` → Adjust config/code

**Last updated**: 2026-08-30
