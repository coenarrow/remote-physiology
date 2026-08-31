# Architecture Overview

## System Overview

This repository is a fork of the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox), originally designed for camera-based remote photoplethysmography (rPPG) -- estimating heart rate from facial video by detecting subtle skin color changes caused by blood flow.

Our extension adds support for **cardiovascular pressure waveform estimation** (Arterial Blood Pressure / ABP, Central Venous Pressure / CVP) using the multimodal **Neckflix dataset**. This shifts the output from a single heart rate value to full waveform prediction, and the input from RGB-only facial video to multimodal camera streams (RGB, infrared, depth) captured at the neck region.

### Key References

- Original paper: [rPPG-Toolbox: Deep Remote PPG Toolbox](https://arxiv.org/abs/2210.00716)
- Original repository: [ubicomplab/rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)


## Data Flow

The pipeline follows a linear flow from raw data to evaluation outputs:

```
Raw Data (video files or HDF5)
    |
    v
Preprocessing
    - Face detection (Haar Cascade or YOLO5Face)
    - Chunking (splitting videos into fixed-length segments)
    - Normalization (DiffNormalized, Standardized, etc.)
    - Resizing (e.g., 72x72)
    |
    v
Cached .npy Files (in CACHED_PATH)
    - Preprocessed video chunks
    - Corresponding label chunks (BVP, ABP, CVP, etc.)
    - File lists in CACHED_PATH/DataFileLists/ (CSV train/val/test splits)
    |
    v
DataLoader (dataset-specific loader following BaseLoader interface)
    |
    v
Model (neural network forward pass)
    |
    v
Evaluation
    - Metrics: MAE, RMSE, MAPE, Pearson correlation, SNR
    - Bland-Altman analysis and plots
    - Post-processing: FFT or peak detection for heart rate derivation
    |
    v
Saved Outputs (runs/exp/)
    - Model checkpoints (best epoch selected via validation)
    - Loss plots (if PLOT_LOSSES_AND_LR: True)
    - Predictions and metrics in logs
```

Preprocessing runs once and caches results. Subsequent runs load directly from the cache, controlled by the `DO_PREPROCESS` flag in the YAML config. Each unique combination of `DATA_TYPE`, `CHUNK_LENGTH`, `RESIZE`, and split range generates its own cache.

**Neckflix takes a different route.** Its cache is an *external input*: zarr stores written by the Neckflix preprocessor container, which this repo reads and never writes. There is no `.npy` cache, no file-list CSVs, and no `DO_PREPROCESS` step:

```
Neckflix preprocessor (ghcr.io/coenarrow/neckflix, external)
    |
    v
Zarr cache: one *.zarr store per recording (raw frames + traces)
    |
    v
NeckflixDataset -- metadata-only construction, lazy per-window reads
    - participant/posture/light/perspective filters (the LOSO mechanism)
    - strided or random windows over frames
    |
    v
Batch dict {frames, labels, label_stats, channel_mask, label_mask, metadata}
    |
    v
DictModel  -- frame transform (resize + DATA_TYPE) then the architecture
    |         returns the same dict plus "predictions": {signal: (B, T)}
    v
MaskedMultiSignalLoss / per-signal evaluation
```

See "The Neckflix Batch-Dict Contract" below.


## Directory Layout

```
rPPG-Toolbox/
|-- config.py                  Central configuration management
|-- main.py                    Original toolbox entry point
|-- neckflix_main.py           Neckflix-specific entry point
|
|-- dataset/
|   |-- data_loader/
|       |-- BaseLoader.py      Abstract base class for all data loaders
|       |-- UBFCrPPGLoader.py  UBFC-rPPG dataset loader
|       |-- PURELoader.py      PURE dataset loader
|       |-- zarr_dataset.py   BaseZarrDataset: lazy loader over zarr caches
|       |-- NeckflixLoader.py  NeckflixDataset (channel map only)
|       |-- neckflix_config.py yacs config -> zarr loader plain-dict config
|       |-- label_transforms.py Per-window label normalisation + inverses
|       |-- ...                Other dataset loaders (SCAMPS, BP4D+, MMPD, etc.)
|       |-- face_detector/     Face detection backends (Haar Cascade, YOLO5Face)
|
|-- neural_methods/
|   |-- batch.py               The Neckflix batch-dict contract (keys + einops moves)
|   |-- frame_transforms.py    Consumer-side DATA_TYPE + resize on (B,C,T,H,W)
|   |-- signals.py             Canonical signal/channel registries
|   |
|   |-- model/                 Neural network architectures (one file per model)
|   |   |-- DictModel.py       Base: batch dict in, batch dict + predictions out
|   |   |-- SignalDictWrapper.py  Adapter for 2-D/3-D backbones
|   |   |-- mamba_compat.py    mamba_ssm shim + pure-torch MambaRef fallback
|   |   |-- DeepPhys.py
|   |   |-- EfficientPhys.py
|   |   |-- TS_CAN.py
|   |   |-- PhysNet.py
|   |   |-- PhysMamba.py
|   |   |-- PhysFormer.py
|   |   |-- RhythmFormer.py
|   |   |-- BigSmall.py
|   |   |-- iBVPNet.py
|   |   |-- FactorizePhys/     (directory-based model with submodules)
|   |   |-- PhysHydra.py       Pressure estimation model
|   |
|   |-- trainer/               Training/validation/testing routines (one per model)
|   |   |-- BaseTrainer.py     Shared DDP setup, rank management, model unwrapping
|   |   |-- MultiSignalTrainer.py  One trainer for every dict-contract model
|   |   |-- DeepPhysTrainer.py
|   |   |-- PhysMambaTrainer.py
|   |   |-- PhysHydraTrainer.py
|   |   |-- ...
|   |
|   |-- loss/                  Loss function implementations
|       |-- NegPearsonLoss.py
|       |-- PhysFormerLossComputer.py
|       |-- PhysHydraLoss.py
|       |-- PhysNetNegPearsonLoss.py
|       |-- MaskedMultiSignalLoss.py
|       |-- RythmFormerLossComputer.py
|
|-- evaluation/
|   |-- metrics.py             Metric computation (MAE, RMSE, MAPE, Pearson, SNR, BA)
|   |-- post_process.py        Signal post-processing (FFT, peak detection)
|   |-- metrics_report.py      Shared HR-metric aggregation and printing
|   |-- BlandAltmanPy.py       Bland-Altman agreement analysis and plotting
|   |-- bigsmall_multitask_metrics.py
|
|-- unsupervised_methods/      Traditional signal processing methods
|   |-- methods/
|   |   |-- CHROME_DEHAAN.py
|   |   |-- GREEN.py
|   |   |-- ICA_POH.py
|   |   |-- LGI.py
|   |   |-- OMIT.py
|   |   |-- PBV.py
|   |   |-- POS_WANG.py
|   |-- unsupervised_predictor.py
|   |-- utils.py
|
|-- configs/
|   |-- train_configs/         YAML configs for training experiments
|   |-- infer_configs/         YAML configs for inference/testing
|
|-- .configs/                  Active experiment configs (UBFC-rPPG validation set)
|-- physhydra_configs/         PhysHydra-specific configs for pressure estimation
|
|-- .slurm_scripts/            SLURM batch scripts for HPC job submission
|-- z_slurm_scripts/           Additional SLURM scripts (LOSO, sweeps)
|-- logs/                      SLURM job stdout/stderr files
|-- runs/                      Training run outputs, checkpoints, loss plots
|-- model_outputs/             Saved model predictions
|-- tools/                     Utility scripts
```


## Configuration System

All experiments are controlled via YAML configuration files. The `config.py` module parses these files and provides a unified configuration object to the rest of the codebase.

### Key Configuration Parameters

**Top-level:**
- `TOOLBOX_MODE`: Either `train_and_test` (full pipeline) or `only_test` (inference only)
- `DEVICE`: Target device (e.g., `cuda:0`)
- `NUM_OF_GPU_TRAIN`: Legacy DataParallel setting; DDP trainers derive GPU count from `--nproc_per_node`

**Data sections (TRAIN / VALID / TEST):**
- `DATA_PATH`: Path to raw dataset files
- `CACHED_PATH`: Path for preprocessed .npy output (and DataFileLists/)
- `DATASET`: Dataset identifier (e.g., `UBFC-rPPG`, `PURE`, `Neckflix`)
- `BEGIN` / `END`: Fractional range for data splitting (e.g., 0.0--0.8 for 80% train)
- `DO_PREPROCESS`: Whether to run preprocessing (skipped if cache exists)
- `DATA_FORMAT`: Tensor layout (e.g., `NDCHW`)
- `FS`: Sampling frequency in Hz

**Preprocessing (nested under PREPROCESS):**
- `DATA_TYPE`: Normalization methods (e.g., `['DiffNormalized', 'Standardized']`)
- `LABEL_TYPE`: Label normalization (e.g., `DiffNormalized`)
- `DO_CHUNK` / `CHUNK_LENGTH`: Whether to split into fixed-length chunks and their size in frames
- `CROP_FACE`: Face detection settings -- `DO_CROP_FACE`, `BACKEND` (`HC` or `Y5F`), bounding box coefficients, dynamic detection options
- `RESIZE`: Target frame dimensions (`H`, `W`)

**Model:**
- `NAME`: Model architecture identifier (e.g., `DeepPhys`, `PhysMamba`, `PhysFormer`)
- Model-specific hyperparameters (e.g., `DROP_RATE`)

**Training:**
- `BATCH_SIZE`, `EPOCHS`, `LR`
- `MODEL_FILE_NAME`: Checkpoint naming prefix
- `PLOT_LOSSES_AND_LR`: Whether to generate loss/LR plots

**Testing / Inference:**
- `METRICS`: List of metrics to compute (e.g., `['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'BA']`)
- `USE_LAST_EPOCH`: If false, uses validation-based best epoch selection
- `EVALUATION_METHOD`: `FFT` or `peak detection` for heart rate derivation
- `EVALUATION_WINDOW`: Optional sliding window evaluation

**Logging:**
- `LOG.PATH`: Output directory for runs (default: `runs/exp`)


## Model / Trainer / Loader Patterns

The codebase follows a consistent one-to-one-to-one pattern between models, trainers, and data loaders.

### Models (`neural_methods/model/`)

Each model is a standalone Python file containing a `torch.nn.Module` subclass with a `forward()` method. Models are purely architectural -- they define the network structure and forward pass but contain no training logic.

Current models: DeepPhys, EfficientPhys, TS_CAN, PhysNet, PhysMamba, PhysFormer, RhythmFormer, BigSmall, iBVPNet, FactorizePhys, PhysHydra.

### Trainers (`neural_methods/trainer/`)

Each model has a corresponding trainer file (e.g., `PhysMambaTrainer.py`) that handles the training loop, validation, testing, and model saving. Trainers inherit from `BaseTrainer`, which provides:

- DDP (Distributed Data Parallel) setup: `self.rank`, `self.world_size`, `self.is_main`
- Model unwrapping utility: `self._unwrap_model()` for accessing the underlying model within a DDP wrapper
- Common initialization logic

Required trainer methods:
- `__init__(self, config, data_loader)` -- Setup model, optimizer, loss, scheduler
- `train(self, data_loader)` -- One epoch of training
- `valid(self, data_loader)` -- Validation pass (all ranks participate for `all_reduce`)
- `test(self, data_loader)` -- Final evaluation
- `save_model(index)` -- Checkpoint saving

Only rank 0 performs printing, model saving, testing, and plot generation in multi-GPU runs.

### Data Loaders (`dataset/data_loader/`)

Each dataset has a loader following the `BaseLoader` interface with these key methods:

- `preprocess_dataset(config_preprocess)` -- Convert raw data to preprocessed .npy format
- `read_video(video_file)` -- Load video frames from disk
- `read_wave(bvp_file)` -- Load physiological signal traces

The BaseLoader handles chunking, caching, file list management, and the `__getitem__` / `__len__` interface for PyTorch DataLoaders.

### Registration

Models are registered in `main.py`'s `train_and_test()` and `test()` functions, where the model name from the YAML config is mapped to the appropriate trainer class.


## Entry Points

### `main.py`

The original rPPG-Toolbox entry point. Supports all standard rPPG datasets (UBFC-rPPG, PURE, SCAMPS, BP4D+, UBFC-Phys, MMPD, iBVP, PhysDrive, etc.) and both supervised neural methods and unsupervised signal processing methods.

### `neckflix_main.py`

The Neckflix entry point. Reads the zarr cache through `NeckflixDataset`, splits by participant (LOSO), and dispatches to either `MultiSignalTrainer` or the unsupervised predictor. DDP is optional: it is used when launched under `torch.distributed.run` and skipped otherwise, so the same script runs on a laptop and on a multi-GPU node.

The two entry points do not share a dataset contract: `main.py` serves the upstream tuple-contract loaders, `neckflix_main.py` serves the batch-dict contract described below.

### Usage

```bash
# Upstream datasets, single GPU
uv run python main.py --config_file configs/train_configs/<config>.yaml

# Upstream datasets, multi-GPU
uv run python -m torch.distributed.run --nproc_per_node=N main.py --config_file configs/train_configs/<config>.yaml

# Neckflix: all seven unsupervised methods (CPU), scored per trace
uv run python neckflix_main.py --config_file configs/neckflix/NECKFLIX_UNSUPERVISED.yaml

# Neckflix: PhysMamba, one LOSO fold
uv run python neckflix_main.py --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml --test_participants P015

# Neckflix: same fold across 4 GPUs
uv run python -m torch.distributed.run --nproc_per_node=4 neckflix_main.py --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml --test_participants P015

# Neckflix: enumerate LOSO folds (metadata-only; safe on a login node)
uv run python tools/list_neckflix_folds.py --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml --prefix P

# Neckflix: summarise a finished run offline (per-signal, physical units)
uv run python tools/summarise_neckflix_outputs.py runs/neckflix_physmamba --by signal participant
```

`MultiSignalTrainer` saves one self-describing record per scored window --
prediction, label, and the `label_stats` that normalised them -- so
`tools/summarise_neckflix_outputs.py` recomputes every number, including the
inverse back to physical units, without touching the zarr cache.


## Naming Conventions

| Category | Convention | Examples |
|----------|-----------|----------|
| Models | PascalCase filenames | `PhysMamba.py`, `PhysFormer.py`, `DeepPhys.py` |
| Trainers | `<Model>Trainer.py` | `PhysMambaTrainer.py`, `DeepPhysTrainer.py` |
| Data loaders | `<Dataset>Loader.py` | `UBFCrPPGLoader.py`, `NeckflixLoader.py` |
| YAML configs | `<DATASET>_<TASK>_<MODEL>.yaml` | `UBFC-rPPG_UBFC-rPPG_UBFC-rPPG_DEEPPHYS.yaml` |
| SLURM scripts | `<Dataset>_<Model>_<GPUs>.slurm` | `UBFC-rPPG_DeepPhys_2GPU.slurm` |
| Loss functions | Descriptive names | `NegPearsonLoss.py`, `PhysHydraLoss.py` |


## Extending the Toolbox

### Adding a New Dataset

1. Create a loader in `dataset/data_loader/` following the `BaseLoader` interface
2. Implement `preprocess_dataset()`, `read_video()`, and `read_wave()`
3. Update `config.py` with any new dataset-specific parameters or paths
4. Create YAML configs defining preprocessing and training parameters

### Adding a New Model

For the upstream tuple-contract datasets:

1. Define the architecture in `neural_methods/model/<ModelName>.py` as a `torch.nn.Module`
2. Create `neural_methods/trainer/<ModelName>Trainer.py` inheriting from `BaseTrainer`
3. Register the model in `main.py`'s `TRAINER_REGISTRY`
4. Create a YAML config with model-specific hyperparameters

For Neckflix, no new trainer is needed -- `MultiSignalTrainer` serves every dict-contract model:

1. Make the architecture a `DictModel` subclass implementing `forward_video(video) -> (B, S, T)`, taking its input width from `self.in_channels` and its output width from `self.out_signals`. A 2-D per-frame backbone needs no change at all -- wrap it in `SignalDictWrapper(backbone, channels, traces, input_mode='frames2d')`.
2. Add a builder to `MODEL_REGISTRY` in `neural_methods/trainer/MultiSignalTrainer.py`.
3. Point a config at it with `MODEL.NAME: <ModelName>`.

### Adding New Metrics

1. Add the metric function to `evaluation/metrics.py`
2. Register it in the metric dispatcher within the evaluation code
3. Add the metric name to the `METRICS` list in experiment YAML configs


## Neckflix Dataset

The Neckflix dataset is a multimodal collection for cardiovascular pressure estimation. It reaches this repository as a **zarr cache**, one store per recording, produced by the external Neckflix preprocessor (`ghcr.io/coenarrow/neckflix` >= 1.0.0). This repo reads that cache and never writes it; the HDF5 loader it replaced is gone.

**Cache layout** (`{cache_dir}/{recording}.zarr`, zarr v3):

```
P015_S01_R3_0_D.zarr
  attrs: recording, participant ("015", unprefixed), session, repeat,
         posture, light, source_resolution, resized_to, tool_version, complete
  1/                          <- perspective (camera) key
    rgb/
      video/frames            (C, T, H, W) uint8, chunked (C, 32, H, W)
      video/  attrs: fps, num_frames
      abp/data                (T,) float64, physical units, index-aligned to frames
      cvp/data
      ecg/data
    ir/  depth/               same shape, uint16, C=1
  2/                          second perspective, same structure
```

Stores are admitted only if `complete is true` and `tool_version >= 1.0.0`; anything else is skipped with a warning. Which traces a recording carries **varies** — some have ABP+CVP, some CVP+ECG, some all three — so `ALLOW_MISSING` plus the per-sample `label_mask` is the normal configuration rather than an edge case.

**Camera modalities:** RGB, IR, Depth (event streams are not consumed).
**Physiological traces:** ABP, CVP, ECG.

Construction is metadata-only: no pixel data is read until `__getitem__`, so instantiating a dataset to enumerate LOSO folds is cheap enough for a login node (`tools/list_neckflix_folds.py`).


## The Neckflix Batch-Dict Contract

Everything downstream of the Neckflix loader passes nested dicts keyed by canonical channel and signal names, so any tensor in the pipeline is identifiable by its key. `neural_methods/batch.py` is the single owner of those key names.

**Per sample** (dataset `__getitem__`):

```python
{"frames":       {ch:  (1, T, H, W) float32},   # raw pixel values, zero-filled where absent
 "labels":       {sig: (T,)         float32},   # per-window normalised
 "label_stats":  {sig: {stat: ()    float32}},  # physical units, for exact inversion
 "channel_mask": {ch:  ()           bool},      # True = real data, not zero fill
 "label_mask":   {sig: ()           bool},
 "metadata":     {"recording_id": str, "camera_id": str, "start_frame": int}}
```

`default_collate` handles the nesting: every tensor gains a leading batch axis and the metadata strings become lists. No custom `collate_fn` exists or is needed.

**Through a model:**

```python
out = model(batch)
out["predictions"]                    # {signal: (B, T)}
out["frames"] is batch["frames"]      # everything else passes through untouched
```

A model is a `DictModel` (`neural_methods/model/DictModel.py`). It owns the canonical channel and trace *order*; the dicts themselves are never iterated for order. A subclass implements only:

```python
def forward_video(self, video):   # (B, C_in, T, H, W) -> (B, S, T)
```

so retrofitting an architecture is a signature change, not a rewrite. `C_in` is `len(channels) * frame_transform.channel_multiplier`, since a `DATA_TYPE` of two transforms feeds the backbone two channel blocks of the same clip.

`DictModel.forward` also accepts a plain `(B, C, T, H, W)` tensor and returns a plain tensor, which is how the upstream tuple-contract trainers keep working against the same model classes.

**Frame preprocessing is consumer-side.** The loader emits raw pixels; `neural_methods/frame_transforms.py` applies `Raw` / `Standardized` / `DiffNormalized` and an optional spatial resize on `(B, C, T, H, W)` tensors, carried by the model that needs it. One cache resolution therefore serves models that want another.

**Reshaping is einops.** New and touched code uses `rearrange` / `reduce` / `einsum` rather than `view` / `permute` / `reshape`, so every shape move states its own meaning.

### Training on the contract

`neural_methods/trainer/MultiSignalTrainer.py` is the one trainer for every dict-contract model, driven by a registry:

```python
MODEL_REGISTRY = {'PhysMamba': _build_physmamba, 'DeepPhys': _build_deepphys}
```

Adding a model is a builder function plus a registry line. The trainer owns DDP, AMP, `MaskedMultiSignalLoss` (per-signal masked mean, so a signal absent from a whole batch contributes exactly 0 rather than NaN), checkpointing, and evaluation reported **per predicted signal**.

Each signal is reported three ways: waveform Pearson/MAE/RMSE in normalised units, the same MAE/RMSE in physical units (mmHg for ABP/CVP -- each window's `label_stats` run back through the exact inverse of its normalisation), and the inherited HR metrics. The physical figure is the error given a perfect estimate of that window's scale, so it measures *shape* expressed in the signal's units; absolute level is a separate problem that per-window normalisation deliberately removes.

### Splits

Splits are participant filters, not percentage slices. LOSO is two dataset instances over the same cache:

```python
train_cfg["filters"]["participant"] = {"include": [],      "exclude": ["015"]}
test_cfg["filters"]["participant"]  = {"include": ["015"], "exclude": []}
```

`dataset/data_loader/neckflix_config.py` builds those from the YAML plus `--test_participants`, including the id convention mismatch: the CLI says `P015`, the store's root attr says `"015"`.


## Pressure Estimation vs Heart Rate Estimation

The extension from heart rate estimation to pressure waveform estimation introduces several architectural differences:

| Aspect | Heart Rate (rPPG) | Pressure Waveform (ABP/CVP) |
|--------|-------------------|----------------------------|
| **Output** | Single scalar value (BPM) | Full temporal waveform |
| **Loss functions** | Rate estimation loss | Waveform regression loss (e.g., negative Pearson, MSE on waveform) |
| **Metrics** | MAE/RMSE on BPM | Waveform morphology, pressure ranges, clinical agreement |
| **Temporal resolution** | Aggregated over window | Sample-level prediction at high sampling rate |
| **Input** | RGB facial video | Multimodal (RGB, IR, Depth) at neck region |
| **Post-processing** | FFT or peak detection | Direct waveform output |

Models targeting pressure estimation (e.g., PhysHydra) may use multi-output architectures for simultaneous ABP and CVP prediction, and require adapted loss functions for waveform regression.


## Dataset Locations

| Dataset | Path | Notes |
|---------|------|-------|
| Neckflix (raw) | `/group/pgh004/carrow/repo/Neckflix/dataset` | Raw captures; input to the external preprocessor |
| Neckflix (zarr cache) | set by `CACHED_PATH` in the config | One `*.zarr` store per recording; what this repo reads |
| PURE | `/group/pgh004/carrow/zipped_datasets/PURE` | Standard rPPG dataset |
| UBFC-rPPG | `/group/pgh004/carrow/zipped_datasets/UBFC-rPPG` | Standard rPPG dataset |

Preprocessed caches are stored at paths defined by `CACHED_PATH` in the YAML config, typically under `PreprocessedData/`.


## Python Environment

The project uses **uv** as its Python package manager. The virtual environment (`.venv/`) is auto-managed by uv.

**Key dependencies:**
- PyTorch with CUDA support
- `zarr` (>=3.3, <4) for the Neckflix cache
- `einops` for every shape move in the Neckflix pipeline
- `mamba-ssm` (Linux/CUDA) or `mamba-ssm-macos`; where neither has a wheel,
  `neural_methods/model/mamba_compat.py` falls back to a pure-PyTorch Mamba block
- `h5py` / `hdf5plugin` for the other datasets' loaders
- Standard scientific Python stack (numpy, scipy, matplotlib)

All scripts are invoked via `uv run python` to ensure the correct environment is used. Full dependency specifications are in `pyproject.toml` and `requirements.txt`, with the lockfile at `uv.lock`.


## Related Repositories

**Neckflix** (`/mmfs1/data/group/pgh004/carrow/repo/Neckflix`, container `ghcr.io/coenarrow/neckflix`):
- Raw data preprocessing pipeline for Kinect Azure captures
- Writes the zarr cache this repo consumes; the coupling is the store schema plus two root-attr gates (`complete`, `tool_version`), not a Python import
- Configuration examples for different modalities and physiological traces
- Utilities for frame processing and trace filtering
- Shares similar config patterns (YAML structure) and package management (uv) with this repository
