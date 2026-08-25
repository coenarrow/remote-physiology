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
|       |-- NeckflixLoader.py  Neckflix multimodal HDF5 loader
|       |-- ...                Other dataset loaders (SCAMPS, BP4D+, MMPD, etc.)
|       |-- face_detector/     Face detection backends (Haar Cascade, YOLO5Face)
|
|-- neural_methods/
|   |-- model/                 Neural network architectures (one file per model)
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
|       |-- RythmFormerLossComputer.py
|
|-- evaluation/
|   |-- metrics.py             Metric computation (MAE, RMSE, MAPE, Pearson, SNR, BA)
|   |-- post_process.py        Signal post-processing (FFT, peak detection)
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

The Neckflix-specific entry point with custom handling for HDF5 multimodal data, pressure waveform-specific logic, and distributed training support via `torch.distributed.run`.

### Usage

```bash
# Single GPU
uv run python main.py --config_file path/to/config.yaml

# Multi-GPU distributed training
uv run python -m torch.distributed.run --nproc_per_node=N \
    main.py --config_file path/to/config.yaml

# Neckflix with participant selection (LOSO cross-validation)
uv run python -m torch.distributed.run --nproc_per_node=4 \
    neckflix_main.py --config_file path/to/config.yaml \
    --test_participants P001
```


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

1. Define the architecture in `neural_methods/model/<ModelName>.py` as a `torch.nn.Module`
2. Create `neural_methods/trainer/<ModelName>Trainer.py` inheriting from `BaseTrainer`
3. Register the model in `main.py`'s `train_and_test()` and `test()` functions
4. Create a YAML config with model-specific hyperparameters

### Adding New Metrics

1. Add the metric function to `evaluation/metrics.py`
2. Register it in the metric dispatcher within the evaluation code
3. Add the metric name to the `METRICS` list in experiment YAML configs


## Neckflix Dataset

The Neckflix dataset is a multimodal collection for cardiovascular pressure estimation, stored in HDF5 format with `hdf5plugin` compression.

**Location:** `/group/pgh004/carrow/repo/Neckflix/dataset`

**Camera modalities:**
- RGB -- Standard color video
- IR -- Infrared video
- Depth -- Depth map video
- IR_RAW, DEPTH_RAW -- Unprocessed sensor data
- EV -- Event camera stream

**Physiological traces:**
- ABP -- Arterial Blood Pressure
- CVP -- Central Venous Pressure
- ECG -- Electrocardiogram

All modalities are temporally synchronized. The data is accessed via `NeckflixLoader.py`, which reads HDF5 files using:

```python
import h5py
import hdf5plugin

with h5py.File('data.h5', 'r') as f:
    video = f['RGB'][:]
    abp = f['ABP'][:]
    cvp = f['CVP'][:]
```

Preprocessing scripts for raw Kinect Azure data live in the related Neckflix repository at `/mmfs1/data/group/pgh004/carrow/repo/Neckflix`.


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
| Neckflix | `/group/pgh004/carrow/repo/Neckflix/dataset` | HDF5 multimodal (RGB/IR/Depth + ABP/CVP/ECG) |
| PURE | `/group/pgh004/carrow/zipped_datasets/PURE` | Standard rPPG dataset |
| UBFC-rPPG | `/group/pgh004/carrow/zipped_datasets/UBFC-rPPG` | Standard rPPG dataset |

Preprocessed caches are stored at paths defined by `CACHED_PATH` in the YAML config, typically under `PreprocessedData/`.


## Python Environment

The project uses **uv** as its Python package manager. The virtual environment (`.venv/`) is auto-managed by uv.

**Key dependencies:**
- PyTorch with CUDA support
- `hdf5plugin` for Neckflix HDF5 data access
- Standard scientific Python stack (numpy, scipy, matplotlib)

All scripts are invoked via `uv run python` to ensure the correct environment is used. Full dependency specifications are in `pyproject.toml` and `requirements.txt`, with the lockfile at `uv.lock`.


## Related Repositories

**Neckflix** (`/mmfs1/data/group/pgh004/carrow/repo/Neckflix`):
- Raw data preprocessing pipeline for Kinect Azure captures
- HDF5 data generation scripts (`preprocess.py`, `simple_preprocess.py`)
- Configuration examples for different modalities and physiological traces
- Utilities for frame processing and trace filtering
- Shares similar config patterns (YAML structure) and package management (uv) with this repository
