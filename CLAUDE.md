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

### Adding a New Dataset (Neckflix Integration)

**Pattern**: Follow `dataset/data_loader/BaseLoader.py` interface

1. **Create loader**: `dataset/data_loader/NeckflixLoader.py`
2. **Implement required methods**:
   - `preprocess_dataset(config_preprocess)`: Convert raw data to preprocessed format
   - `read_video(video_file)`: Load video frames (multimodal: RGB/IR/Depth)
   - `read_wave(bvp_file)`: Load physiological traces (ABP/CVP/ECG from HDF5)
3. **Optional overrides**: `__len__`, `__getitem__`, `save`, `load` (generally not recommended)
4. **Update config.py**: Add dataset parameters and paths
5. **Create YAML configs**: Define preprocessing and training parameters

**Neckflix-specific considerations**:
- HDF5 storage format with `hdf5plugin` for compression
- Multiple camera modalities (RGB, IR, Depth) synchronized with traces
- Pressure waveforms (ABP, CVP) require different preprocessing than PPG
- See `/mmfs1/data/group/pgh004/carrow/repo/Neckflix` for preprocessing examples

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

**For Neckflix/Pressure estimation**:
- See `physhydra_configs/` for examples
- May need custom parameters for multimodal inputs
- Pressure waveform-specific preprocessing settings

### Python Entry Points

**`main.py`**: Original toolbox entry
- Supports all standard datasets (UBFC-rPPG, PURE, SCAMPS, etc.)
- Both supervised and unsupervised methods

**`neckflix_main.py`**: Neckflix-specific entry
- Custom handling for HDF5 multimodal data
- Pressure waveform-specific logic
- Distributed training support via `torch.distributed.run`

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

### Working with HDF5 Data (Neckflix)

**Dependencies**: `hdf5plugin` for compressed HDF5 support

**Pattern** (see `/mmfs1/data/group/pgh004/carrow/repo/Neckflix` for examples):
```python
import h5py
import hdf5plugin

with h5py.File('data.h5', 'r') as f:
    video = f['RGB'][:]      # Load RGB frames
    ir = f['IR'][:]          # Load IR frames
    depth = f['DEPTH'][:]    # Load depth frames
    abp = f['ABP'][:]        # Load ABP trace
    cvp = f['CVP'][:]        # Load CVP trace
```

**Preprocessing considerations**:
- Multiple modalities may have different frame rates
- Synchronization between video and physiological traces critical
- Chunk-based processing for memory efficiency

## Dataset Locations

| Dataset | Path | Notes |
|---------|------|-------|
| Neckflix | `/group/pgh004/carrow/repo/Neckflix/dataset` | HDF5 multimodal (RGB/IR/Depth + ABP/CVP/ECG) |
| PURE | `/group/pgh004/carrow/zipped_datasets/PURE` | Standard rPPG dataset |
| UBFC-rPPG | `/group/pgh004/carrow/zipped_datasets/UBFC-rPPG` | Standard rPPG dataset |

Additional datasets will be added to `/group/pgh004/carrow/zipped_datasets/` over time.

## Project-Specific Notes

### Neckflix Dataset Characteristics

**Location**: `/group/pgh004/carrow/repo/Neckflix/dataset`

**Data format**: HDF5 with compression via `hdf5plugin`

**Modalities**:
- **Video**: RGB, IR (Infrared), Depth, IR_RAW, DEPTH_RAW, EV (Event camera)
- **Physiological traces**: ABP (Arterial Blood Pressure), CVP (Central Venous Pressure), ECG

**Key differences from standard rPPG datasets**:
- Multimodal camera inputs (not just RGB)
- Pressure waveforms instead of/in addition to PPG
- HDF5 storage format vs individual video files
- Preprocessing scripts in Neckflix repo: `preprocess.py`, `simple_preprocess.py`

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
- Install dependencies: Handled by `setup.sh uv`
- Run scripts: `uv run python <script.py>`
- Virtual env: `.venv/` (auto-managed by uv)

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
