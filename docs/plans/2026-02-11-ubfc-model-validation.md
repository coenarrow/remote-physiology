# UBFC-rPPG DeepPhys Validation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify the repository is correctly set up by running a full DeepPhys train/test cycle on UBFC-rPPG — first on 1 GPU, then on 2 GPUs.

**Architecture:** Update the DeepPhys YAML config in `.configs/` with correct dataset paths and splits (train 0-0.8, valid 0.8-0.9, test 0.9-1.0 all on UBFC-rPPG). Write two SLURM scripts: one for single-GPU and one for multi-GPU. Verify preprocessing, training, and metrics generation work end-to-end.

**Tech Stack:** SLURM, PyTorch (via `uv run`), `torchrun` for DDP, V100 GPUs on `gpu` partition

---

## Configuration Values

```yaml
# Paths (same for TRAIN, VALID, and TEST sections)
DATA_PATH: "/group/pgh004/carrow/zipped_datasets/UBFC-rPPG"
CACHED_PATH: "/group/pgh004/carrow/zipped_datasets/PreprocessedData"
EXP_DATA_NAME: ""
DATASET: UBFC-rPPG

# Splits
TRAIN:  BEGIN: 0.0, END: 0.8
VALID:  BEGIN: 0.8, END: 0.9
TEST:   BEGIN: 0.9, END: 1.0

# Epochs (minimal validation run)
EPOCHS: 3
```

---

### Task 1: Update DeepPhys Config

**Files:**
- Modify: `.configs/UBFC-rPPG_UBFC-rPPG_UBFC-rPPG_DEEPPHYS.yaml`

**Step 1: Update TRAIN section paths, splits, and epochs**

Change these fields in the `TRAIN` and `TRAIN.DATA` blocks:

```yaml
TRAIN:
  BATCH_SIZE: 4
  EPOCHS: 3
  MODEL_FILE_NAME: UBFC_UBFC_UBFC_deepphys
  DATA:
    DATASET: UBFC-rPPG
    DO_PREPROCESS: True    # First run: generates cached .npy files
    DATA_PATH: "/group/pgh004/carrow/zipped_datasets/UBFC-rPPG"
    CACHED_PATH: "/group/pgh004/carrow/zipped_datasets/PreprocessedData"
    EXP_DATA_NAME: ""
    BEGIN: 0.0
    END: 0.8
```

**Step 2: Update VALID section paths and splits**

```yaml
VALID:
  DATA:
    DATASET: UBFC-rPPG
    DO_PREPROCESS: True
    DATA_PATH: "/group/pgh004/carrow/zipped_datasets/UBFC-rPPG"
    CACHED_PATH: "/group/pgh004/carrow/zipped_datasets/PreprocessedData"
    EXP_DATA_NAME: ""
    BEGIN: 0.8
    END: 0.9
```

**Step 3: Update TEST section**

Change TEST to use UBFC-rPPG (currently set to PURE):

```yaml
TEST:
  USE_LAST_EPOCH: False    # Use validation to find best epoch
  DATA:
    DATASET: UBFC-rPPG
    DO_PREPROCESS: True
    DATA_PATH: "/group/pgh004/carrow/zipped_datasets/UBFC-rPPG"
    CACHED_PATH: "/group/pgh004/carrow/zipped_datasets/PreprocessedData"
    EXP_DATA_NAME: ""
    BEGIN: 0.9
    END: 1.0
```

**Step 4: Keep NUM_OF_GPU_TRAIN as 1**

The 1-GPU script runs first, so keep:

```yaml
NUM_OF_GPU_TRAIN: 1
```

No other fields change (keep existing DATA_FORMAT: NDCHW, LR: 9e-3, CHUNK_LENGTH: 180, RESIZE: 72x72, MODEL.NAME: DeepPhys, DROP_RATE: 0.2, PLOT_LOSSES_AND_LR: True).

---

### Task 2: Create Single-GPU SLURM Script

**Files:**
- Create: `.slurm_scripts/UBFC-rPPG_DeepPhys_1GPU.slurm`

**Step 1: Write the script**

```bash
#!/bin/bash
#SBATCH --job-name=DeepPhys_UBFC_1GPU
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/DeepPhys_UBFC_1GPU_%j.out
#SBATCH --error=logs/DeepPhys_UBFC_1GPU_%j.err

mkdir -p logs

cd "/group/pgh004/carrow/repo/rPPG-Toolbox"

module load cuda

echo "Starting DeepPhys training on UBFC-rPPG (1 GPU)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"

uv run python -m torch.distributed.run --nproc_per_node=1 \
    main.py --config_file .configs/UBFC-rPPG_UBFC-rPPG_UBFC-rPPG_DEEPPHYS.yaml
```

---

### Task 3: Submit Single-GPU DeepPhys Job

This is the first job — it will preprocess UBFC-rPPG and run 3 epochs of training.

**Step 1: Submit the job**

```bash
cd /group/pgh004/carrow/repo/rPPG-Toolbox
sbatch .slurm_scripts/UBFC-rPPG_DeepPhys_1GPU.slurm
```

**Step 2: Monitor the job**

```bash
squeue -u $USER
# Once running, tail the log:
tail -f logs/DeepPhys_UBFC_1GPU_<jobid>.out
```

**Step 3: Verify success**

Check the log for:
1. Preprocessing completed messages (cached `.npy` files created in `CACHED_PATH`)
2. Training loss printed for 3 epochs
3. Test metrics printed: MAE, RMSE, MAPE, Pearson, SNR, BA
4. Loss plot saved (PLOT_LOSSES_AND_LR: True)
5. No errors in the `.err` file

```bash
# Check for errors
cat logs/DeepPhys_UBFC_1GPU_<jobid>.err
# Check output directory
ls runs/exp/
```

---

### Task 4: Create Multi-GPU SLURM Script

**Files:**
- Create: `.slurm_scripts/UBFC-rPPG_DeepPhys_2GPU.slurm`

**Step 1: Write the script**

```bash
#!/bin/bash
#SBATCH --job-name=DeepPhys_UBFC_2GPU
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:2
#SBATCH --time=2:00:00
#SBATCH --output=logs/DeepPhys_UBFC_2GPU_%j.out
#SBATCH --error=logs/DeepPhys_UBFC_2GPU_%j.err

mkdir -p logs

cd "/group/pgh004/carrow/repo/rPPG-Toolbox"

module load cuda

echo "Starting DeepPhys training on UBFC-rPPG (2 GPUs)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"

uv run python -m torch.distributed.run --nproc_per_node=2 \
    main.py --config_file .configs/UBFC-rPPG_UBFC-rPPG_UBFC-rPPG_DEEPPHYS.yaml
```

**Step 2: Update config for 2 GPUs**

Before submitting, update `.configs/UBFC-rPPG_UBFC-rPPG_UBFC-rPPG_DEEPPHYS.yaml`:

```yaml
NUM_OF_GPU_TRAIN: 2
```

Also set `DO_PREPROCESS: False` in TRAIN, VALID, and TEST sections (preprocessing already done by Task 3).

---

### Task 5: Submit Multi-GPU DeepPhys Job

**Step 1: Submit the job**

```bash
cd /group/pgh004/carrow/repo/rPPG-Toolbox
sbatch .slurm_scripts/UBFC-rPPG_DeepPhys_2GPU.slurm
```

**Step 2: Monitor the job**

```bash
squeue -u $USER
tail -f logs/DeepPhys_UBFC_2GPU_<jobid>.out
```

**Step 3: Verify success**

Same checks as Task 3:
1. Training loss printed for 3 epochs
2. Test metrics printed: MAE, RMSE, MAPE, Pearson, SNR, BA
3. Loss plot saved
4. No errors in the `.err` file

**Step 4: Compare 1-GPU vs 2-GPU results**

Metrics should be similar (not identical due to DDP nondeterminism). Key check is that both complete without errors.

---

### Task 6: Verify and Commit

**Step 1: Confirm both runs produced outputs**

```bash
ls runs/exp/
```

Expected: model checkpoints, loss plots, metric outputs from both runs.

**Step 2: Commit config and SLURM changes**

```bash
cd /group/pgh004/carrow/repo/rPPG-Toolbox
git add .configs/UBFC-rPPG_UBFC-rPPG_UBFC-rPPG_DEEPPHYS.yaml \
        .slurm_scripts/UBFC-rPPG_DeepPhys_1GPU.slurm \
        .slurm_scripts/UBFC-rPPG_DeepPhys_2GPU.slurm
git commit -m "feat: add UBFC-rPPG DeepPhys validation configs and SLURM scripts (1GPU + 2GPU)"
```

---

## Troubleshooting

**Job fails immediately**: Check `.err` file. Common causes:
- Module not loaded (`module load cuda`)
- Wrong path (typo in DATA_PATH)
- GPU not available (try `sinfo -p gpu`)

**OOM on V100**: Reduce `BATCH_SIZE` to 2, or reduce `RESIZE` dimensions.

**Preprocessing hangs**: UBFC-rPPG face detection (Haar Cascade) can be slow. Allow up to 30 min for full preprocessing.

**Metrics look unreasonable**: With only 3 epochs, metrics won't be competitive. The goal is to verify the pipeline works end-to-end, not achieve good performance.

**2-GPU fails but 1-GPU works**: Check that `NUM_OF_GPU_TRAIN: 2` is set in the config, and `--nproc_per_node=2` matches the `--gres=gpu:v100:2`.

---

## Next Steps

After DeepPhys validates successfully on both 1-GPU and 2-GPU:
- Repeat for remaining models: EfficientPhys, PhysFormer, PhysMamba, RhythmFormer, TS_CAN, PhysNet
- Each model needs its own config update (paths, splits, epochs) and SLURM script

---

Last updated: 2026-02-11
