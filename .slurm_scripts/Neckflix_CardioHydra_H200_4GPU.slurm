#!/bin/bash
#SBATCH --job-name=Neckflix_CardioHydra_H200_4GPU
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j_Neckflix_CardioHydra_H200_4GPU.out
#SBATCH --error=logs/%j_Neckflix_CardioHydra_H200_4GPU.err

# CardioHydra's LOSO sweep on Neckflix at 200 px: the shipped recipe
# (configs/hpc_configs_200, settled on the dev box 2026-10-14/15, development
# notes, experiment 14), 20 epochs a fold, on the 200 px cache.
#
#   sbatch .slurm_scripts/Neckflix_CardioHydra_H200_4GPU.slurm                     # every participant the cache admits
#   sbatch .slurm_scripts/Neckflix_CardioHydra_H200_4GPU.slurm 1 2 3 4 5 7 8 9 10 11 12 13 14 15 17 22 23 24 25 26 27 28 29 30 31
#   sbatch .slurm_scripts/Neckflix_CardioHydra_H200_4GPU.slurm 32 33 34 35 36 37 38 39 40 41 43 44 45 46 47 49 50 51 52 53 54 55 56 57 58
#
# The two half-lists are the 50 participants of the ported models' sweeps,
# for two nodes at once; both write into runs/${EXPERIMENT}/ (each fold is
# its own stamped directory). main.py does not resume a fold.
#
# Why 20 epochs. The cluster check of 2026-10-18 (Neckflix_CardioCheck,
# job 151998, 30 epochs on P033 and P036, scored every epoch) put the
# held-out ABP CCC at its peak by epoch 15-22 on both folds (0.71 and 0.77)
# and flat or sliding after, CVP flat from epoch 3, and the level error set
# by epoch 2, while the training loss kept falling: the second half of a
# 30-epoch schedule fitted the training participants and transferred
# nothing. HoldCosine is relative to the run's length, so the schedule is
# the same shape in 20 epochs with its tail where the peak was; it must be
# one run, never a continuation.
#
# Why the 200 px cache. The HPC cache was native 650 px and every window was
# shrunk to 200 in a loader worker, which on 2026-10-18 ran the check at
# 3.2 s a step with the A100s idle (the area-average resize of 2026-10-14
# as a dense einsum; development notes, experiment 9's postscript). The
# cache at /group/pgh004/carrow/caches/neckflix_zarr_200 (the Neckflix
# cacher with --resize 200 200: INTER_AREA for colour and infrared,
# nearest for depth) is what configs/datasets/neckflix_hpc.yaml now names,
# so the loader resizes nothing and a window read is a tenth of the bytes.
# Checkpoints from the native cache saw an area-averaged depth channel;
# these see nearest-neighbour depth, as the cacher writes it.
#
# Layout: four folds at once, one process and one A100 each at batch 4
# (--parallel 4 --nproc-per-node 1 --batch-size 4), the global batch of
# every hpc_configs_200 sweep, with no DistributedDataParallel. The check
# ran CardioHydra at 6.8 GB reserved a process at batch 2 in bf16 (coarse-
# grid masks, 2026-10-14), so batch 4 sits near 13 GB of the card's 39.5.
# 10 workers a fold (40 in all) and 400G of host memory. With the loader
# off the critical path the card sets the pace; the first epoch's step time
# in the fold's log.txt is the number to read (the check's was 3.2 s, the
# ported sweeps' 1.45 s, the card alone should be well under 1 s), and at
# 1 s a step a fold of 20 epochs with its held-out passes is about 1.2 h,
# 50 folds four at a time about 15 h; at 2 s a step, 30 h. Mamba3's Triton
# kernels compile on each fold's first step (about a minute).
#
# The checkout must carry the coherent-pooling commit of 2026-10-18 or
# later (OpenCV resize, neckflix_hpc.yaml on the 200 px cache; `uv sync`
# after pulling, the lock gained opencv-python-headless): `git pull`, then
# `git log -1` and `git status` before sbatch, and leave the checkout alone
# until the job finishes. A smoke in salloc on one GPU first, if the check
# job has not already run on this cache:
#
#   uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
#       --test-participant-id 33 \
#       --config configs/hpc_configs_200/cardiohydra_FS30_W10S1_RGBID_ABP-CVP_H200W200.yaml \
#       --experiment smoke_cardiohydra --epochs 1 --batch-size 4 --limit-windows 8
#
# Each fold writes its own log.txt under runs/${EXPERIMENT}/; score the
# sweep with scripts/evaluate.py runs/${EXPERIMENT} when it is done.

EXPERIMENT="neckflix_hpc_cardiohydra_H200W200_20ep"
PARTICIPANTS=("$@")

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

HELD_OUT=()
if [ "${#PARTICIPANTS[@]}" -gt 0 ]; then
    HELD_OUT=(--test-participant-id "${PARTICIPANTS[@]}")
fi

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}"
echo "CPUs on node: ${SLURM_CPUS_ON_NODE}, usable here: $(nproc)"
echo "commit: $(git log -1 --format='%h %s')"
echo "held out: ${PARTICIPANTS[*]:-every participant}"

uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
    "${HELD_OUT[@]}" \
    --config configs/hpc_configs_200/cardiohydra_FS30_W10S1_RGBID_ABP-CVP_H200W200.yaml \
    --experiment "${EXPERIMENT}" \
    --epochs 20 --batch-size 4 --num-workers 10 --parallel 4 --nproc-per-node 1
