#!/bin/bash
#SBATCH --job-name=Neckflix_CardioConv_H200_4GPU
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j_Neckflix_CardioConv_H200_4GPU.out
#SBATCH --error=logs/%j_Neckflix_CardioConv_H200_4GPU.err

# CardioConv's LOSO sweep on Neckflix at 200 px: the shipped recipe
# (configs/hpc_configs_200: TRACE_INDEPENDENT false from experiment 14,
# COHERENT_POOLING true from experiment 16), 20 epochs a fold, on the 200 px
# cache. The same layout, cache and epoch count as CardioHydra's sweep
# (Neckflix_CardioHydra_H200_4GPU.slurm, whose header has the reasons), so
# the two compare like for like: same rate estimator, wavelet bank, masks
# and kinematics, a fixed coherent readout here against learned trunks
# there.
#
#   sbatch .slurm_scripts/Neckflix_CardioConv_H200_4GPU.slurm                      # every participant the cache admits
#   sbatch .slurm_scripts/Neckflix_CardioConv_H200_4GPU.slurm 1 2 3 4 5 7 8 9 10 11 12 13 14 15 17 22 23 24 25 26 27 28 29 30 31
#   sbatch .slurm_scripts/Neckflix_CardioConv_H200_4GPU.slurm 32 33 34 35 36 37 38 39 40 41 43 44 45 46 47 49 50 51 52 53 54 55 56 57 58
#
# Why coherent pooling. With the masked mean of the intensity as its
# candidate, CardioConv's training ABP CCC term sat at 0.62-0.65 for 25
# epochs on the cluster check of 2026-10-18 (job 151998) and the folds
# finished at ABP CCC 0.22 (P033) and 0.21 (P036), r 0.43 and 0.36: the
# readout did not fit the shape even on the training participants, on the
# same masks CardioHydra reads to 0.65 and 0.74. Pooling the wavelet
# harmonics with every pixel rotated into phase instead (development notes,
# experiment 16) took the training term to 0.27 on the dev box and the P033
# fold to ABP CCC 0.59-0.64, r 0.75-0.78, CVP 0.31-0.35, against 0.38 /
# 0.59 / 0.23 for the masked mean; one fold, so this sweep is the
# confirmation. Its held-out peak came at epochs 2-4 on the dev box's
# 10-epoch schedule, so 20 epochs here is generous, kept equal to
# CardioHydra's for the comparison.
#
# CardioConv peaked at 6.2 GB reserved a process at batch 2 in bf16 on the
# check, so batch 4 sits near 12 GB a card. Its step was the loader's on
# the check, as CardioHydra's; with the 200 px cache the card sets the
# pace and a fold of 20 epochs should take about an hour, 50 folds four at
# a time 13-25 h depending on the step time the first fold's log.txt shows.
#
# The checkout must carry the coherent-pooling commit of 2026-10-18 or
# later (COHERENT_POOLING on CardioConvConfig and in every CardioConv YAML,
# OpenCV resize, neckflix_hpc.yaml on the 200 px cache; `uv sync` after
# pulling): `git pull`, then `git log -1` and `git status` before sbatch,
# and leave the checkout alone until the job finishes. A smoke in salloc on
# one GPU first:
#
#   uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
#       --test-participant-id 33 \
#       --config configs/hpc_configs_200/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H200W200.yaml \
#       --experiment smoke_cardioconv --epochs 1 --batch-size 4 --limit-windows 8
#
# Each fold writes its own log.txt under runs/${EXPERIMENT}/; score the
# sweep with scripts/evaluate.py runs/${EXPERIMENT} when it is done.

EXPERIMENT="neckflix_hpc_cardioconv_H200W200_20ep"
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
    --config configs/hpc_configs_200/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H200W200.yaml \
    --experiment "${EXPERIMENT}" \
    --epochs 20 --batch-size 4 --num-workers 10 --parallel 4 --nproc-per-node 1
