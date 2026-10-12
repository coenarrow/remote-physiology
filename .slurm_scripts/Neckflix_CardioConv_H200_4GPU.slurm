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

# CardioConv on Neckflix at 200 px, 30 epochs a fold, the whole node: as long
# as the ported models' 20-epoch sweeps plus their 10-epoch continuations, in
# one run, so it needs no continuation. Its recipe's learning rate is
# constant, so 30 straight epochs follow the schedule 20 + 10 would; the only
# difference from a continued run is that Adam's moments never restart.
#
#   sbatch .slurm_scripts/Neckflix_CardioConv_H200_4GPU.slurm 1 2 3 4 5 7 8 9 10 11 12 13 14 15 17 22 23 24 25 26 27 28 29 30 31
#   sbatch .slurm_scripts/Neckflix_CardioConv_H200_4GPU.slurm 32 33 34 35 36 37 38 39 40 41 43 44 45 46 47 49 50 51 52 53 54 55 56 57 58
#
# The 50 folds (the participants of the 20-epoch sweeps) go as two jobs of 25,
# which may run at once on two nodes and write into the same
# runs/${EXPERIMENT}/ (each fold is its own stamped directory). main.py does
# not resume, and all 50 in one job would outrun the time limit; with no
# arguments it holds out every participant the dataset config admits.
#
# Two folds at a time, two A100s each at batch 2 a card, a global batch of 4
# like every ported model's (--parallel 2 --nproc-per-node 2). Batch 4 does
# not fit one card: probed at 200 px in bfloat16 on the dev box (2026-10-11),
# the bounded-mask CardioConv peaked at 7.7 GB allocated at batch 1 and
# 13.9 GB at batch 2, about 6 GB a sample, so about 26 GB at batch 4 before the
# allocator's slack. Every parameter takes a gradient with the training gate
# open or closed, so DistributedDataParallel runs it as it runs RhythmFormer.
# Batch 2 a card stays under about 25 GB of the A100's 39.5. The finished
# sweeps were bound by the loader, not the cards (about 1.5 s a step whatever
# the model, and about 1.5 min an epoch for the test pass and records), and
# the probe puts this model at 2.5 times FactorizePhys a sample, which keeps
# it near that floor: an epoch of 5.5-7 min, a median fold 2.7-3.4 h for 30
# epochs (the participant with the most test windows about 5.5 h), a job of
# 25 two at a time 38-52 h. 5 workers a process (20 in all) and 400G of host
# memory, as RhythmFormer's sweep.
#
# The model lives on the cardiohydra branch, which is not on origin: commit
# it (this script, configs/hpc_configs_200/cardio*_H200W200.yaml and the
# model code) and push it, then on the cluster `git fetch && git checkout
# cardiohydra` and check `git log -1` before sbatch. A pull on main is not
# enough: it keeps main's older script of this name, which trains the
# pre-redesign CardioConv. Every fold imports the code and reads its config
# when it starts, so leave the checkout alone until both jobs finish; jobs
# started from the same checkout meanwhile (the ported models'
# continuations) run this branch's code too, so run these from a second
# checkout to keep them apart. The model has not run on the cluster yet; a
# smoke in salloc on two GPUs first:
#
#   uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
#       --test-participant-id 15 \
#       --config configs/hpc_configs_200/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H200W200.yaml \
#       --epochs 1 --batch-size 2 --limit-windows 8 --parallel 1 --nproc-per-node 2
#
# Each fold writes its own log.txt under runs/${EXPERIMENT}/.

EXPERIMENT="neckflix_hpc_cardioconv_H200W200_30ep"
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
echo "held out: ${PARTICIPANTS[*]:-every participant}"

uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
    "${HELD_OUT[@]}" \
    --config configs/hpc_configs_200/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H200W200.yaml \
    --experiment "${EXPERIMENT}" \
    --epochs 30 --batch-size 2 --num-workers 5 --parallel 2 --nproc-per-node 2
