#!/bin/bash
#SBATCH --job-name=Neckflix_CardioTuning_P033_4GPU
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j_Neckflix_CardioTuning_P033_4GPU.out
#SBATCH --error=logs/%j_Neckflix_CardioTuning_P033_4GPU.err

# The CardioConv / CardioHydra tuning set on one fold, P033 held out, at the
# 200 px interface of the hpc_configs_200 sweeps: every YAML in
# configs/hpc_tuning_p33/, one after another, 10 epochs each, the whole node
# per arm (--parallel 1 --nproc-per-node 4). Each arm's fold lands in
# runs/neckflix_hpc_tuning_p33_<arm>/, so the arms are read side by side
# with tools/tuning_report.py and the fold directories of the sweeps.
#
#   sbatch .slurm_scripts/Neckflix_CardioTuning_P033_4GPU.slurm            # every arm, one job
#   sbatch .slurm_scripts/Neckflix_CardioTuning_P033_4GPU.slurm cardiohydra_baseline cardioconv_dense
#
# With arguments, only the named arms (YAML stems) run; without, every arm
# in ARM_ORDER below. Every arm was settled on the dev card on 2026-10-14/15
# (notes, experiment 14; the hpc_configs_200 cardio YAMLs carry the
# verdicts), on the 128 px cache upsampled to 200, float32, batch 1: this
# job is the optional check of those verdicts on the native 200 px cache,
# bf16, global batch 4, and is not needed before the LOSO sweep. The two
# shipped recipes come first, then the arms that lost, in the order of how
# close they came. In one job, an arm that fails is reported and the next
# one starts.
#
# Batch 1 a card, four cards: a global batch of 4, the batch of every
# hpc_configs_200 sweep (two cards at batch 2), so a rate tuned here carries
# straight over to the LOSO run. Peak reserved memory a card at the recipe
# (rate 4 / wavelet 3, coarse-grid masks, bf16, one step): under 11 GB at
# batch 1 for CardioConv, so batch 2 a card would also fit the A100's 39.5 GB.
# The sweeps ran about 1.5 s a step, loader-bound, and 5.5-7 min an epoch at
# this interface, so a 10-epoch arm is about 1-2 h. 5 workers a process (20
# in all), as the sweeps.
#
# Neither model has run under DistributedDataParallel before. What DDP needs,
# a gradient for every parameter on every step, was checked on the dev box
# (2026-10-14, runs/dev_cardio/exp12_coarse_check/grad_coverage.py): with the
# rate gate open and with it closed (the closest-sample path), every
# parameter of both models has a gradient, zero where the gate is closed.
# Mamba3's Triton kernels (CardioHydra) have not run on the cluster yet,
# though the same mamba-ssm 2.3.2 install ran PhysMamba's sweep. A smoke in
# salloc on two A100s before any cardio job, one arm of each model, costs
# minutes and is the one check the dev box cannot do (bf16, DDP, Triton):
#
#   uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc #       --test-participant-id 33 --config configs/hpc_tuning_p33/cardiohydra_baseline.yaml #       --experiment smoke_tuning_p33 --epochs 1 --batch-size 1 --limit-windows 8 #       --parallel 1 --nproc-per-node 2

ARMS=("$@")
CONFIG_DIR="configs/hpc_tuning_p33"
ARM_ORDER=(cardiohydra_baseline cardioconv_dense cardioconv_baseline
           cardiohydra_tv0.02 cardiohydra_lr5e-4 cardiohydra_mamba0.3
           cardioconv_mixer31x32 cardioconv_lr5e-4 cardiohydra_abpamp)

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

if [ "${#ARMS[@]}" -eq 0 ]; then
    ARMS=("${ARM_ORDER[@]}")
fi

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}"
echo "CPUs on node: ${SLURM_CPUS_ON_NODE}, usable here: $(nproc)"
echo "commit: $(git log -1 --format='%h %s')"
echo "arms: ${ARMS[*]}"

FAILED=()
for arm in "${ARMS[@]}"; do
    config="${CONFIG_DIR}/${arm}.yaml"
    if [ ! -f "${config}" ]; then
        echo "no such arm: ${config}"
        FAILED+=("${arm}")
        continue
    fi
    echo "=== arm ${arm}: $(date) ==="
    uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
        --test-participant-id 33 \
        --config "${config}" \
        --experiment "neckflix_hpc_tuning_p33_${arm}" \
        --epochs 10 --batch-size 1 --num-workers 5 --parallel 1 --nproc-per-node 4
    status=$?
    echo "=== arm ${arm} exited ${status}: $(date) ==="
    if [ "${status}" -ne 0 ]; then
        FAILED+=("${arm}")
    fi
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "failed arms: ${FAILED[*]}"
    exit 1
fi
echo "all ${#ARMS[@]} arm(s) done: $(date)"
