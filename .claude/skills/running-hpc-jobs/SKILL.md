---
name: running-hpc-jobs
description: Use when running, training, preprocessing, testing, or benchmarking anything on the HPC cluster - submitting SLURM jobs, picking a partition or GPU type, writing .slurm scripts, requesting an interactive salloc session, launching multi-GPU distributed training, or diagnosing jobs stuck pending, OOM errors, port conflicts, and nproc_per_node mismatches.
---

# Running HPC Jobs

## The Iron Rule

**This repo is checked out on a login node. NEVER run computational tasks directly.**

Every training, inference, preprocessing, or benchmark run goes through SLURM — `sbatch` for
production, `salloc` for interactive debugging. No exceptions:

- Not "just a quick test"
- Not "only a few epochs"
- Not "just to see if it imports" (use `salloc`)
- Not preprocessing, which is CPU/IO heavy and will still get you killed

## GPU Resources

| Partition | Max GPUs | Resource flag | Use for |
|-----------|----------|---------------|---------|
| `gpu` | 2x V100 | `--gres=gpu:v100:N` | Development/testing (preferred) |
| `pophealth` | 4x A100 | `--gres=gpu:a100:N` | Dev fallback, production runs |
| `medical` | 4x H100 | `--gres=gpu:h100:N` | Dev last resort, primary for production |

### Choosing a partition for dev/test runs (<10 min expected)

Escalate only when the job is actually pending:

1. `gpu` with V100s. If pending →
2. `scancel`, resubmit to `pophealth` with A100s. If pending →
3. `scancel`, resubmit to `medical` with H100s. If pending →
4. Back to `gpu` and let it queue.

Test runs use **2 GPUs** — if it works on 2, assume it works on 3 and 4.

## Writing a SLURM Script

Scripts live in [.slurm_scripts/](../../../.slurm_scripts/) as `<Dataset>_<Model>_<Options>.slurm`.
Copy an existing script rather than writing from scratch —
`.slurm_scripts/UBFC-rPPG_DeepPhys_2GPU.slurm` is the canonical template.

```bash
#!/bin/bash
#SBATCH --job-name=DeepPhys_UBFC_2GPU
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                  # size to the job; NEVER --mem=0
#SBATCH --gres=gpu:v100:2
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j_DeepPhys_UBFC_2GPU.out
#SBATCH --error=logs/%j_DeepPhys_UBFC_2GPU.err

mkdir -p logs
cd "/group/pgh004/carrow/repo/rPPG-Toolbox"
module load cuda

uv run python -m torch.distributed.run --nproc_per_node=2 \
    main.py --config_file .configs/UBFC-rPPG_UBFC-rPPG_UBFC-rPPG_DEEPPHYS.yaml
```

Required in every script:

- **Log naming**: `logs/%j_<Model>_<Dataset>_<Options>.{out,err}`, plus `mkdir -p logs`
- **`module load cuda`** — GPU jobs fail without it
- **`uv run`** for all Python; never bare `python`
- **`--nproc_per_node` must equal the `--gres` GPU count**
- **`--mem`** set to the minimum the job needs (the 2-GPU UBFC-rPPG DeepPhys run uses 32G)

## Job Arrays: LOSO Cross-Validation

```bash
#SBATCH --partition=medical
#SBATCH --gres=gpu:h100:4
#SBATCH --array=1-57%1        # one job per participant, serialised

PORT=$((29500 + (SLURM_JOB_ID % 1000) + SLURM_ARRAY_TASK_ID))
PARTICIPANT=$(printf "P%03d" "${SLURM_ARRAY_TASK_ID}")

uv run python -m torch.distributed.run --nproc_per_node=4 --master_port="${PORT}" \
    neckflix_main.py --config_file <config> --test_participants "${PARTICIPANT}"
```

Always derive the port from the job ID. Fixed ports collide across concurrent jobs.

For hyperparameter sweeps, use job arrays over config variations (or Hydra/W&B), on `gpu` for
quick iteration.

## Interactive Sessions

```bash
salloc --job-name=Interactive_Session --partition=pophealth \
    --nodes=1 --mem=160000 --ntasks=16 --gres=gpu:a100:1 --time=5:00:00

module load cuda
cd /mmfs1/data/group/pgh004/carrow/repo/rPPG-Toolbox
uv run python neckflix_main.py --config_file physhydra_configs/physHydra_RGB_CVP.yaml
exit    # release the allocation when done
```

## Monitoring

```bash
squeue -u $USER              # your jobs
watch -n 1 squeue -u $USER   # auto-refresh
sinfo -p medical             # partition availability
scontrol show job <jobid>    # why a job is pending
scancel <jobid>              # cancel
tail -f logs/<jobid>_*.out   # follow output
nvidia-smi                   # GPU status (compute session only)
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Job stuck pending | Partition busy | Follow the escalation ladder above |
| `CUDA not available` | Missing module | `module load cuda` in the script/session |
| Address already in use | Fixed master port | Derive `PORT` from `SLURM_JOB_ID` |
| Hangs at distributed init | `--nproc_per_node` ≠ `--gres` count | Make them match |
| OOM in `logs/*.err` | Batch/resolution too large | Lower batch size, image size, or chunk length in the YAML |

Different GPU types need different config tuning — image size, chunk length, and batch size in the
YAML often need adjusting when moving between V100, A100, and H100.

**Debugging workflow**: reproduce in `salloc` on a single GPU → shrink the config (lower resolution,
shorter chunks) → read `logs/*.err` → scale back up via `sbatch` once it works.

## Paths

- Working directory: `/mmfs1/data/group/pgh004/carrow/repo/rPPG-Toolbox` (also reachable as
  `/group/pgh004/carrow/repo/rPPG-Toolbox`, which is what the SLURM scripts use)
- Group storage: `/group/pgh004/` — accessible from compute nodes

## Red Flags — Stop

- About to run `python train.py` / `uv run python main.py` straight from the shell
- "I'll just run this small thing on the login node"
- Writing a `.slurm` script from scratch instead of copying the template
- `--mem=0`, or a hardcoded `--master_port`
- `--nproc_per_node` that doesn't match the requested GPU count

**All of these mean: stop, and go through SLURM properly.**
