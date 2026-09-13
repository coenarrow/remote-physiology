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
- Not preprocessing, which is CPU/IO heavy and will still get you killed —
  and not the first `uv run --project dataset/cachers/neckflix ...` either, which
  syncs a second 249 MB environment before it does any work

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
`.slurm_scripts/Neckflix_PhysMamba_4GPU.slurm` is the canonical template.

```bash
#!/bin/bash
#SBATCH --job-name=Neckflix_PhysMamba_2GPU
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                  # size to the job; NEVER --mem=0
#SBATCH --gres=gpu:v100:2
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j_Neckflix_PhysMamba_2GPU.out
#SBATCH --error=logs/%j_Neckflix_PhysMamba_2GPU.err

mkdir -p logs
cd "/group/pgh004/carrow/repo/remote-physiology"
module load cuda

uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc --test-participant-id 15 \
    --model physmamba --interface configs/interfaces/interface_neckflix.yaml \
    --training configs/training/physmamba_training.yaml --parallel 2 --nproc-per-node 1
```

The command is the dev-box command plus two numbers: `--parallel` folds at a time and
`--nproc-per-node` GPUs per fold (torchrun inside `main.py`; ports derived from the job id
automatically). Leave the participants off to hold out every one in turn.

Required in every script:

- **Log naming**: `logs/%j_<Model>_<Dataset>_<Options>.{out,err}`, plus `mkdir -p logs`
- **`module load cuda`** — GPU jobs fail without it
- **`uv run`** for all Python; never bare `python`
- **`--gres` equals `--parallel` times `--nproc-per-node`** — `main.py` refuses otherwise
- **`--cpus-per-task`** at least `--parallel` times `--nproc-per-node` times the recipe's `NUM_WORKERS`
- **`--mem`** set to the minimum the job needs (a 2-GPU dev run fits in 32G)
- **A `<name>_hpc.yaml` dataset config** — the committed dataset files point at the dev-box
  cache, so the cluster loads `BASE: <name>.yaml` plus its own `CACHED_PATH`, as
  `configs/datasets/pure_hpc.yaml` does; run directories then read `runs/<name>_hpc_<model>/`

## LOSO Sweeps

`main.py` is the sweep: one job holds out every participant in turn, `--parallel` folds at
a time, each in its own subprocess with its own `log.txt` under `runs/<dataset>_<model>/`.
Four small folds at once on a 4-GPU node is `--parallel 4 --nproc-per-node 1`; one
high-resolution fold across the node is `--parallel 1 --nproc-per-node 4`. No job array,
no fold list, no hand-derived port. `docs/hpc_pure_physmamba.md` walks the PURE sweep
through end to end, cache build included.

For hyperparameter sweeps, use job arrays over config variations, on `gpu` for quick
iteration.

## Interactive Sessions

```bash
salloc --job-name=Interactive_Session --partition=pophealth \
    --nodes=1 --mem=160000 --ntasks=16 --gres=gpu:a100:1 --time=5:00:00

module load cuda
cd /mmfs1/data/group/pgh004/carrow/repo/remote-physiology
uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc --test-participant-id 15 \
    --model physmamba --interface configs/interfaces/interface_neckflix.yaml \
    --training configs/training/physmamba_3ep.yaml --limit-windows 8
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
| Address already in use | A hand-run torchrun with a fixed port | `main.py` derives ports from `SLURM_JOB_ID` plus the fold's slot; do the same by hand |
| `needs N GPUs and M are visible` | `--gres` ≠ `--parallel` × `--nproc-per-node` | Make them match |
| `no admitted store` | Dataset config points at the dev-box cache | Use the `<name>_hpc.yaml` config |
| OOM in `logs/*.err` | Batch/resolution too large | Lower batch size, image size, or chunk length in the YAML |

Different GPU types need different config tuning — image size, chunk length, and batch size in the
YAML often need adjusting when moving between V100, A100, and H100.

**Debugging workflow**: reproduce in `salloc` on a single GPU → shrink the config (lower resolution,
shorter chunks) → read `logs/*.err` → scale back up via `sbatch` once it works.

## Paths

- Working directory: `/mmfs1/data/group/pgh004/carrow/repo/remote-physiology` (also reachable as
  `/group/pgh004/carrow/repo/remote-physiology`, which is what the SLURM scripts use)
- Group storage: `/group/pgh004/` — accessible from compute nodes
- The cache preprocessor is a **git submodule**, and the HPC checkout predates
  it, so `dataset/cachers/neckflix` is an empty directory there until someone runs
  `git submodule update --init` once in the working directory (then
  `git -C dataset/cachers/neckflix checkout main`). An empty submodule does not fail
  loudly: `uv run --project dataset/cachers/neckflix` silently falls through to this
  project instead, downloads torch, and dies in a compiler. Check
  `test -f dataset/cachers/neckflix/pyproject.toml` before trusting a cache-build job.

## Building a Cache (CPU, no GPU)

Preprocessing is a SLURM job like any other — CPU partition, no `module load
cuda`, no `--gres`. Budget **~12 GB RAM per worker** (`--mem`); the event
camera dominates that, so `--perspectives 1 2` is both cheaper and avoids the
ECF HDF5 codec that only the docker image builds.

```bash
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "/group/pgh004/carrow/repo/remote-physiology"
uv run --project dataset/cachers/neckflix neckflix-preprocess \
    --input-dir <raw Neckflix root> --output-dir <cache dir> \
    --resize 256 256 --perspectives 1 2 --num-workers 2
uv run python tools/validate_cache.py <cache dir>
```

Model it on `.slurm_scripts/PURE_Cache.slurm`, the ready-made cache job for PURE
(whose cacher lives in the tree at `dataset/cachers/pure`, no submodule), or on
`.slurm_scripts/Neckflix_Unsupervised.slurm`, the other CPU-only template. A
cacher resolves from its own `uv.lock` and its own Python 3.12, so the first run
in a fresh checkout syncs a second environment — do that inside the job, not on
the login node.

## Red Flags — Stop

- About to run `python train.py` / `uv run python main.py` straight from the shell
- "I'll just run this small thing on the login node"
- Writing a `.slurm` script from scratch instead of copying the template
- `--mem=0`, or a hardcoded `--master_port`
- `--parallel` times `--nproc-per-node` that doesn't match the requested GPU count

**All of these mean: stop, and go through SLURM properly.**
