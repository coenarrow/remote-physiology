# Running the PURE PhysMamba test run on the HPC

The HPC twin of the dev-box run: PhysMamba on PURE, three epochs, every one
of the ten participants held out in turn. Same scripts, same configs, same
recipe. What differs is that the cache is built on the cluster from the raw
dataset, and that two folds run at a time on two GPUs.

What it leaves, under `runs/pure_hpc_physmamba/` in the HPC checkout
(`<test dataset>_<model>`, or `runs/<NAME>/` with `--experiment NAME`), one
directory per fold:

```text
runs/pure_hpc_physmamba/PHYSMAMBA_PURE_HPC.01_<YYYYMMDDHHMM>/
  config.yaml       everything the fold ran on, resolved cache path included
  log.txt           the fold's whole output (main.py writes it)
  losses.csv        one row per epoch, 1-based
  model.pt          the latest epoch's weights
  epoch_01/         model.pt and the scored test_records/ of epoch 1
  epoch_02/
  epoch_03/
... nine more, .02 to .10
```

About 24 MB per fold, so roughly a quarter of a gigabyte for the sweep.

Everything below happens in this order. Steps 1 and 2 are on the dev box,
the rest on the cluster.

## 1. The launcher

`main.py` is the fold pool: `--parallel K` folds at a time,
`--nproc-per-node N` processes per fold, `--experiment NAME` for the
directory under `runs/`. The SLURM script in step 7 passes the first two;
the README's Experiments section describes all three. Nothing to swap or
build before the run.

## 2. Commit and push (dev box)

The cluster gets code through git, and the tree currently holds the whole
day's work uncommitted: the script merge, `main.py`, the per-epoch scoring,
the deleted `tests/`, `.gitattributes`, `configs/datasets/pure_hpc.yaml`,
`.slurm_scripts/PURE_Cache.slurm`, `.slurm_scripts/PURE_PhysMamba_3ep.slurm`
and this document.

```bash
git status --short          # check nothing unexpected is in there
git add -A
git commit
git push origin main
```

`.gitattributes` pins `*.sh` and `*.slurm` to LF, so the scripts arrive
runnable on Linux regardless of the Windows `core.autocrlf` setting. If git
reports either SLURM script as modified right after the commit, run
`git add --renormalize .` and commit once more.

## 3. Pull on the login node

```bash
cd /group/pgh004/carrow/repo/remote-physiology
git pull
git log --oneline -1        # the commit from step 2
```

The Neckflix preprocessor submodule is not needed for PURE. The PURE
cacher lives in the tree at `dataset/cachers/pure`, so it is already there.

## 4. Build the environment (inside an allocation, never on the login node)

`uv sync` compiles `mamba-ssm` and `causal-conv1d` against the project's
torch, which needs the CUDA toolkit and a fair amount of CPU. Do it in an
interactive session:

```bash
salloc --job-name=Interactive_Session --partition=gpu --nodes=1 --ntasks=8 --mem=32G --gres=gpu:v100:1 --time=1:00:00
module load cuda
cd /group/pgh004/carrow/repo/remote-physiology
uv sync --no-install-package mamba-ssm --no-install-package causal-conv1d
uv sync
uv run python -c "import torch, mamba_ssm; print(torch.__version__, torch.cuda.is_available(), mamba_ssm.__version__)"
exit
```

The two-step sync is deliberate: torch has to be installed before the two
CUDA extensions are built. The first sync takes a while; uv caches the
builds, so later syncs are fast. If the home directory quota is tight,
put the cache on group storage first with
`export UV_CACHE_DIR=/group/pgh004/carrow/.uv_cache`.

The check line should print the torch version, `True`, and `2.3.1`. If
`mamba_ssm` fails to import, PhysMamba still runs on the pure-PyTorch
fallback, but slowly; fix the build rather than accept that.

## 5. Build the PURE cache on the cluster

The run reads a zarr cache: one store per recording, fifty-nine stores
named `SS-TT.zarr`, each carrying its participant in its root attrs. It is
built on the cluster, from the raw PURE dataset, by the CPU job
`.slurm_scripts/PURE_Cache.slurm`. The raw dataset is the directory
holding the `SS-TT` recording directories, each with its PNG frames and
its JSON oximeter sidecar; it has to be on storage the compute nodes can
read.

Two paths sit at the top of that script. Set them before submitting:

| Variable | Meaning | As written |
| --- | --- | --- |
| `PURE_RAW` | the raw PURE root | `/group/pgh004/carrow/data/PURE`, a guess: edit it |
| `PURE_CACHE` | where the stores go | `/group/pgh004/carrow/caches/pure_zarr`, the path `pure_hpc.yaml` reads; change both together |

```bash
sbatch .slurm_scripts/PURE_Cache.slurm
```

The job refuses to start if `PURE_RAW` does not exist, then runs the
cacher with four workers and `--resize 256 256`, counts the stores, and
runs the validator over the cache. The cacher has its own environment
under `dataset/cachers/pure`, synced on the job's first run, which is why
the sync belongs inside the job rather than on the login node.

On the frame size: native is 640x480 and 27 GB. At 256x256 the cache is
about 6 GB and every interface in `configs/interfaces/` (the largest wants
144) is served by downsampling only. `--resize 128 128` halves that again
and is exact for PhysMamba, but BigSmall's 144 would then be upsampled.
Change the line in the script if you want a different size; the frames
are resized once more, to the interface's size, at load time regardless.

Reruns skip stores already finished, so a job that times out can simply
be resubmitted; `--overwrite` on the `pure-preprocess` line rebuilds.

When it finishes, the job log should end with the validator's verdict and
the store count should be fifty-nine:

```bash
tail -n 20 logs/<jobid>_PURE_Cache.out
ls -d /group/pgh004/carrow/caches/pure_zarr/*.zarr | wc -l
```

Every store must pass the validator; `docs/cache-contract.md` says what
is checked.

## 6. Point the dataset config at it

The only path in the whole run is the cache path. It lives in
`configs/datasets/pure_hpc.yaml`, which is `pure.yaml` with one key
overridden:

```yaml
BASE: pure.yaml
CACHED_PATH: "/group/pgh004/carrow/caches/pure_zarr"
```

Edit `CACHED_PATH` if the cache landed elsewhere, and commit that. Every
other config is path-free: the interface, the recipe and the model config
describe the experiment, not the machine. `runs/` and `logs/` are relative
to the checkout and both are gitignored.

Because the dataset is loaded as `pure_hpc`, the folds land in
`runs/pure_hpc_physmamba/` as `PHYSMAMBA_PURE_HPC.<participant>_<stamp>`, and
`--datasets` and `--test-participant-dataset` both say `pure_hpc`. The alternative, editing
`pure.yaml` in place on the cluster, works too but leaves the checkout
dirty, which every fold's `config.yaml` then records.

## 7. Submit

```bash
cd /group/pgh004/carrow/repo/remote-physiology
sbatch .slurm_scripts/PURE_PhysMamba_3ep.slurm          # all ten participants
sbatch .slurm_scripts/PURE_PhysMamba_3ep.slurm 01 02    # or just these folds
```

What the script asks for, and why:

| Line | Value | Reason |
| --- | --- | --- |
| `--partition` | `gpu` | the dev/test partition; if it sits pending, `scancel` and resubmit to `pophealth` with `--gres=gpu:a100:2`, then `medical` with `--gres=gpu:h100:2` |
| `--gres` | `gpu:v100:2` | `--parallel 2` times `--nproc-per-node 1` |
| `--cpus-per-task` | `8` | `--parallel 2` times the recipe's `NUM_WORKERS` of 4 |
| `--mem` | `32G` | two PhysMamba folds at 128x128, batch 4, fit comfortably |
| `--time` | `2:00:00` | a fold took nine minutes on the dev box; ten folds two at a time is under an hour |

The command inside is the dev-box command with two numbers added:
`--parallel 2 --nproc-per-node 1`. Nothing else differs. `main.py` checks
the participants against the cache before the first fold, deals the two
GPUs to the two running folds through `CUDA_VISIBLE_DEVICES`, and derives
its ports from the job id, so nothing in the script names a port.

## 8. Watch it

```bash
squeue -u $USER
tail -f logs/<jobid>_PURE_PhysMamba_3ep.out       # one line per fold start and finish
tail -f runs/pure_hpc_physmamba/PHYSMAMBA_PURE_HPC.01_*/log.txt    # a fold's own output
cat runs/pure_hpc_physmamba/PHYSMAMBA_PURE_HPC.01_*/losses.csv     # grows by a row per epoch
```

The job log says `fold 1/10: holding out pure_hpc 01 on GPU(s) 0 -> runs/pure_hpc_physmamba/...`
as each fold starts and `fold 1/10: done in ...s` as each finishes. A
failed fold says so with its `log.txt` path; the running fold finishes, no
new fold starts, and the job exits non-zero naming the failures.

## 9. Bring the results back

The whole sweep is about 250 MB. From PowerShell on the dev box:

```powershell
scp -r <user>@<login-node>:/group/pgh004/carrow/repo/remote-physiology/runs/pure_hpc_physmamba D:\runs_hpc\
```

Each fold's `config.yaml` records the commit it ran on, the resolved cache
path and the exact `scripts/run.py` command, so a fold can be reproduced
from the directory alone.

## If something goes wrong

| Symptom | Cause | Fix |
| --- | --- | --- |
| `mamba_ssm` import fails, or `uv sync` dies in a compiler | CUDA toolkit not on the path when the extension built | `module load cuda` before `uv sync`, inside an allocation |
| `--test-participant-dataset 'pure_hpc' is not one of the loaded datasets` | flags name different datasets | both flags say `pure_hpc` |
| `PURE_RAW does not exist` in the cache job's log | the raw dataset is not where the script says | edit `PURE_RAW` at the top of `PURE_Cache.slurm` |
| the cache job fails while syncing the cacher's environment | the compute node cannot download packages or Python 3.12 | run `uv sync --project dataset/cachers/pure` once on the login node (a download, not compute) and resubmit |
| `no admitted store` or `matches no admitted store` | wrong `CACHED_PATH`, or the cache job did not finish | check the path in `pure_hpc.yaml`; count the stores and read the cache job's log |
| `--parallel 2 x --nproc-per-node 1 needs 2 GPUs and 1 are visible` | `--gres` and the two numbers disagree | make `--gres` equal their product |
| `$'\r': command not found` | a CRLF SLURM script | `git add --renormalize .` and recommit; `.gitattributes` prevents it from here on |
| Job pending for long | partition busy | the escalation ladder in step 7 |
| CUDA out of memory in a fold's `log.txt` | two folds on one card, or a bigger interface than this one | `--parallel 1`, or run `tools/memory_report.py` with the same four config flags to size it |
