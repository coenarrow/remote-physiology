# remote-physiology

Camera-based estimation of physiological signals — not just heart rate. This is a heavily diverged fork of [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox).
rPPG-Toolbox was built to predict PPG-like waveforms from facial videos.
This repository is built around the idea of neck-focussed videos, with models able to predict **multiple signals at once**.
We've gone slightly further, and enforce a bit more standardisation.

## Standardised components

### The datasets

How to add a new dataset is explicitely outlined in [adding_a_dataset](docs\adding_a_dataset.md).
To add a new dataset to the repo, it requires a `cacher`, which processes the dataset into a `zarr` formatted directory.
This, compared with `hdf5` allows multiple read-writes, so training can be more easily parallelised.

### The models

Adding a new model is explicitely outlined in [adding_a_model](docs\adding_a_model.md).
We're enforcing models to be able to accept arbitrary numbers of channels, frame-size, and duration (all within reason).
Furthermore, preprocessing steps are to be included directly in the model class themselves, so each model only ever accepts raw frames directly.

We have ported over a number of models, from the implementation in [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox).
Where these models were only designed to predict a single waveform, we define the [`PerTraceCopies` class](src/models.py), where we make `n` copies of the model, in parallel, to predict `n` traces.

Models designed with the idea of predicting multiple waveforms are of the [`MultiTraceModel` class](src/models.py).


### The labels

Neck-focussed videos could feasibly predict mutliple waveforms. As such, we've set up the repository to be able to handle the following waveform inputs:

- PPG: Photoplethysmography signals, like the finger-based PPG signal often used in multiple datasets
- RR: Respiratory rate, typically collected by chest straps
- ECG: Single lead ECG signals
- ABP: Arterial blood pressure signals - these must be continuously collected.
- CVP: Central venous pressure signals, also known as right atrial pressure.

Other continuous waveform signals may be worth adding later, but these are the currently supported waveforms.

##

## The pipeline

One store per recording in a **zarr cache** written by the preprocessor.

**Labels are normalised by signal, not by config.** Every label window is
handled one of two ways, fixed by which signal it is
([`src/signal_transforms.py`](src/signal_transforms.py), the `SIGNALS`
table):

| Signals | Label the model trains on |
| --- | --- |
| PPG, ECG, respiration (RR) | z-scored over the window: only the waveform matters |
| everything else (ABP, CVP, SpO2, ...) | raw, in physical units: the level is part of the prediction |

The window's mean and standard deviation ride in the batch, so predictions
and labels come back in physical units exactly. There is no interface key
for this and no switch; a new signal picks its side when it is added to the
table. Frames likewise reach every model raw, and each backbone applies its
own published input normalisation as its first stage.

## Getting Started

### Installation

```bash
git clone --recurse-submodules https://github.com/coenarrow/remote-physiology.git
```

#### Linux and MacOS

Installation is straightforward;

```bash
uv sync
```

#### Windows

There are no pre-compiled binaries for `mamba-ssm` and `causal-conv1d` for Windows, so they need to be built. From powershell, run the following;

```powershell
Set-ExecutionPolicy -Scope Process Bypass
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
& "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:DISTUTILS_USE_SDK = 1; $env:MSSdk = 1
uv sync --no-install-package mamba-ssm --no-install-package causal-conv1d
uv sync
```

### Synthetic neck dataset

We provide a synthetic neck submodule as part of the project, to demonstrate the functionality of the toolbox [`tools/synthetic_neck`](tools/synthetic_neck) (its README documents the presets, the traces and what the frames carry).
It's a stand-in for Neckflix: neck videos with a propagating carotid and jugular pulse and exact ABP, CVP, ECG, PPG and respiration ground truth.
The generator writes cache-contract stores directly, so the result is a dataset compliant with the caching requirements named by [`configs/datasets/synthetic_neck.yaml`](configs/datasets/synthetic_neck.yaml) and specified by [`dataset/data_loader/SYNTHETIC_NECK.md`](dataset/data_loader/SYNTHETIC_NECK.md).

[`tools/synthetic_neck_cli.py`](tools/synthetic_neck_cli.py) runs the generator in this project's environment, so `--device auto` renders on the GPU (on a GPU, `--jobs 1` is usually fastest).
Each command below writes 8 recordings of 128x128 pixel frames, one per prior preset (`demo` has the pulse turned far up so it is visible by eye, `high_snr` is the midpoint, `base` is the plausible generator):

```bash
  uv run python tools/synthetic_neck_cli.py generate --n 50 --size 128 --device auto --jobs 2 --compression 5 --priors tools/synthetic_neck/priors/demo.yaml --out data/synthetic_neck_demo
```

```bash
  uv run python tools/synthetic_neck_cli.py generate --n 50 --size 128 --device auto --jobs 2 --compression 5 --priors tools/synthetic_neck/priors/high_snr.yaml --out data/synthetic_neck_high_snr
```

```bash
  uv run python tools/synthetic_neck_cli.py generate --n 10 --size 128 --device auto --jobs 2 --compression 5 --priors tools/synthetic_neck/priors/base.yaml --seed 2026 --out data/synthetic_neck
```

### Cache Validator

We provide a tool to validate caches, checking that they are in the correct format.
After generating the synthetic_neck dataset, we can test it via:

```bash
uv run python tools/validate_cache.py data/synthetic_neck_demo
```

### Viewer

We also provide a `napari` based viewer for inspection of data.
To view the synthetic neck dataset, you can use:

```bash
# browse it in napari: every store, perspective, modality and trace (omit the path to pick the folder)
uv run python tools/viewer.py data/synthetic_neck_demo
```

## Training and Testing Separately

### Training an uninitialised model

You can train an unitialised model across on single dataset, using:

```bash
# on a single dataset
uv run scripts/run.py \
--datasets synthetic_neck_demo \
--config configs/default_configs/physmamba_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml \
--epochs 10
# default output ./runs/PHYSMAMBA_SYNTHETIC_NECK_DEMO.all_<datetime>

# on multiple datasets
uv run scripts/run.py \
--datasets synthetic_neck_demo synthetic_neck_high_snr \
--config configs/default_configs/physmamba_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml \
--epochs 10
# default output example ./runs/PHYSMAMBA_SYNTHETIC_NECK_DEMO.all-SYNTHETIC_NECK_HIGH_SNR.all_<datetime>
```

The config file describes how the dataset is passed to the model, ensuring the model always gets correctly formatted data.

#### Continuing training from a model checkpoint

Say we wanted to take a model that was already trained, and the output has been saved in `./runs/PHYSMAMBA_SYNTHETIC_NECK_DEMO.all_<datetime>`, and now we wanted to train it further.
We could continue to train it on the original dataset, by using:

```bash
uv run scripts/run.py \
--init-from runs/PHYSMAMBA_SYNTHETIC_NECK_DEMO.all_<datetime> \
--epochs 10
```

Or we could take it and train it further on an different dataset at a lower learning rate, by using:

```bash
uv run scripts/run.py \
--init-from runs/PHYSMAMBA_SYNTHETIC_NECK_DEMO.all_202609291413 \
--datasets synthetic_neck_high_snr \
--epochs 10 \
--lr 0.0003
```

### Evaluating a pretrained model on a dataset

Now we have our pretrained models, let's try taking it and testing it on a new dataset.
If we take the fine-tuned model we had (first trained on the demo, and then on the high_snr), and we now want to test it on the noisiest of the datasets, we can do this by running:

```bash
# uses last epoch by default, tests on whole synthetic_neck dataset
uv run scripts/test.py \
--checkpoint runs/PHYSMAMBA_SYNTHETIC_NECK_HIGH_SNR.all_202609291436 \
--test-dataset synthetic_neck

# or if you wanted to pick a specific epoch
uv run scripts/test.py \
--checkpoint runs/PHYSMAMBA_SYNTHETIC_NECK_HIGH_SNR.all_202609291436 \
--test-dataset synthetic_neck \
--epoch 10

# or if you want to test only on a specific participant of a specific dataset
uv run scripts/test.py \
--checkpoint runs/PHYSMAMBA_SYNTHETIC_NECK_HIGH_SNR.all_202609291436 \
--test-dataset synthetic_neck  \
--test-participant-id 1 \
--epoch 10
```

## Combined Train and Test

### Leave one out folds (Experiments)

Given the limited data availability of neck-focused datasets, we want to do leave-one-out folds, at significantly higher cost of multiple training runs.
For example, if we have N participants, we will need to train N different models, and then test on our held out participant.

One model on a set of datasets, holding out one participant after another:
`main.py` runs the folds, each one exactly the `scripts/run.py` command
above with one participant held out, as its own subprocess with its own run
directory and `log.txt` under `runs/<dataset>_<model>/` (`runs/synthetic_neck_physmamba/`;
`--experiment NAME` picks another name). Each fold's `config.yaml` records
the command that reproduces it alone. Nothing is written at the experiment
level.

Models can be trained on more than 1 dataset, but we need to specify not only the test dataset, but the id of the participant in that dataset.

It can be run like:

```bash
uv run scripts/run.py \
--datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo \
--test-participant-dataset synthetic_neck \
--test-participant-id 1 \
--config configs/default_configs/physmamba_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml \
--epochs 10 \
--batch-size 2 \
--num-workers 4 \
--runs-dir runs/synthetic_neck_high_snr+synthetic_neck_demo+synthetic_neck
```

`main.py` is set up to run leave-one-out folds by default.
We can do this by running

```bash
# every participant of synthetic_neck in turn, one fold at a time
uv run main.py \
--datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo \
--test-participant-dataset synthetic_neck \
--config configs/default_configs/physmamba_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml \
--epochs 10 \
--batch-size 2 \
--num-workers 4 \
--experiment synthetic_leave_one_out
```

We have also set up `main.py` such that it is easily portable for HPC clusters with multiple gpus.
`--parallel K` runs K folds at a time;
`--nproc-per-node N` gives each fold N processes, one per GPU.

```bash
# every participant a fold, two folds at a time, one GPU per fold (folds share a GPU when fewer than 2 are visible)
uv run python main.py \
--datasets synthetic_neck \
--test-participant-dataset synthetic_neck \
--parallel 2 \
--config configs/default_configs/physmamba_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml \
--epochs 10

# every participant a fold, one fold at a time, each across 4 GPUs under torch.distributed.run (Linux only)
uv run python main.py \
--datasets synthetic_neck \
--test-participant-dataset synthetic_neck \
--nproc-per-node 4 \
--config configs/default_configs/physmamba_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml \
--epochs 10

# every participant a fold, two folds at a time, each across 2 GPUs: needs 2 x 2 = 4 GPUs visible (Linux only)
uv run python main.py \
--datasets synthetic_neck \
--test-participant-dataset synthetic_neck \
--parallel 2 \
--nproc-per-node 2 \
--config configs/default_configs/physmamba_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml \
--epochs 10
```

### Evaluating a finished run

Once the folds have trained and scored their held-out participants, pool them into one report:

```bash
# the last epoch of every fold; recordings covered for 20 to 30 s count
uv run scripts/evaluate.py runs/synthetic_benchmark_physmamba

# or a specific epoch, other bounds, and further attributes to describe
uv run scripts/evaluate.py runs/synthetic_benchmark_physmamba \
--epoch 8 \
--segment-min 20 --segment-max 30 \
--additional-params abp_site synthetic_neck.traces.heart_rate_bpm
```

The report lands in `runs/synthetic_benchmark_physmamba/evaluation/` as `report.md` and `report.html`, with every table and figure beside it.
It covers who was tested, the clinical accuracy of each parameter (ISO 81060-3 statistics, Bland-Altman plots) and the usual deep-learning metrics; `docs/evaluation.md` lists what each table holds.
Every fold must have finished at least one epoch, so run it once the experiment is complete.

### Train on dataset(s) A, B..., test on dataset X

In a typical rPPG workflow, you train a model on one or more datasets, and then test it on an unseen dataset.
We have built in this functionality, and this can be run using:

```bash
TODO
```

## Models/Algorithms

We have taken and tried to keep very similar implementations of the deep learning algorithms implemented in rPPG Toolbox.
From there, we have implemented the following:

- DeepPhys
- EfficientPhys
- FactorizePhys
- PhysFormer
- PhysMamba
- PhysNet
- RhythmFormer
- TS-CAN
- iBVPNet

We have also implemented;

- PhysMamba2
- PhysMamba3

using updated versions of PhysMamba, using the Mamba2 and Mamba3 layers.

We can run a benchmark to just test the learning capacity of all of these models by individually running the following (using LOO folds);

```bash
# PhysMamba
uv run main.py --datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo --test-participant-dataset synthetic_neck --config configs/default_configs/physmamba_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml --epochs 10 --batch-size 2 --num-workers 4 --experiment synthetic_benchmark_physmamba

# FactorizePhys
uv run main.py --datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo --test-participant-dataset synthetic_neck --config configs/default_configs/factorizephys_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml --epochs 10 --batch-size 2 --num-workers 4 --experiment synthetic_benchmark_factorizephys

# PhysFormer
uv run main.py --datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo --test-participant-dataset synthetic_neck --config configs/default_configs/physformer_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml --epochs 10 --batch-size 2 --num-workers 4 --experiment synthetic_benchmark_physformer

# PhysNet
uv run main.py --datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo --test-participant-dataset synthetic_neck --config configs/default_configs/physnet_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml --epochs 10 --batch-size 2 --num-workers 4 --experiment synthetic_benchmark_physnet

# RhythmFormer
uv run main.py --datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo --test-participant-dataset synthetic_neck --config configs/default_configs/rhythmformer_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml --epochs 10 --batch-size 2 --num-workers 4 --experiment synthetic_benchmark_rhythmformer

# iBVPNet
uv run main.py --datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo --test-participant-dataset synthetic_neck --config configs/default_configs/ibvpnet_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml --epochs 10 --batch-size 2 --num-workers 4 --experiment synthetic_benchmark_ibvpnet

# PhysMamba2
uv run main.py --datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo --test-participant-dataset synthetic_neck --config configs/default_configs/physmamba2_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml --epochs 10 --batch-size 2 --num-workers 4 --experiment synthetic_benchmark_physmamba2

# PhysMamba3
uv run main.py --datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo --test-participant-dataset synthetic_neck --config configs/default_configs/physmamba3_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml --epochs 10 --batch-size 2 --num-workers 4 --experiment synthetic_benchmark_physmamba3

# efficientphys
uv run main.py --datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo --test-participant-dataset synthetic_neck --config configs/default_configs/efficientphys_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml --epochs 10 --batch-size 2 --num-workers 4 --experiment synthetic_benchmark_efficientphys

# TS-CAN
uv run main.py --datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo --test-participant-dataset synthetic_neck --config configs/default_configs/tscan_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml --epochs 10 --batch-size 2 --num-workers 4 --experiment synthetic_benchmark_tscan
```

## Tips for running on HPC/GPUs

GPUs are dealt to the running folds through `CUDA_VISIBLE_DEVICES`: one each,
shared, when N is 1 (`tools/memory_report.py` says how many folds fit on a
card), disjoint groups of N otherwise. Both default to 1, so the plain
command is the same on the dev box and on the cluster, where the SLURM file
adds only the allocation and the two numbers. On Windows N stays 1, because
that torch build has no libuv and the launcher's rendezvous store cannot
start without it. On a terminal each running fold shows a progress bar over
its epochs with the latest training loss; in a log file the bars are silent.
A fold that fails stops new folds from starting; the running ones finish,
and the exit names the failures. `--limit-windows` passes through to every
fold. On the cluster,
[`.slurm_scripts/PURE_PhysMamba_3ep.slurm`](.slurm_scripts/PURE_PhysMamba_3ep.slurm)
and [`.slurm_scripts/Neckflix_PhysMamba_4GPU.slurm`](.slurm_scripts/Neckflix_PhysMamba_4GPU.slurm)
are the templates, and [docs/hpc_pure_physmamba.md](docs/hpc_pure_physmamba.md)
walks a run through end to end.

To put another architecture on the contract, new or migrated from upstream,
follow [docs/adding_a_model.md](docs/adding_a_model.md): one backbone package,
one registry line and builder, one config YAML (the paper's model section,
interface and training recipe), and the command above.
Starting points to copy are
[`neural_methods/model/_model_template/_template.py`](neural_methods/model/_model_template/_template.py) and
DeepPhys's [config file](configs/original_model_config/deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml).

The original papers are linked from the
[upstream README](https://github.com/ubicomplab/rPPG-Toolbox#notebook-algorithms).

## Citation, license, acknowledgement

This fork exists because rPPG-Toolbox was an excellent starting point. If you
use anything here that derives from it — the model implementations above, the
unsupervised methods, the evaluation ideas — please cite the original work:

```bibtex
@article{liu2022rppg,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu and Tang, Jiankai and Zhang, Yuzhe and Wang, Yuntao and Sengupta, Soumyadip and Patel, Shwetak and McDuff, Daniel},
  journal={arXiv preprint arXiv:2210.00716},
  year={2022}
}
```

The upstream license (Responsible AI Source Code License) carries over; see
[LICENSE](LICENSE).
