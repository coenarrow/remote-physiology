# Experiment 12 — The coarse-grid design checked without retraining; the P033 tuning job (2026-10-14)

The user asked whether the masks still look good under the coarse-grid design
without a new training run, then to use that design in the HPC tuning job, and
to check the job locally while the A100s were busy. Nothing was trained beyond
two one-epoch smokes.

## The coarse-grid design

Session remote-physiology-c7 moved the field, power, kinematics and both mask
networks onto the `WAVELET_DOWNSCALE` grid; each window's maps are upscaled
bilinearly to the input grid, the clip is pooled under the fine maps, and the
TV priors act on the coarse grid. Factor 1 is lossless against main e20f6ec.
The change is in the working tree, not yet committed.

## Experiment 11b's weights on the coarse grid (no retraining)

The experiment 11b CardioConv checkpoint (epoch 30, trained with the field
upscaled back, masks at 72 px) was run with its saved `WAVELET_DOWNSCALE`
overridden at inference (`runs/dev_cardio/exp12_coarse_check/test_at.py`):
factor 1 reproduces experiment 11 exactly; factor 2 keeps the same mask
structures, smoother, with shares within 0.06. Scored over all of P033 (217
strided windows, 33 measurements):

| P033, CardioConv 11b weights | ABP CCC | CVP CCC | ABP r | CVP r | ABP MAE | CVP MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Experiment 11, full resolution | 0.51 | 0.31 | 0.70 | 0.49 | | |
| **Factor 2, coarse grid** | **0.507** | **0.316** | 0.70 | 0.50 | 15.4 | 2.9 |

The coarse grid costs nothing on this fold, and the weights trained at full
resolution transfer to it as they are.

## Memory at 200 px (the HPC interface)

One training step, rate gate open, bfloat16, `tools/memory_report.py` through
`runs/dev_cardio/exp8_probes/fit_probe.py`, on the 16 GB dev card (numbers
above 14.9 GB paged, and are still the allocator's own):

| 200 px, `WAVELET_DOWNSCALE` 2 | Batch 1 alloc / reserved | Batch 2 alloc / reserved |
| --- | --- | --- |
| CardioConv | 8.0 / 14.3 GB | 15.3 / 25.5 GB |
| CardioHydra | 8.7 / 14.3 GB | 17.3 / 26.9 GB |

Against the A100's 39.5 GB: batch 2 a card fits both models, batch 4 does not.
The full-resolution design did not fit batch 1 (26.3 and 30.2 GB reserved,
`experiment_11.md`).

## The tuning job

- **Configs:** `configs/hpc_tuning_p33/<arm>.yaml`, each `BASE:` on the
  `hpc_configs_200` cardio YAML (200 px, bf16, HoldCosine, both shrink
  factors 2) plus its one change. The six arms (agreed 2026-10-14):

  | Arm | Change |
  | --- | --- |
  | cardioconv_baseline | none |
  | cardioconv_tv0.02 | `PULSATILITY_TV` 0.005 -> 0.02 |
  | cardioconv_lr5e-4 | `LR` 1e-3 -> 5e-4 |
  | cardiohydra_baseline | none |
  | cardiohydra_tv0.02 | `PULSATILITY_TV` 0.005 -> 0.02 |
  | cardiohydra_mamba0.3 | `MAMBA_LR_SCALE` 0.1 -> 0.3 |

- **Script:** `.slurm_scripts/Neckflix_CardioTuning_P033_4GPU.slurm` runs
  every arm in turn, P033 held out, 30 epochs, `--parallel 1
  --nproc-per-node 4`, batch 1 a card (a global batch of 4, the sweeps'),
  into `runs/neckflix_hpc_tuning_p33_<arm>/`. About 2.5-3.5 h an arm.
- **Prerequisite:** the coarse-grid code must be committed and pushed before
  the cluster pulls.

## Local check of the job (2026-10-14, 11:11-11:24)

`runs/dev_cardio/exp12_coarse_check/chain_local_check.sh`: each baseline arm
through the SLURM script's own `main.py` command on the dev-box Neckflix
cache, 200 px, 1 epoch, 8 windows, one process (Windows cannot torchrun).

| Arm | Fold | Step time | Final loss | `nonfinite_steps/` |
| --- | --- | --- | --- | --- |
| cardioconv_baseline | done in 274 s, exit 0 | 169 s (paging) | 1.17 | none |
| cardiohydra_baseline | done in 328 s, exit 0 | 194 s (paging) | 1.33 | none |

Both wrote `epoch_01/{model.pt,test_records}` and `losses.csv` under
`runs/smoke_tuning_p33_<arm>/`. Covered: the `BASE` chain, the fold pool and
experiment layout, the coarse-grid code at 200 px bf16, a training step and
the held-out pass. Not covered: DistributedDataParallel across four cards,
the HPC cache path, and throughput.

## Carry-over to the cluster (what can be checked from here)

Neither model has run under DistributedDataParallel, and the dev box cannot
(no libuv in the Windows torch). DDP's one requirement that the model can
break, a gradient for every parameter on every step, was checked with
`grad_coverage.py`: one real step through `tools/memory_report.py` at 40 px,
batch 2, the rate gate forced open and forced closed (nothing passes, so the
closest sample runs with its priors weighted 0 and its waveform masked).

| Model | Gate | Parameters | Without a gradient | All-zero gradient |
| --- | --- | --- | --- | --- |
| CardioConv | open | 264 | 0 | 0 |
| CardioConv | closed | 264 | 0 | 34 (masks, mixer, head) |
| CardioHydra | open | 323 | 0 | 1 |
| CardioHydra | closed | 323 | 0 | 93 (masks, SSM predictor, head) |

The zero gradients are real zeros through the graph (the loss multiplies by
the mask rather than indexing), which is what DDP needs; a missing one would
have failed the step. CardioHydra is CUDA-only (Mamba3's Triton kernels), so
its check ran on the dev card.

## Decision: tune locally, on the recipe the cluster will run (afternoon of 2026-10-14)

The A100s were busy, and the user's principle is that whatever is tuned here
runs unchanged on the cluster. At 200 px the dev card paged on the recipe as
it stood (19.9 GB reserved in float32). The memory was the rate network, not
the masks:

| 200 px, CardioConv, fp32, batch 1, one step | Peak reserved |
| --- | --- |
| rate 2 / wavelet 2 | 19.9 GB (pages) |
| rate 2 / wavelet 4 | 11.2 GB |
| rate 4 / wavelet 4 | 3.9 GB |

So the shared recipe (`hpc_configs_200`, both cardio YAMLs) is now
`RATE_DOWNSCALE` 4 / `WAVELET_DOWNSCALE` 3: the rate network at 50 px and
the masks on a 66 px grid, the scale of the 128 px local runs. A 16-window
epoch timed the same at 200 px on that recipe as at 128 px, rate 2 /
wavelet 2: 3.9 s a step including start-up. In the real run the CardioConv
baseline settled at 1.4 s a step with the card at 97% and 8 GB used, so
about 17 min an epoch with the test pass and about 3 h for a 10-epoch arm. The local 200 px frame is the 128 px cache
upsampled; every stage that does work runs at or below 66 px, so none of
them sees invented detail.

Agreed: 10 epochs an arm, the two baselines first (required: no existing run
is on this recipe, and CardioHydra has none on the current code at all),
then the variants in the ranked order only if the baselines leave the
question open. Local runs are float32 (the Turing card has no bfloat16);
that and the cache are the only differences from the cluster.
`runs/dev_cardio/exp13_local_tuning/chain_baselines.sh` runs the baselines
into `runs/local_tuning_p33_<arm>/` (experiment 13).

Other cluster facts, from the finished sweeps' P033 folds: 628 training
windows and 217 test windows, the same counts as the local cache; the
`neckflix_hpc` config and cache path were used by every sweep; mamba-ssm
2.3.2.post1 and triton 3.8 have been pinned since 2026-09-19, before the
PhysMamba sweep ran on them, so the install exists, though no Mamba3 block
has run there yet. `uv.lock` is unchanged by the working tree.
