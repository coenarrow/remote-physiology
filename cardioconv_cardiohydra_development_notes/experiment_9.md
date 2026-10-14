# Experiment 9 — Synthetic training after experiment 8 (2026-10-12)

The user asked to check that both models still train, as before: do they learn
region isolation and decent waveforms on the synthetic split?

## Setup

The protocol is experiments 2 and 3's, with these changes:
- **Input size: 72 px.** 128 px no longer fits with full-size masks
  (`experiment_8.md`).
- **Shrink factors:** `RATE_DOWNSCALE` 2, `WAVELET_DOWNSCALE` 1.
- **Configs:** `runs/dev_cardio/exp9_configs/`, scratch configs built on the
  H72W72 ones.

Unchanged:
- Training on high_snr + demo for 30 epochs at batch 1.
- Testing on `synthetic_neck`, then `evaluate.py` and `visualise.py` with 6
  windows.

The chain script is `runs/dev_cardio/exp9_chain.sh`. Its exit codes are now
real: the old chains read `$?` after `$(date)` and always printed 0.

Timing: about 190 s per epoch for CardioConv and 198 s for CardioHydra, with
epoch 1 slower. Both runs exited 0, and neither has a `nonfinite_steps/`.

## Results

| Synthetic test, mean CCC | ABP | CVP |
| --- | --- | --- |
| **9 CardioConv** (72 px, masks full size, kinematics) | **0.19** | **0.24** |
| **9 CardioHydra** (72 px, masks full size) | **0.28** | **0.17** |
| 2 CardioConv (128 px, masks on a 64 px grid) | 0.33 | 0.28 |
| 3 CardioHydra (128 px, masks on a 64 px grid) | 0.33 | 0.22 |
| 4 FactorizePhys (72 px) | 0.25 | 0.18 |
| 4 RhythmFormer (72 px) | 0.21 | 0.09 |

The training loss (1 − CCC) at epoch 30:
- CardioConv: ABP 0.62, CVP 0.65. Experiment 2 had 0.49 / 0.61, flat from
  epoch 1.
- CardioHydra: ABP 0.49 and CVP about 0.70 by epoch 20, flat from epoch 5.

## The masks isolate nothing

Mean over the 6 visualised test windows, per channel (`scores.csv`):

| Model | Pulsatility effective px (of 5,184) | Pulsatility on neck (chance 0.618) | ABP mass in artery | CVP mass in vein | Vessel share of grid |
| --- | --- | --- | --- | --- | --- |
| 9 CardioConv | 5,120–5,134 | 0.619–0.631 | 0.008–0.010 | 0.012–0.021 | 0.023 |
| 9 CardioHydra | 5,091–5,105 | 0.619–0.626 | 0.008–0.009 | 0.012–0.015 | 0.023 |
| 2 CardioConv (reference) | not recorded | 0.626–0.687 | 0.006–0.012 | 0.014–0.030 | 0.023 |

What this shows:
- **The pulsatility map is uniform pixel speckle.** Its effective pixel count is
  within 2% of the whole grid, and its mass on the neck is at chance. On the
  64 px grid, experiment 2's map was weak but above chance.
- **The trace masks are speckle too.** Their mass inside the true artery and
  vein is at or below chance.
- **The rate is right.** For example, store 3 reads 87.5 bpm against a label of
  87.9.

## Reading

This is the failure `experiment_8.md` warned of:
- **The map scores noise.** With the field upscaled to the full grid, each
  pixel's pulsatility index divides a smooth band power by its own noisy
  temporal variance, so the map ranks pixel noise.
- **The priors cannot regroup it.** The bounded correction and the TV prior
  (`PULSATILITY_TV` 0.005, tuned on the shrunk grid) cannot pull it back into
  regions.
- **The shrink did the averaging.** At factor 2, the shrink averaged that noise
  over a 2x2 block before anything read it.

The synthetic vessels barely show outside depth
(`synthetic-neck-vessels-only-in-depth`). So the most this map could find
here is the neck, and it does not find even that.

## Not yet separated

Four things changed against experiment 2, so this run cannot separate them:
- **Grid size.** The masks work on a 72 px grid, with no averaging before the
  transform.
- **Kinematics (CardioConv only).** The kinematics branch starts at zero.
- **The rate network sees 36 px, not 64.** Its waveform term is 0.08, so it
  works.
- **Input size.** The input is 72 px rather than 128.

The cleanest next test is CardioConv at `WAVELET_DOWNSCALE` 2 with the masks
kept on the shrunk grid. Today the field is always upscaled, so this needs a
switch for whether to upscale. That would show whether resolution alone
explains the loss of isolation.

## Diagnostic, and experiment 10: area-average resize (evening of 2026-10-12)

**The diagnostic.** `runs/dev_cardio/exp9_configs/pulsatility_diagnostic.py`
plotted the pulsatility index's two terms for test store 3, window 210, at four
resolutions. The figure and CSV are in
`exp9_cardioconv_synth_72/epoch_30/diagnostics/`.

- **The input carried a lattice.** At 72 px, both band power and temporal
  variance showed a regular grid pattern. The cause was the dataset's resize,
  bilinear without antialias. At 128 -> 72 it blends 1-4 source pixels in a
  repeating pattern, so per-pixel noise varied in a grid.
- **Averaging recovers the neck.** The index separated the neck from the rest
  by d' 0.22 (G) at 72 px, 0.62 at 36, 1.70 at 18, and 0.83 for the field from
  36 upscaled to 72.
- **The trained network adds almost nothing.** Its final map reached d' 0.36
  (G), little above the raw index.

**The fix.** `src/frame_transforms.resize_video` now takes the exact area average
when shrinking and stays linear when growing. It is identical to torch's area
mode at whole-number factors and to bilinear when upsampling. Per-pixel noise
variance at 128 -> 72 is now 0.15-0.31 (mean 0.21); bilinear gave 0.21-0.95
(mean 0.44). This changes every model's resized inputs.

**Experiment 10** repeats experiment 9's CardioConv run with only the resize
changed: `runs/dev_cardio/exp10_cardioconv_synth_72_area`, chain
`exp10_chain.sh`. It exited 0, has no `nonfinite_steps/`, and the epoch-30 rate
waveform term is 0.05.

| Synthetic test, mean CCC (median) | ABP | CVP |
| --- | --- | --- |
| 9, bilinear resize | 0.19 (0.17) | 0.24 (0.19) |
| **10, area resize** | **0.27 (0.31)** | **0.27 (0.19)** |
| 2, 128 px, masks on a 64 px grid | 0.33 | 0.28 |

| Masks, 6 test windows (scores.csv) | Pulsatility effective px | On neck (chance 0.618) | ABP mass in artery | CVP mass in vein |
| --- | --- | --- | --- | --- |
| 9 | 5,120-5,134 | 0.619-0.631 | 0.008-0.010 | 0.012-0.015 |
| 10 | 5,082-5,128 | 0.621-0.640 | 0.008-0.012 | 0.011-0.015 |

**What the resize fixed and what it did not.**
- **Waveforms improved.** ABP recovered most of the way to experiment 2, and
  CVP fully.
- **The ABP prediction now tracks beat by beat.** The store-3 panel shows the
  rate at 87.7 against a label of 87.9. The edge spikes remain.
- **The phase masks show structure.** In G, the ABP mask lies along the neck as
  a faint band, and the background takes the area off the neck.
- **The pulsatility map is still essentially uniform.** It is only marginally
  above chance on the neck.
- **Nothing lands on the vessels.** The trace masks' mass in the artery and
  vein is still at chance.

At full resolution, the per-pixel index does not separate the neck, even with
clean inputs. Separating it takes averaged evidence: wavelet factor 2, or more.
