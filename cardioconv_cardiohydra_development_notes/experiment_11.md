# Experiment 11 — CardioConv on Neckflix with full-resolution masks (2026-10-12/13)

**The ask.** The user asked to run experiment 10's setup on Neckflix to see
whether the neck, carotid or jugular show up in the pulsatility and phase
masks. The setup: area-average resize, 72 px, rate /2, wavelet 1, masks at
input resolution. The run is the standard P033 hold-out for 30 epochs.

## Runs

- **exp11, `exp11_cardioconv_neckflix_p33_area`: batch 2, epochs 1-14.**
  - It paged from epoch 1: the GPU sat at 15.98 of 16 GB, steps took 16 s, and
    epochs about 74 min with the test pass.
  - The one-step probe of experiment 8 (13.7 GB) did not predict the steady
    state.
  - The user stopped it after epoch 14.
- **exp11b, `exp11b_cardioconv_neckflix_p33_area_b1`: epochs 15-30.**
  - It continued with `run.py --init-from .../epoch_14` at batch 1, using
    7.3 GB, 21 min of training per epoch and about 25 min with the test pass.
  - The datasets, model, recipe (constant LR) and hold-out are the same. Only
    the Adam state restarted.
  - It ran 14:26-21:08 on 2026-10-13, exited 0, and has no
    `nonfinite_steps/`.
  - `run.py --init-from` did not score the run, so `evaluate.py` was run
    afterwards.

## Results, P033 held out (epoch 30)

| Model (P033) | ABP CCC | CVP CCC |
| --- | --- | --- |
| **11b CardioConv**, full-res masks, kinematics | **0.51** | **0.31** |
| 5 CardioConv, old masks, shrunk grid | 0.56 | 0.29 |
| 7b CardioHydra, v2 masks, epoch 12 | 0.59 | 0.34 |
| 6a CardioHydra, epoch 30 | 0.69 | 0.01 (CVP mask died) |

**Level.**
- ABP mean is -13.7 mmHg off, and systolic -38.9: the predicted amplitude is
  too small.
- CVP mean is -2.2 mmHg off.
- The heart rate is right (-0.3 bpm).

**Level tracking within P033.** The prediction does not follow P033's level:

| Parameter | ref_sd | pred_sd | slope_within |
| --- | --- | --- | --- |
| ABP mean | 2.8 | 4.0 | 0.50 |
| CVP mean | 1.96 | 0.15 | 0.01 |

The CVP prediction is nearly constant in level.

## Masks

The panels are in `epoch_30/visuals/neckflix/`, with interim ones drawn on the
CPU at exp11 epochs 6 and 13 and exp11b epoch 19. The masks were stable from
epoch 6 onwards. Unlike the synthetic runs, they have clear structure:

- **Pulsatility** sits on the neck: about 1,300-3,000 of 5,184 pixels, with
  clothing and background dark. Within the neck it picks out elongated streaks
  along it.
- **ABP and CVP take different structures.**
  - In R1 (0°), R and G give ABP a lower horizontal streak (46-57%) and CVP a
    separate streak and spot above it (21-33%); B swaps them.
  - In R5 (90°), depth gives ABP a compact spot low on the neck (64%), and R
    and G give it an oblique streak up the neck.
- **Unverified as carotid or jugular.** Neckflix has no vessel labels, so this
  is a visual read. The trace assignment still flips between channels: this
  is experiment 7's open phase-reference problem.

## Held-out curve: more epochs, or a different rate? (2026-10-14)

Every second epoch's P033 records were scored on scratch copies, under
`runs/dev_cardio/exp11_epoch_curve/`:

| Epoch | ABP CCC | CVP CCC | ABP r | CVP r | ABP mean err | ABP sys err | CVP mean err |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.42 | 0.15 | 0.51 | 0.25 | -11.8 | -39.4 | -1.6 |
| 6 | 0.49 | 0.21 | 0.65 | 0.32 | -11.4 | -38.4 | -1.7 |
| 10 | 0.47 | 0.25 | 0.66 | 0.37 | -12.4 | -40.2 | -1.9 |
| 14 | 0.48 | 0.28 | 0.66 | 0.43 | -11.9 | -40.1 | -2.1 |
| 18 | 0.53 | 0.28 | 0.69 | 0.46 | -11.5 | -37.6 | -2.0 |
| 22 | 0.42 | 0.30 | 0.64 | 0.47 | -13.2 | -43.6 | -2.0 |
| 26 | 0.53 | 0.32 | 0.71 | 0.50 | -12.3 | -37.6 | -2.1 |
| 28 | 0.55 | 0.31 | 0.70 | 0.51 | -9.4 | -35.9 | -2.0 |
| 30 | 0.51 | 0.31 | 0.70 | 0.49 | -13.7 | -38.9 | -2.2 |

**Reading.**
- **Epochs: no clear gain from more.**
  - ABP shape (r) had plateaued at about 0.66-0.70 from epoch 10.
  - CVP shape rose steadily, 0.25 -> 0.50, and flattened over the last 4-6
    epochs.
  - More epochs would gain little, and only on CVP.
- **Learning rate: noisy at the end.**
  - At the constant 1e-3, the held-out ABP CCC swings by up to 0.1 between
    neighbouring epochs (0.53 at 18, 0.42 at 22, 0.55 at 28), with no trend.
  - A decay over the last third, step or cosine, would likely settle it near
    the top of that range.
- **Level: not training-limited.**
  - ABP systolic is about -38 mmHg from epoch 2 to 30. ABP mean is -9 to -15.
  - CVP mean drifts from -1.6 to -2.2.
  - Neither more epochs nor a rate change addresses that.
- **The training loss is not comparable across epoch 14/15.** It drops sharply
  where the batch size changed, from 2 to 1, while the held-out scores do not
  jump.
- **Caveat: one subject, 33 measurements.** Differences of a few hundredths
  are within this curve's own noise.

## Learning-rate decay added (2026-10-14)

CardioHydra shows the same late-run swing. Its held-out P033 records were
scored per epoch under `runs/dev_cardio/exp7b_epoch_curve/`:

| Run (constant rate) | ABP CCC | CVP CCC |
| --- | --- | --- |
| 6a, epochs 16-26, every second | 0.66, 0.65, 0.70, 0.65, 0.66, 0.72 | 0.35-0.38, flat |
| 7b, epochs 2-12 | 0.44 -> 0.59, still rising | peak 0.39 at 8, then 0.34 |

**Change.** `HoldCosine` joins `src/trainer.py`'s schedules: the recipe's rate
for the first two-thirds of the run's steps, then a half cosine to zero over
the last third (`COSINE_TAIL`). The scheduler scales each parameter group
from its own rate, so CardioHydra's trunks (0.1x) decay in proportion. A run
continued with `--init-from` starts the schedule afresh over its own epochs.
All 12 cardio configs now use it; the baseline models keep `OneCycle`, and
the five paper configs whose upstream `StepLR` never fires keep `Constant`.
Not yet trained with.
