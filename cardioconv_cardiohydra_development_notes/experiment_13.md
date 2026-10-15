# Experiment 13 — Local tuning on the HPC recipe; the background bin follows coherence (2026-10-14)

Experiment 12 set the shared recipe (`hpc_configs_200`, both cardio models:
200 px, `RATE_DOWNSCALE` 4, `WAVELET_DOWNSCALE` 3, HoldCosine, batch 1 a
card) and the plan: tune on the dev card, P033 held out, 10 epochs an arm,
baselines first. This note holds the runs and one design change made
between them.

## Run 1: CardioConv baseline, old background rule (the control)

`runs/local_tuning_p33_cardioconv_baseline/`, started 12:58, 1.4 s a step,
8 GB, 16 min an epoch. Held-out P033, scored on scratch copies every second
epoch (`runs/dev_cardio/exp13_local_tuning/curve_cardioconv_baseline/`):

| Epoch | ABP CCC | ABP r | CVP CCC | CVP r | ABP mean err | ABP sys err |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.20 | 0.27 | 0.08 | 0.14 | -10.0 | -44.0 |
| 2 | 0.36 | 0.46 | 0.14 | 0.23 | -10.0 | -41.3 |
| 4 | 0.32 | 0.47 | 0.19 | 0.32 | -10.4 | -43.5 |
| 6 | 0.32 | 0.55 | 0.21 | 0.34 | -16.8 | -49.2 |
| 8, 10 | see the end of this note | | | | | |

Training ABP CCC loss flat at 0.42-0.45 from epoch 1; the rate network's
waveform term 0.22 (the 72 px runs reached 0.05-0.08); no non-finite steps.
Masks (6 held-out windows, `epoch_NN/visuals/`, `panel.png` and the new
`overlay.png`): the pulsatility maps find the neck; green's ABP mask lies
along the carotid's course in the sitting window; CVP mass sits separately,
with some off the neck (jaw in one window, collar scatter in depth). The
background held 24-30 % of the pulsatile mass in every channel from epoch 2
on, unchanged by training.

## The background bin (changed between run 1 and run 2)

The user asked why the background mask looked like a trained region and
whether its priors were in the loss. They were not: the TV and coherence
priors run over the trace classes only, and the one background term
(`background_pulsatile`, a hinge above an even third) read 0.000 in every
epoch. The appearance was the complement of two smooth masks, drawn times
pulsatility. But the rule behind it had no physical basis:

- a trace's bin is `2 R cos(phase - centre)`, the background's was 0, so a
  perfectly coherent pixel (R = 1, logits +2 / 0 / -2) still gave the
  background 12 %, which is the floor seen above;
- a wandering pixel (R = 0) was a three-way tie, left to the bounded network.

**Change** (`phase_masker.py`): the background's bin is the log-odds of "no
coherent source here", `ln(BACKGROUND_PRIOR_ODDS) - n R^2`, where `R` is the
window's mean resultant length of the relative phase at the fundamental and
`n` the window's independent phase samples. The argument: `R` is the
circular measure of phase concentration and a monotone function of the
pixel's band SNR (phase jitter variance about `1 / (2 SNR)`, so `R` about
`exp(-1 / (4 SNR))`); the trace bins are already the von Mises
log-likelihood of "concentrated at the centre"; `n R^2` is the Rayleigh
statistic, the log-likelihood ratio of a coherent source against uniform
phase; the prior is the base rate, vessels a few percent of the neck (2.3 %
of the synthetic grid), so 40 to 1 against each trace.

**Why `n` matters, measured.** A first version used a linear bin,
`PHASE_GAIN (1 - 2 R)`, crossing over at `R = 1/3`, on the assumption that a
noise pixel's `R` over 300 frames is near 0. It is not: the Morlet's time
support (`exp(-t^2 / B)`, `B = 1.5`, about `sqrt(2 B) = 1.7` beats between
independent coefficients) makes a noise pixel's phase drift rather than
jump, so a 10 s window at 67 bpm holds about six independent phases.
`runs/dev_cardio/exp13_local_tuning/resultant_diagnostic.py`, on the control's
epoch-6 weights, window R3 at 45 degrees, start 0, and on white noise of
the same shape through the same model:

| `R` quantiles 5 / 25 / 50 / 75 / 95 % | R | G | I | D |
| --- | --- | --- | --- | --- |
| white noise | .09 .21 .33 .45 .63 | .08 .20 .31 .43 .61 | .09 .20 .31 .43 .60 | .09 .22 .33 .46 .64 |
| real, top 2 % pulsatile pixels | .81 .91 .98 .99 .99 | .81 .87 .96 .97 .98 | .90 .97 .99 .99 1.0 | .05 .09 .16 .26 .42 |
| real, top 10 % pulsatile | .36 .70 .90 .97 .99 | .46 .75 .87 .95 .98 | .20 .65 .92 .98 1.0 | .07 .13 .22 .36 .52 |
| real, bottom half by pulsatility | .00 .00 .29 .58 .85 | .00 .00 .39 .70 .89 | .00 .00 .30 .63 .85 | .00 .00 .14 .33 .56 |

- Noise `R` has median 0.32 and 95th percentile 0.62: the Rayleigh null
  `sqrt(ln 2 / n)` at `n = 6.4`, which is what `beats / sqrt(2 B)` gives for
  this window. Under the linear bin 47 % of noise pixels went to a trace.
- The neck's pulsatile spots in colour and infrared hold `R` 0.9-1.0, far
  above the noise tail; they are coherent sources.
- Depth's pulsatile spots are *not* coherent (`R` 0.1-0.4). Under either new
  rule depth is mostly background; its "pulsatility" on this window is not
  phase-locked to the pulse.
- Real low-pulsatility pixels sit above white noise (95th percentile 0.85),
  because video noise is not white: flicker and slow motion give it phase
  structure. Amplitude, the pulsatility map, already handles those.

So `n` is computed per window from the estimated rate (`beats / sqrt(2 B)`,
detached: a feature, not a path), and the crossover on a 10 s window falls
at `R` about 0.6, above 95 % of noise pixels and below the coherent spots.

| Pixel, 10 s window at 67 bpm (`n` 6.4) | Trace at centre | Background | Shares (trace / other / background) |
| --- | --- | --- | --- |
| R = 1 | +2.0 | -2.7 | 0.97 / 0.02 / 0.01 |
| R = 0.8 | +1.6 | -0.4 | 0.85 / 0.03 / 0.12 |
| R = 0.6 (crossover) | +1.2 | +1.4 | 0.44 / 0.04 / 0.52 |
| R = 0.33 (noise median) | +0.7 | +3.0 | 0.09 / 0.02 / 0.89 |
| R = 0 | 0 | +3.7 | 0.02 / 0.02 / 0.96 |

Below `R` of about 0.5 the background's margin exceeds the network's
pairwise reach (1.8 nats), so a pixel with no phase evidence is background
whatever the network says; between 0.5 and 0.8 the network decides.

What it does not do: ballistocardiographic motion is phase-locked and keeps
`R` near 1, so it still lands in a trace; that is the kinematics stage's job.

On the control's epoch-6 weights (never trained under it), two windows:
background share R 36-37 %, G 20-25 %, B 46-52 %, I 36-37 %, D 77-80 %,
against 22-30 % everywhere under the old rule. Green keeps most of its mass
in the traces because its spots are the most coherent.

**Removed with it:** `background_pulsatile` (code, `PHASE_PRIORS`, and the
`BACKGROUND_PULSATILE: 0.5` line of the 12 cardio configs). Under the new
bin the background should hold the incoherent pulsatile mass, however much
that is, and a cap at a third would fight it on a noisy window. No
parameters change, so every checkpoint loads; a checkpoint whose saved
config names the removed term needs it stripped
(`runs/dev_cardio/exp13_local_tuning/visualise_oldbg.py` does that, and can
draw under either rule).

**Checks.** Ruff clean. One CPU training step of CardioConv at 64 px, batch
2, gate open and gate closed: every parameter has a gradient (the DDP
condition), as before; `beats` is detached, so the bin opens no path into
the rate network. The old-rule wrapper (`visualise_oldbg.py old`) reproduces
the control run's own shares exactly, so the control's masks can be drawn
faithfully; `visualise_oldbg.py new` draws any checkpoint under the current
rule. The trained answer is run 2.

## Run 2: both baselines under the new bin

`runs/dev_cardio/exp13_local_tuning/chain_v3.sh` trains `cardioconv_baseline`
and `cardiohydra_baseline` into `runs/local_tuning_bg_p33_<arm>/`,
evaluation and masks after each. (The control's process died at epoch 7
when its chain was stopped at 15:03: `TaskStop` does not stop a Git Bash
chain here, and the chain later launched a duplicate run; see the memory
note. The control's epoch 2/4/6 scores and masks stand.)

### CardioConv, new bin (15:37-18:39, 1.4 s a step, 8 GB)

| Epoch | ABP CCC | ABP r | CVP CCC | CVP r | ABP mean err | ABP sys err | G: ABP / CVP / bg share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.37 | 0.46 | 0.09 | 0.17 | -6.7 | -40.1 | 52 / 16 / 32 |
| 4 | 0.34 | 0.54 | 0.14 | 0.27 | -12.7 | -45.2 | 60 / 15 / 25 |
| 6 | 0.33 | 0.52 | 0.18 | 0.34 | -12.7 | -46.5 | 52 / 15 / 33 |
| 8 | 0.25 | 0.52 | 0.16 | 0.30 | -16.7 | -51.7 | 58 / 14 / 28 |
| **10** | **0.28** | **0.52** | **0.18** | **0.31** | -16.1 | -49.4 | 58 / 14 / 28 |
| control, old bin, 6 | 0.32 | 0.55 | 0.21 | 0.34 | -16.8 | -49.2 | 57 / 19 / 24 |

Training ABP CCC loss 0.43 -> 0.42 (flat), CVP 0.66 -> 0.75; rate waveform
term 0.33 -> 0.15; no non-finite steps. Background share at epoch 10 by
channel: G 28 %, R 49 %, I 44 %, B 61 %, D 74 %.

**Reading.**

- The waveforms match the control epoch for epoch to 6 (within 0.03), so
  the new bin cost nothing there and gained nothing either.
- The decay (epochs 8-10) did not lift the held-out score. ABP shape sat at
  r 0.52 from epoch 4 on; CCC fell to 0.25-0.28 because the predictions
  scaled down (mean error -13 -> -16 mmHg, systolic -46 -> -49). CVP r
  plateaued at 0.30-0.34 from epoch 6.
- Against experiment 11b at 72 px (ABP 0.47 / CVP 0.25 at epoch 10, r 0.66 /
  0.37), this recipe is behind on ABP shape by about 0.14 in r. Candidates,
  not separated here: the rate network at 50 px (its waveform term ends at
  0.15 against 0.05-0.08 at 36 px from 72 px frames: it sees upsampled
  128 px data), the 66 px mask grid, or the shorter run.
- The masks did what the bin was for: the background column holds the
  shirt, collar and speckle and the neck's structures stay in the traces;
  green's carotid band and infrared's high CVP streak are present from
  epoch 2 and unchanged by training; depth's one blob is ABP, the rest of
  depth background. The equilibrium set by epoch 2 barely moves after.
- Level tracking is nil as ever (`slope_within` 0.04 on ABP mean).

### CardioHydra, new bin (18:39-21:56, 1.6 s a step)

| Epoch | ABP CCC | ABP r | CVP CCC | CVP r | ABP mean err | ABP sys err | G: ABP / CVP / bg share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.56 | 0.72 | 0.28 | 0.49 | -15.6 | -27.1 | 49 / 17 / 34 |
| 4 | 0.68 | 0.85 | 0.40 | 0.61 | -14.3 | -25.3 | 49 / 17 / 34 |
| 6 | 0.65 | 0.86 | 0.38 | 0.65 | -15.4 | -27.2 | 49 / 15 / 36 |
| 8 | 0.62 | 0.89 | 0.38 | 0.62 | -17.9 | -30.0 | 49 / 16 / 35 |
| **10** | **0.64** | **0.90** | **0.40** | **0.60** | -18.6 | -26.0 | 49 / 16 / 35 |
| CardioConv, new bin, 10 | 0.28 | 0.52 | 0.18 | 0.31 | -16.1 | -49.4 | 58 / 14 / 28 |
| 11b CardioConv, 72 px, 30 | 0.51 | 0.70 | 0.31 | 0.49 | -13.7 | -38.9 | |
| 7b CardioHydra, 72 px, old design, 12 | 0.42 | | 0.39 | | | | |

Training ABP CCC loss 0.30 -> 0.18, still falling at epoch 10 (CardioConv's
sat at 0.42); rate waveform term 0.31 -> 0.17; no non-finite steps.

**Reading.**

- **The best P033 result of any cardio run, by a wide margin**: ABP r 0.90
  (median 0.91), CVP r 0.60-0.65, CVP CCC 0.40. The predicted heart rate
  matches the label on every measurement. Systolic error 26 mmHg against
  CardioConv's 49: the beats carry about two-thirds of the true amplitude.
- **Same front end, same masks, different predictor.** The mask shares are
  the same as CardioConv's under the new bin from epoch 2 and never move,
  so the whole difference is the state-space predictor reading those masks
  against the convolutional one. The model choice is settled for this
  recipe.
- **The decay did not settle the level.** ABP r rose through epoch 10 while
  the mean error drifted from -14 to -19 mmHg, so CCC peaked at epoch 4
  (0.68) and ended 0.64. The scale is set by the statistics head and the
  level is not training-limited (as every run has shown); shape is.
- **Masks**: identical structures to CardioConv's, green's carotid band and
  infrared's high CVP streak, background on the shirt and collar.

**Consequences for the tuning set.** With CardioHydra at this level on the
baseline recipe, the CardioConv arms (TV prior, lower rate) have little
left to decide; the arms worth running are CardioHydra's (Mamba rate scale
0.3, the TV prior), plus a `RATE_DOWNSCALE` 2 arm, since the rate network at
50 px ends with a waveform term of 0.17 against the 0.05-0.08 of the 72 px
runs and may be the ceiling on CVP. The 30-epoch LOSO sweep on the
CardioHydra baseline recipe is defensible as it stands.
