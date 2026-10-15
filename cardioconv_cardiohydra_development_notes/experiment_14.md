# Experiment 14 — Final local tuning for the 200 px HPC run (2026-10-14/15)

The last development session: 24 h on the dev card to settle the settings
the two cardio models ship with on the `hpc_configs_200` recipe (200 px,
rate 4 / wavelet 3, HoldCosine, Adam 1e-3, bf16, 30 epochs, batch 1 a card on
four A100s, global batch 4), where they are compared with the 200 px sweeps
of the other models. Plan agreed with the user before any run: the loss
weights of the two cardio YAMLs may be tuned (the other models' shared
interface is untouched), small code changes are allowed, and an arm ships
only if the masks still look good.

## Setup

- Local proxy of the cluster recipe, as experiment 13: the 128 px cache
  upsampled to 200, fp32, batch 1, P033 held out, 10 epochs an arm (~3 h
  with the held-out pass, scoring and panels). In optimiser steps the proxy
  is generous: 628 x 10 = 6,280 steps here against 157 x 30 = 4,710 on four
  cards at global batch 4; HoldCosine is relative to the run.
- Chain: `runs/dev_cardio/exp14_final_tuning/chain_arms.sh` takes arms from
  `arms.txt` in order (editable between arms), trains each into
  `runs/final_tuning_p<id>_<arm>/`, scores epochs 2, 6 and 10 on scratch
  copies (`score_folds.py`, `curve_*.csv`) and draws the masks at 6 and 10.
- **Precision, learned the hard way:** the first launch of arm 1 (21:58) ran
  the `configs/hpc_tuning_p33/` YAMLs as they are, which inherit
  `PRECISION: bfloat16` from the HPC recipe; the Turing card emulates bf16
  and the run crawled at 7.3 s a step (1.5 at float32). Stopped at 22:12 and
  restarted from float32 copies (`exp14_final_tuning/configs/<arm>.yaml`,
  `BASE` on the arm plus `TRAIN.PRECISION: float32`), as experiment 13 had
  done. Fifteen minutes lost.
- One code change: `MIXER_HIDDEN` and `MIXER_TAPS` on `CardioConvConfig`
  (defaults 16 / 15, the values the mixer had), passed through the builder
  to `ChannelMixer`. Checkpoints written before the keys existed carry a
  compiled config without them and no longer load; `patch_mixer_keys.py`
  adds the defaults to the experiment-13 CardioConv runs' `model.pt` and
  `config.yaml` files.

## Step 0: what the baselines and the bar look like

### CardioHydra baseline (experiment 13, run 2), held-out P033 by epoch

| Epoch | ABP CCC | ABP r | CVP CCC | CVP r | ABP mean err | ABP sys err | ABP dia err | CVP mean err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.56 | 0.72 | 0.28 | 0.49 | -15.6 | -27.2 | -12.2 | -2.4 |
| 4 | **0.68** | 0.85 | **0.40** | 0.61 | -14.3 | -25.3 | -9.5 | -2.0 |
| 6 | 0.65 | 0.86 | 0.38 | 0.65 | -15.4 | -27.2 | -7.4 | -2.5 |
| 8 | 0.62 | **0.89** | 0.38 | 0.62 | -17.9 | -30.0 | -8.6 | -2.4 |
| 10 | (see below) | | | | | | | |

The shape keeps improving (r 0.72 -> 0.89, the best any model has shown on
this participant) while the CCC falls from epoch 4: the predicted scale
shrinks (systolic -25 -> -30 while the diastolic error is -9), so the CCC
is held by the bias-correction factor (0.70 at epoch 8), not the shape.
That is what arm 1 is for.

### Rate gate (`gate_rate.py`, 200 training windows, one random window a recording)

| Checkpoint | pass (<= 8 bpm) | <= 10 bpm (buffer) | <= 3 bpm | error quantiles 50/75/90/95/99 % |
| --- | --- | --- | --- | --- |
| CardioHydra baseline, epoch 8 | 0.83 | 0.83 | 0.80 | 1.1 / 2.2 / 29 / 65 / 95 bpm |
| CardioConv baseline, epoch 10 | 0.80 | 0.80 | 0.77 | 1.0 / 2.4 / 29 / 72 / 96 bpm |

Four fifths of the training windows pass, and the failures are almost all
outside the wavelet buffer (outright misreads, not near misses): the gate
does not starve the downstream stages at this recipe, and
`RATE_TOLERANCE_BPM` is not an arm.

### The bar at 200 px: the sweeps' folds for P030-P039 at epoch 20 (`sweeps_p30s.csv`)

Held-out ABP CCC (CVP CCC in brackets), each sweep's last epoch:

| Participant | FactorizePhys | PhysFormer | RhythmFormer | PhysNet | iBVPNet |
| --- | --- | --- | --- | --- | --- |
| 30 | 0.31 (0.28) | 0.09 (0.34) | 0.03 (0.18) | 0.03 (0.04) | 0.02 (0.03) |
| 31 | 0.66 (0.11) | 0.10 (0.53) | 0.08 (0.64) | 0.03 (0.08) | 0.09 (0.07) |
| 32 | 0.00 (0.14) | 0.02 (0.06) | 0.05 (0.09) | 0.01 (0.01) | 0.04 (0.01) |
| **33** | **0.78 (0.43)** | 0.39 (0.43) | 0.05 (0.55) | 0.01 (0.02) | 0.04 (0.04) |
| 34 | 0.06 (0.14) | 0.08 (0.04) | 0.07 (0.17) | 0.01 (0.01) | 0.03 (0.02) |
| 35 | 0.00 (0.40) | 0.03 (0.32) | 0.09 (0.42) | 0.01 (0.02) | 0.02 (0.05) |
| 36 | 0.01 (0.22) | 0.27 (0.16) | 0.64 (0.18) | 0.03 (0.03) | 0.06 (0.03) |
| 37 | 0.73 (0.15) | 0.04 (0.19) | 0.07 (0.22) | 0.01 (0.04) | 0.05 (0.02) |
| 38 | 0.66 (0.13) | 0.03 (0.09) | 0.05 (0.05) | 0.01 (0.03) | 0.06 (0.03) |
| 39 | 0.71 (0.27) | 0.73 (0.19) | 0.05 (0.24) | 0.02 (0.03) | 0.07 (0.02) |

On P033 at 200 px FactorizePhys scores ABP CCC 0.78 (r 0.81, mean error
-3.3, systolic -13.6) and CVP 0.43; PhysFormer r 0.80 but CCC 0.39 (level
-17); RhythmFormer CVP 0.55. PhysNet and iBVPNet predict nothing on any of
these ten participants at 200 px (PhysNet scored 0.72 on P033 at 72 px), and
RhythmFormer's ABP works on P036 only. The CardioHydra baseline's epoch-4
scores (0.68 / 0.40) are second to FactorizePhys on ABP and level with it
on CVP, with shape (r 0.85-0.89) above every model's.

**Confirmation participant:** P035, on whom every model's ABP CCC is at
most 0.09 while three models still find CVP shape (0.32-0.42), so the
traces are usable; P034 is the alternative (every model under 0.17 on
both). P032 scores zero on everything and is left alone.

## Arms

### Arm 1: CardioHydra, ABP soft-max / soft-min weights 0.05 -> 0.2 (`cardiohydra_abpamp`), 22:13-00:11, stopped after epoch 6

| Epoch | ABP CCC / r | baseline ABP CCC / r | CVP CCC / r | baseline CVP | ABP mean / sys / dia err | train ABP CCC loss (baseline) |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.41 / 0.47 | 0.56 / 0.72 | 0.35 / 0.54 | 0.28 / 0.49 | -3.7 / -28.4 / +5.3 | 0.36 (0.26) |
| 4 | 0.42 / 0.51 | 0.68 / 0.85 | 0.42 / 0.57 | 0.40 / 0.61 | -6.1 / -30.1 / +3.4 | 0.35 (0.24) |
| 6 | 0.45 / 0.54 | 0.65 / 0.86 | 0.39 / 0.59 | 0.38 / 0.65 | -8.3 / -28.1 / -0.8 | 0.36 (0.22) |

**Verdict: not shipped; experiment 6b's answer again.** The heavier terms
move the ABP waveform up (mean error -16 -> -8, diastolic from -12 to -1)
and do not widen it (systolic stays at -28 to -30); the pulse pressure is
not what the terms buy. They cost shape: the training ABP CCC loss sits at
0.36 for six epochs where the baseline's fell to 0.22, and held-out r is
0.54 against 0.86 at epoch 6. On the window panels the predicted ABP has
the right level but a jagged, double-peaked beat. CVP is unchanged. The
masks are the baseline's to the percent (G: ABP 50 % / CVP 16 % /
background 34 %, D 80 % background), so the loss change reached the
waveform path only. Stopped at 00:11 to give the hour to the next arm.

What this says about the level problem: the statistics head's spread, not
its mean, is what is short, and a window-statistics weight pulls the mean
rather than the spread. Any further level work goes to the head's spread
path (its input or its initial scale), not to the loss weights.

### Arm 2: CardioHydra, `MAMBA_LR_SCALE` 0.1 -> 0.3 (`cardiohydra_mamba0.3`), 00:12-02:09, stopped after epoch 6

| Epoch | ABP CCC / r | baseline | CVP CCC / r | baseline | ABP mean / sys / dia err | train ABP CCC loss (baseline) |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.41 / 0.64 | 0.56 / 0.72 | 0.27 / 0.47 | 0.28 / 0.49 | -19.2 / -37.9 / -9.5 | 0.28 (0.26) |
| 4 | 0.59 / 0.80 | 0.68 / 0.85 | 0.38 / 0.56 | 0.40 / 0.61 | -16.0 / -31.0 / -8.5 | 0.26 (0.24) |
| 6 | 0.60 / 0.81 | 0.65 / 0.86 | 0.29 / 0.43 | 0.38 / 0.65 | -16.0 / -31.1 / -7.5 | 0.24 (0.22) |

**Verdict: not shipped; 0.1 stands.** The tripled trunk rate did not make
the model learn faster (training losses within 0.02 of the baseline's at
every epoch) and left every held-out score below the baseline's at epochs
2, 4 and 6, CVP most (0.29 / 0.43 against 0.38 / 0.65 at epoch 6). The
level errors are the baseline's. Masks identical to the baseline's shares
(G: 50 / 16 / 34 %, D 80 % background). Stopped at 02:09.

### Arm 3: CardioConv, mixer 15 taps x 16 hidden -> 31 x 32 (`cardioconv_mixer31x32`), 02:11-04:05, stopped after epoch 6

| Epoch | ABP CCC / r | baseline | CVP CCC / r | baseline | ABP mean / sys err | train ABP CCC loss (baseline) |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.36 / 0.46 | 0.37 / 0.46 | 0.09 / 0.17 | 0.09 / 0.17 | -9.7 / -40.4 | 0.44 (0.45) |
| 4 | 0.36 / 0.48 | 0.34 / 0.54 | 0.12 / 0.26 | 0.14 / 0.27 | -12.2 / -42.1 | 0.40 (0.41) |
| 6 | 0.38 / 0.51 | 0.33 / 0.52 | 0.16 / 0.29 | 0.18 / 0.34 | -10.7 / -41.4 | 0.38 (0.40) |

**Verdict: not shipped; 15 x 16 stands (the defaults of the new keys).**
Doubling the mixer's reach to a full beat and its width changed nothing
the held-out fold can see: every score within 0.05 of the baseline's at
every epoch, the training ABP CCC loss 0.38 against 0.40 at epoch 6. The
mixer's capacity is not what holds CardioConv at r 0.5 on this recipe; the
candidates it is given (masked means of the intensity and the kinematics,
one per channel and trace) are. The next arm on CardioConv therefore
widens its inputs, not its mixer: `TRACE_INDEPENDENT: false`, every trace
mixed from every candidate.

### Arm 4: CardioHydra, `PULSATILITY_TV` 0.005 -> 0.02 (`cardiohydra_tv0.02`), 04:04-06:00, stopped after epoch 6

| Epoch | ABP CCC / r | baseline | CVP CCC / r | baseline | ABP mean / sys / dia err | train ABP CCC loss (baseline) |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.49 / 0.71 | 0.56 / 0.72 | 0.27 / 0.42 | 0.28 / 0.49 | -17.6 / -33.2 / - | 0.28 (0.26) |
| 4 | 0.63 / 0.82 | 0.68 / 0.85 | 0.33 / 0.50 | 0.40 / 0.61 | -15.7 / -25.3 / - | 0.23 (0.24) |
| 6 | 0.53 / 0.83 | 0.65 / 0.86 | 0.40 / 0.54 | 0.38 / 0.65 | -19.8 / -35.2 / -10.9 | - |

Masks at epoch 6, mean over the six held-out windows, against the
baseline's epoch 6: pulsatility effective pixels up by 4-8 % in every
channel (G 26,700 against 24,700), the trace shares within 0.05 (G: ABP
46 / CVP 16 / background 38 % against 49 / 15 / 36 %). The panels show the
same bands, a little softer.

**Verdict: not shipped; 0.005 stands.** Four times the TV prior smooths the
pulsatility map slightly and buys nothing: the map was already on the neck
at 0.005 on the 66 px grid, and the scores are the baseline's or below
(ABP CCC 0.53 against 0.65 at epoch 6, CVP r 0.54 against 0.65). Stopped
at 06:00.

### Arm 5: CardioConv, `TRACE_INDEPENDENT` true -> false (`cardioconv_dense`), 06:00-09:01, the full 10 epochs

| Epoch | ABP CCC / r | baseline | CVP CCC / r | baseline | ABP mean / sys err | train ABP CCC loss (baseline) |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.36 / 0.47 | 0.37 / 0.46 | 0.18 / 0.29 | 0.09 / 0.17 | -9.2 / -41.0 | 0.45 (0.45) |
| 4 | 0.39 / 0.54 | 0.34 / 0.54 | 0.16 / 0.27 | 0.14 / 0.27 | -10.3 / -41.7 | 0.43 (0.41) |
| 6 | 0.38 / 0.59 | 0.33 / 0.52 | 0.23 / 0.39 | 0.18 / 0.34 | -13.3 / -44.3 | 0.40 (0.40) |
| 8 | 0.35 / 0.56 | 0.25 / 0.52 | 0.21 / 0.35 | 0.16 / 0.30 | -14.9 / -44.7 | 0.39 (0.42) |
| 10 | **0.38 / 0.58** | 0.28 / 0.52 | **0.22 / 0.37** | 0.18 / 0.31 | -14.4 / -44.1 | 0.39 (0.43) |

Masks at epoch 6 against the CardioConv baseline's: pulsatility a little
sharper (G 18,500 against 21,900 effective pixels), trace shares up by
0.02-0.03, the same bands. No non-finite steps.

**Verdict: shipped.** `TRACE_INDEPENDENT: false` in the CardioConv
`hpc_configs_200` YAML. With every trace mixed from every channel's
candidates for both traces, CardioConv leads its baseline on both traces
from epoch 6 on, by 0.04-0.07 in r and 0.04-0.10 in CCC, at the same level
errors. The one CardioConv arm that moved anything, and the direction it
points: the mixer's inputs, not its capacity, limit CardioConv on this
recipe. One participant, 33 measurements: a lead of this size is
suggestive, not settled; the P035 confirmation is the check.

### Arm 6: CardioHydra, `LR` 1e-3 -> 5e-4 (`cardiohydra_lr5e-4`), 09:02-10:58, stopped after epoch 6

| Epoch | ABP CCC / r | baseline | CVP CCC / r | baseline | ABP mean / sys err | train ABP CCC loss (baseline) |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.49 / 0.69 | 0.56 / 0.72 | 0.29 / 0.45 | 0.28 / 0.49 | -15.8 / -30.9 | 0.28 (0.26) |
| 4 | 0.57 / 0.77 | 0.68 / 0.85 | 0.27 / 0.44 | 0.40 / 0.61 | -16.1 / -29.2 | 0.25 (0.24) |
| 6 | 0.52 / 0.81 | 0.65 / 0.86 | 0.34 / 0.53 | 0.38 / 0.65 | - | - |

**Verdict: not shipped; 1e-3 stands.** At half the rate the training loss
falls at the baseline's pace but the held-out shape lags it at every
epoch (ABP r 0.77 against 0.85 at epoch 4, CVP r 0.53 against 0.65 at
epoch 6), and the CCC slide after epoch 4 is there too (0.57 -> 0.52), so
the slide is not the rate's doing. Masks the baseline's. Stopped at 10:58.

## What ships, and what the day said

| Arm | Model | Change | Epoch-6 ABP CCC / r | Epoch-6 CVP CCC / r | Verdict |
| --- | --- | --- | --- | --- | --- |
| baseline | CardioHydra | none | 0.65 / 0.86 | 0.38 / 0.65 | **ships** |
| 1 | CardioHydra | ABP max/min weights 0.2 | 0.45 / 0.54 | 0.39 / 0.59 | level up, pulse not wider, shape lost |
| 2 | CardioHydra | Mamba trunks 0.3x | 0.60 / 0.81 | 0.29 / 0.43 | no faster learning, lower scores |
| 4 | CardioHydra | pulsatility TV 0.02 | 0.53 / 0.83 | 0.40 / 0.54 | map softer, not cleaner |
| 6 | CardioHydra | LR 5e-4 | 0.52 / 0.81 | 0.34 / 0.53 | lags held-out shape at every epoch |
| baseline | CardioConv | none | 0.33 / 0.52 | 0.18 / 0.34 | reference |
| 3 | CardioConv | mixer 31 taps x 32 hidden | 0.38 / 0.51 | 0.16 / 0.29 | capacity is not the limit |
| 5 | CardioConv | `TRACE_INDEPENDENT` false | 0.38 / 0.59 | 0.23 / 0.39 | **ships** (0.38 / 0.58, 0.22 / 0.37 at epoch 10) |

- **CardioHydra ships as it stood** (`hpc_configs_200/cardiohydra_*.yaml`
  unchanged but for comments): Adam 1e-3, HoldCosine, Mamba trunks 0.1x,
  TV 0.005, the shared loss weights. Every lever tried scored below it.
  Its best P033 reading, ABP CCC 0.68 / r 0.85 and CVP 0.40 / 0.61 at epoch
  4, is second to FactorizePhys's 0.78 / 0.43 on this fold at 200 px and
  above every other sweep model, most of which predict nothing on P030-P039
  at 200 px. Its ABP shape (r 0.89 by epoch 8) is the best any model has
  shown on P033; what holds the CCC is scale.
- **CardioConv ships with `TRACE_INDEPENDENT: false`.** The mixer's inputs,
  not its width, limit it; even so it stays far behind CardioHydra (r 0.58
  against 0.89) and is the weaker of the two on this recipe.
- **The level problem is the statistics head's spread.** Heavier window
  statistics lift the mean and cost shape (arm 1, experiment 6b); the
  pulse pressure stays 20 mmHg short in every run. Any further level work
  goes to the spread path (its inputs or initial scale), not to the loss.
- **30 epochs on the cluster are not too many for CardioHydra.** In
  optimiser steps the cluster's 30 epochs at global batch 4 (4,710) are
  fewer than the local 10 at batch 1 (6,280), and HoldCosine's tail sits in
  the last third of either. The slide in CCC after epoch 4 here is slow
  (0.68 -> 0.62 by epoch 8 while r rises) and the lower rate did not stop it.
- **Transfer caveats, unchanged:** bf16, DDP across four cards and the
  native 200 px detail were not exercised here; the `salloc` smoke in the
  SLURM script's header is the check before the LOSO job.

## Confirmation on P035 (the shipped recipes on a second held-out participant)

P035 was chosen from `sweeps_p30s.csv` as a participant with usable traces
on whom every 200 px sweep model fails on ABP (CCC at most 0.09) while
three still find CVP shape (0.32-0.42). Same chain, same local proxy, 10
epochs, 629 training windows, 240 held-out windows (40 measurements).

### CardioHydra, the recipe as it stands (`final_tuning_p35_cardiohydra_baseline`, 10:59-14:10, exit 0, no non-finite steps)

| Epoch | ABP CCC / r | ABP MAE | CVP CCC / r | CVP MAE | ABP mean / sys / dia err | CVP mean err | HR err |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.68 / 0.79 | 12.0 | 0.38 / 0.57 | 1.7 | +9.4 / +9.5 / +6.8 | +1.2 | -0.1 |
| 4 | 0.76 / 0.85 | 10.3 | 0.41 / 0.61 | 1.7 | +8.1 / +13.4 / +7.7 | +1.4 | -0.1 |
| 6 | 0.85 / 0.90 | 7.7 | 0.40 / 0.61 | 1.8 | +4.8 / +5.0 / +0.1 | +1.3 | 0.0 |
| 8 | 0.83 / 0.90 | 8.5 | 0.40 / 0.61 | 1.9 | +6.3 / +9.2 / +0.1 | +1.2 | +0.1 |
| **10** | **0.86 / 0.93** | **7.3** | 0.40 / 0.61 | 1.9 | +6.2 / +8.8 / +2.8 | +1.1 | +0.1 |

| P035, 200 px, each sweep's epoch 20 | ABP CCC / r | CVP CCC / r | ABP mean / sys err |
| --- | --- | --- | --- |
| FactorizePhys | 0.01 / 0.01 | 0.40 / 0.65 | +18.6 / +20.7 |
| RhythmFormer | 0.09 / 0.21 | 0.42 / 0.66 | +5.0 / -15.8 |
| PhysFormer | 0.03 / 0.11 | 0.32 / 0.75 | +11.1 / -10.0 |
| PhysNet | 0.01 / 0.01 | 0.03 / 0.07 | |
| iBVPNet | 0.03 / 0.03 | 0.05 / 0.12 | |

**The recipe confirms, and then some.** On the participant where every
sweep model reads no arterial waveform, CardioHydra reaches ABP CCC 0.86 /
r 0.93 with an MAE of 7.3 mmHg, the best ABP reading of any model on any
fold in this work, at the right heart rate (0.1 bpm) and with the level
within 6 mmHg (high this time: P035 sits below the training population
where P033 sat above it, the constant-level problem seen from the other
side). CVP shape is level with the best sweep models (r 0.61 against
0.65-0.75) at a better level (+1.1 against +5 to +19 on ABP's scale).
Unlike P033 the CCC does not slide late: it rises to the last epoch under
the cosine tail, so 30 epochs on the cluster are not a worry on this fold
either. Masks at epoch 6 (`epoch_06/visuals/neckflix/`): the pulsatility
map on the neck in every channel, ABP on a diagonal band across it in
colour, CVP on spots at the band's ends in infrared, depth one bright blob
for ABP; shares G 39 / 14 / 48 %, D 85 % background.

Why the sweep models fail here and CardioHydra does not is not separated
by this run; the rate gate and the wavelet bank at the estimated rate are
the design's answer to a participant whose pulse the generic networks do
not lock onto, and the heart-rate error of 0.1 bpm says the rate network
locks on.

### CardioConv, the shipped recipe, `TRACE_INDEPENDENT` false (`final_tuning_p35_cardioconv_dense`, 14:11-17:17, exit 0, no non-finite steps)

| Epoch | ABP CCC / r | ABP MAE | CVP CCC / r | CVP MAE | ABP mean / sys err | CVP mean err | HR err |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.22 / 0.35 | 19.4 | 0.25 / 0.34 | 1.7 | +14.4 / -8.5 | +0.7 | -0.3 |
| 4 | 0.34 / 0.45 | 15.7 | 0.27 / 0.37 | 1.7 | +9.2 / -11.7 | +0.9 | 0.0 |
| 6 | 0.36 / 0.47 | 15.5 | 0.30 / 0.43 | 1.7 | +9.4 / -11.6 | +1.2 | +0.2 |
| 8 | 0.37 / 0.52 | 16.6 | 0.31 / 0.44 | 1.6 | +12.3 / -7.5 | +1.1 | 0.0 |
| **10** | **0.41 / 0.54** | 15.2 | **0.33 / 0.45** | 1.6 | +10.2 / -9.4 | +0.9 | +0.1 |

CardioConv on P035 reads as it did on P033: ABP r climbs slowly to 0.54
and CVP to 0.45, above every sweep model's ABP on this participant (0.09)
and above its own P033 numbers (0.58 r at epoch 10 there, but CCC 0.38
against 0.41 here), far under CardioHydra's 0.93 / 0.86 from the same
front end. Masks at epoch 6 are CardioHydra's to within a few percent of
share (G 39 / 15 / 46 %, D 86 % background): the shared front end lands in
the same place for both models, and the waveform path is where they part.
The rate is right (0.1 bpm).

**Both shipped recipes hold on the second fold.** The gap between the
models is the design, not the tuning: a linear mixer over masked means
cannot do what the Mamba trunk does with the per-frame power, phase and
kinematic features, on either participant.
