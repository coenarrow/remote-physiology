# Experiment 16 — Coherent wavelet pooling for CardioConv (2026-10-18)

A two-hour test on the dev box, P033 held out, of one change to CardioConv's
readout, after the cluster check (experiment 14's addendum, the
`Neckflix_CardioCheck_H200_4GPU` job of 2026-10-18) showed CardioConv's
training ABP CCC term stuck at 0.62-0.65 for 25 epochs on the native 200 px
cache while CardioHydra's fell from 0.41 to 0.25 on the same masks.
CardioConv did not fit the ABP shape even on the training participants;
its masks matched CardioHydra's; the limit was the readout.

## The change: `COHERENT_POOLING`

CardioConv's intensity candidate is the masked mean of the raw pixel
intensities under `pulsatility x trace mask`. The mask is a phase bin, so the
pixels inside it pulse at phases spread across the bin and a plain mean
partly cancels them, worst at the harmonics that carry the beat's shape.

`coherent_candidate` (`neural_methods/model/cardioconv/channel_predictor.py`)
pools in the wavelet domain instead. Per pixel, harmonic ``n`` of the
blended complex field (``Kinematics.harmonics``) is rotated back by ``n``
times the pixel's window-mean relative phase at the fundamental (relative to
the channel's pulsatility-weighted pulse, the phase masker's reference; a
delay shifts harmonic ``n`` by ``n`` times the fundamental's shift). The
rotated harmonics are averaged under the same mask weight and the real
parts summed: the band-passed pulse with every pixel in step. In-phase
signal adds, noise does not. Nothing is learned; the mask weight carries the
gradient as before; the mixer, kinematic candidates and statistics head are
untouched. `COHERENT_POOLING: true` on `CardioConvConfig` switches it on;
every shipped YAML has it `false`.

## Run

`runs/final_tuning_p33_cardioconv_coherent`: the shipped CardioConv recipe
(`cardioconv_dense`, `TRACE_INDEPENDENT` false) in float32, batch 1, 128 px
cache grown to 200, 10-epoch HoldCosine schedule, stopped after epoch 6 at
the two-hour mark (18:15-20:06). 1.41 s a step, the same as the dense
baseline. Scored with `score_folds.py` at epochs 2, 4 and 6; references are
the dev-box dense baseline on the same fold and schedule
(`final_tuning_p33_cardioconv_dense`) and the dev-box CardioHydra baseline
(experiment 14's table).

### Training ABP CCC term (1 - CCC)

| epoch | coherent | dense baseline | CardioHydra |
| --- | --- | --- | --- |
| 1 | 0.32 | - | - |
| 2 | 0.29 | 0.45 | 0.26 |
| 4 | 0.28 | 0.41 | 0.24 |
| 6 | 0.27 | 0.40 | - |

The dense baseline's floor on the cluster was 0.62; the coherent arm is
below it from the first epoch.

### Held out, P033, 33 paired 10 s measurements

| epoch | arm | ABP CCC / r | CVP CCC / r | ABP mean err | systolic / diastolic err |
| --- | --- | --- | --- | --- | --- |
| 2 | **coherent** | **0.63 / 0.76** | **0.34 / 0.49** | -9.7 | -33 / 0 |
| 2 | dense baseline | 0.36 / 0.47 | 0.18 / 0.29 | -9.2 | -41 / +7 |
| 2 | CardioHydra | 0.56 / 0.72 | 0.28 / 0.49 | -17.6 | -33 / - |
| 4 | **coherent** | **0.64 / 0.75** | 0.31 / 0.50 | -11.6 | -29 / -7 |
| 4 | dense baseline | 0.39 / 0.54 | 0.16 / 0.27 | -10.3 | -42 / +6 |
| 4 | CardioHydra | 0.68 / 0.85 | 0.40 / 0.61 | -15.7 | -25 / - |
| 6 | **coherent** | 0.59 / 0.78 | **0.35 / 0.51** | -15.0 | -35 / -9 |
| 6 | dense baseline | 0.38 / 0.59 | 0.23 / 0.39 | -13.3 | -44 / +2 |
| 6 | CardioHydra | 0.65 / 0.86 | 0.38 / 0.65 | -19.8 | -35 / -11 |

### Masks and trace, epoch 6, window `P033_S01_R1_0_D_1_0`

The same masks as the dense baseline and CardioHydra on this window:
carotid band in R, G and I (G: ABP 53 %, CVP 19 %, background 28 %), the
venous patch above and medial to it. The predicted ABP is a clean pulse at
the right rate with a swing of about 55 mmHg against the label's 75, where
the dense baseline's swung 25 with beat-to-beat wander. Its shape is close
to sinusoidal: the sharp systolic upstroke and the dicrotic notch are
softened, which is the remaining systolic error. The CVP prediction is a
single-peaked pulse against a label with distinct a and v waves.

## Reading

- **Coherent pooling finds information the masked mean threw away.** On the
  fold where the dense baseline's ABP CCC sat at 0.35-0.39 for ten epochs,
  the coherent arm reads 0.59-0.64 and r 0.75-0.78, with the training
  shape term below anything the dense readout reached on either machine.
  CVP doubles, 0.18 to 0.34 at epoch 2. The masks are the same; the readout
  was the limit, as experiment 14 concluded, and this is the readout.
- **It closes most of the gap to CardioHydra, not all of it.** At epoch 2
  the coherent arm is ahead of CardioHydra on every score; by epoch 4 to 6
  CardioHydra leads on ABP r (0.85 against 0.75) and CVP, and the coherent
  arm's ABP CCC eases from 0.64 to 0.59 as its level error grows. What the
  trunks add on top of coherent pooling is beat shape: the pooled waveform
  is fundamental-dominated, and the higher harmonics, which the model
  already computes, are averaged with the same weight as the fundamental
  rather than weighted by their own coherence.
- **One fold, three scored epochs.** A lead of 0.2 to 0.3 in CCC over the
  baseline on 33 measurements is not a close call, but the P035 and P036
  folds, where CardioConv's baseline read 0.41 and the cluster's 0.23, are
  the confirmation, and the native 200 px cache is where it has to hold.

## Next

1. Confirm on P035 and P036 on the dev box, or directly on the cluster on
   the 200 px cache with the check script: `COHERENT_POOLING: true` is one
   line in the `hpc_tuning_p33` arm YAML.
2. Weight the pooled harmonics by their coherence under the mask before
   the sum (the phase masker's scale and temporal coherence features are
   per harmonic already), or hand the mixer the pooled harmonics as
   separate candidates and let it weight them: the mixer's inputs again,
   not its depth.
3. If it holds, decide whether CardioConv ships with it. It keeps
   CardioConv's property of a fixed, physical readout with no learned
   trunk, so the ablation against CardioHydra stays clean and becomes
   stronger.

Uncommitted at the time of writing: `channel_predictor.py`,
`cardioconv.py`, `src/models.py`, `src/model_config.py`, the six CardioConv
YAMLs and the template, beside the day's resize change
(`src/frame_transforms.py`, experiment 9's postscript).
