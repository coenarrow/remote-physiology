# Experiment 15 — Visualising the kinematics (2026-10-16)

The user asked how the kinematic features could be seen. The models compute
them from the wavelet field (`_shared_modules/kinematics.py`), the
statistics head reads the pooled speed as its level cue (the Bramwell-Hill
argument of `cardio_development.md`), and until today nobody had looked at
them on a real neck.

## What is computed

Per pixel, frame and harmonic, on the wavelet grid, from the harmonic's
scale-blended complex field W: amplitude |W|; the wavevector k = -grad(phase)
and its magnitude (radians per input pixel); the phase speed omega / |k|
(signed, input pixels per frame); the unit heading k / |k|. The models see
log1p amplitude, |k|, sign * log1p |speed| and the heading per frame, and
the statistics head sees log1p |speed| pooled under each trace's mask per
channel and harmonic.

## What was added (`scripts/visualise.py`, `cardioconv.capture`)

- `capture()` keeps the coarse (field-grid) masks, the wavenumber and the
  fundamental's blended complex field beside what it kept; `Kinematics`
  returns that field as `harmonics`.
- `kinematics.png` per window: one row per channel, the fundamental's log
  amplitude, wavelength 2 pi / |k| (log, 3-1000 px), signed speed (symlog,
  +-100 px/frame) with the trace masks' contours, and the heading as
  arrows (block resultant times pulsatile weight); every map over the
  channel's mean frame with the pulsatility map as alpha, on fixed scales
  so windows compare. Below: the mask-pooled signed speed per trace over
  the frames, one line per channel, against the label waveform.
- `masks.gif` gains a second strip: the fundamental's wrapped phase on the
  field's grid (cyclic colours, pulsatility as alpha).
- `scores.csv` gains, per channel and trace: the pooled speed the head
  reads (px/frame), the mask-weighted median wavelength, the heading's
  circular mean and resultant length; and the window's label ABP / CVP mean
  and 95th percentile.
- `--windows 0` runs every window, `--no-images` writes scores only, and
  with more than a dozen windows `speed_vs_pressure.png` scatters each
  channel's pooled ABP speed against the window's label ABP mean.

## What it shows on P035 (CardioHydra as shipped, epoch 10, 240 windows)

| Channel | pooled ABP speed, median (SD) px/frame | median wavelength px | heading resultant, median |
| --- | --- | --- | --- |
| R | 2.00 (0.49) | 42 | 0.03 |
| G | 2.15 (0.50) | 46 | 0.04 |
| B | 1.92 (0.27) | 41 | 0.03 |
| I | 1.81 (0.51) | 39 | 0.04 |
| D | 0.77 (1.01) | (absent in half the recordings) | 0.00 |

- **There is no travelling wave at this resolution.** The heading's
  resultant length under the masks is 0.02-0.12 in every channel and
  window: the per-pixel wave directions are random. The arrow panels draw
  dots. The median wavelength is 40-60 input pixels, a quarter of the
  frame, where a pulse at metres per second over a 1 Hz beat would have a
  wavelength of metres: |k| is set by noise and the vessel's geometry, not
  by propagation. In the phase animation the pulsatile patch changes colour
  in step; nothing sweeps along the vessel.
- **The pooled speed is a smooth positive number, about 2 px/frame, nearly
  the same for ABP and CVP and across colour channels**, and over the
  window's frames it sits flat with occasional blips rather than modulating
  within the beat (the time traces). As omega / |k| with omega fixed by the
  rate and |k| a noise floor, it behaves as an inverse of the local phase
  roughness, i.e. a signal-quality measure, not a wave speed.
- **It does correlate with the window's pressure, but between recordings,
  not within them.** Over the 240 windows r is 0.59 (R), 0.71 (G), 0.47
  (B), 0.50 (I) against the label ABP mean. P035's six recordings are three
  postures; ABP falls from 73 mmHg supine to 68 sitting and the speed from
  2.8-3.3 to 1.9-2.1, so the between-recording r is 0.55-0.78 while the
  within-recording r (label and speed centred per recording) is -0.07 to
  0.12. Posture changes the view geometry and the pulsatile region's shape
  with the pressure, so this is a confound as much as a cue, and one
  participant cannot separate the two. Across participants the level error
  is a constant per participant (experiment 14), so the head does not read
  it either way.

**Reading.** The kinematics stage is a physically motivated feature that,
at 66 px of real detail and 30 fps, does not measure what it was named
for. The level cue it feeds the statistics head is a mask-quality number
that happens to move with posture. Nothing changes in the models today;
this is what any further work on level has to start from, and the
visualiser now shows it per window. The all-window P035 scores are kept at
`runs/dev_cardio/exp14_final_tuning/p35_kinematics_scores_all_windows.csv`.
