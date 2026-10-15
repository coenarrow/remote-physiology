# CardioConv and CardioHydra

CardioConv and CardioHydra are the two models developed in this work for
estimating **absolute** arterial and central venous pressure waveforms (ABP and
CVP, in mmHg) from video of the neck. The models adapted from the rPPG-Toolbox
are generic spatiotemporal networks that learn a mapping from pixels to a
waveform. These two are different: they are built around the physics of the
signal they are looking for. The design rests on four observations:

1. **Every cardiac pulsation in the neck beats at the heart rate.** Arterial
   and venous pulses share a fundamental frequency and its harmonics, so once
   the heart rate is known, the only part of the video worth analysing is the
   narrow band around those frequencies.
2. **Arterial and venous pulses differ in phase.** The carotid artery and the
   internal jugular vein pulse at the same rate but at different points in
   the cardiac cycle. Phase, not amplitude or location, is what separates them.
3. **A pulse is a travelling wave.** Its speed of propagation is linked to
   vessel wall stiffness, and through it to pressure (the Moens–Korteweg and
   Bramwell–Hill relations). The way the pulse moves across the skin is
   therefore a possible cue to pressure *level*, not only to waveform shape.
4. **Shape and level are separate problems.** The pixel intensities that carry
   the waveform's shape carry no calibrated information about its level in
   mmHg. The models predict the two separately and combine them only at the
   output.

Both models share one front end, which implements points 1–3, and one output
stage, which implements point 4. They differ only in the module in between,
which turns the masked video into a waveform.

## Overview

Each model takes a multi-channel clip, `(B, C, T, H, W)`, and returns one
waveform per trace, `(B, S, T)`. Here `C` is the video channels (on Neckflix,
R, G, B, infrared and depth, stacked) and `S` is the traces predicted (ABP and
CVP). Each video channel goes through the front end separately, using the
same weights, so colour, infrared and depth each produce their own masks and
features.

```text
 video ──► Rate Estimator ──► heart rate f_HR
   │                              │
   └──────────► Wavelet Transform at {f_HR, 2 f_HR, 3 f_HR} ──► complex field W
                                  │
          ┌───────────────────────┼─────────────────────────┐
          ▼                       ▼                         ▼
  Pulsatility Mask ──► Phase Masks (S traces + bg)     Kinematics
          │                       │                         │
          └──────────── masked pooling ─────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
     CardioConv: temporal conv mixer     CardioHydra: bidirectional Mamba trunk
                └─────────────────┬─────────────────┘
                                  ▼
                       shape-only waveform per trace
                                  │
                                  ▼
            Statistics Head (level + spread) ──► waveform in mmHg
```

The shared stages are:

- Rate estimator
- Wavelet transform
- Pulsatility masker
- Phase masker
- Kinematics
- Statistics head

Both models are *multi-trace* by design. One network predicts every trace
together, because the traces **compete for pixels**: the phase masker
assigns each pixel to the traces through a single softmax. Two independent
single-trace copies could not do this.

## Rate Estimator

The arterial and venous pulses both appear in the video at the heart rate and
its harmonics. Knowing the heart rate tells the rest of the model which
frequencies to analyse, so the estimate acts as a prior that steers the model
towards pulsatile content. The idea is similar to Amelard et al., who
recovered the jugular venous pulse by correlating each pixel with a
synchronous reference pulse signal. In our case the reference is estimated
from the video, not measured with a contact sensor.

**Architecture.** The rate estimator is one copy of a single-trace rPPG
network. The current default is RhythmFormer (the `RATE_ESTIMATOR` key in the
config), chosen after an ablation that compared candidate backbones on held-out
participants. It reads the clip spatially downsampled by area averaging
(`RATE_DOWNSCALE`), because it only needs the rate. It predicts a single pulse
waveform. The heart rate is read from the peak of that waveform's power
spectrum, made differentiable as a sharpened spectral centroid: the power on a
1 bpm grid (30–200 bpm) is raised to a high power before its centroid is
taken. This gives the largest peak, interpolated between bins.

**Training target.** The model does not see the ground-truth rate at test
time. During training the estimator is supervised with a **fused cardiac
reference**. The available cardiac traces of the window (ABP, CVP and ECG
where present) are combined into one spectrum. Its largest bin is the target
rate, read exactly as the evaluation reads a rate. The traces are also fused
into one waveform that beats at that rate. The estimator's loss is
$1 - r$, where $r$ is the Pearson correlation between its predicted
waveform and the fused waveform, maximised over a short range of lags. The lag
search allows for the pulse transit time between the reference traces and the
neck.

**Training gate.** The estimator is trained from scratch, together with the
rest of the model. It is not pre-trained or frozen. Instead, training is gated
per sample: any sample whose estimated rate is more than a tolerance (8 bpm)
from the target skips every later stage, and its output is masked out of the
loss. Early in training almost no sample passes the gate, so the downstream
modules only start learning once the rate estimator is reliable. The reason is
simple: **if the rate is wrong, the wavelet field is taken at the wrong
frequencies, and everything downstream is learning from noise.** The tolerance
sits inside the wavelet bank's frequency buffer (below), so a sample that passes
is guaranteed to have its true fundamental inside the bank. At test time no
label reaches the model and every sample runs.

An alternative estimator, `Spectral`, fuses attention-pooled per-channel
spectra directly instead of predicting a waveform. It is kept as an option
but is not the default.

## Wavelet Transform

A wavelet transform analyses a signal with short oscillating kernels
(wavelets) stretched to different scales. Unlike a Fourier transform, it
localises content in both frequency (scale) and time (position). With a
complex wavelet, the output at each pixel, frame and scale is a complex
number, $W = |W| e^{i\phi}$. Its magnitude gives the local strength of the
oscillation at that frequency, and its argument gives its phase.

We use the complex Morlet wavelet,

$$
\psi(t) = (\pi B)^{-1/2}\, e^{2\pi i C t}\, e^{-t^2/B},
$$

with bandwidth $B = 1.5$ and centre frequency $C = 1$ (`cmor1.5-1.0`).

**Conditioned on the heart rate.** Transforming every pixel over every
possible scale would be expensive, and most of the result would be noise. The
transform is therefore evaluated only at a small bank of scales around the
estimated heart rate and its harmonics:

- the fundamental, $f_{HR}$, and the next two harmonics, $2f_{HR}$ and
  $3f_{HR}$ (three harmonics in total);
- around each harmonic, three scales spaced $\pm 10$ bpm apart. Each Morlet
  kernel is broad enough that neighbouring scales overlap and cover the band.

This makes nine scales per pixel, not dozens, and they move with each
participant's heart rate.

**Implementation.** The transform is computed in closed form in the frequency
domain. The Fourier transform of a Morlet wavelet at scale $a$ is a Gaussian,
$\sqrt{a}\, e^{-\pi^2 B (a f - C)^2}$. Each pixel's trace is mean-removed,
zero-padded (so the convolution is linear rather than circular),
Fourier-transformed once, multiplied by the Gaussian for each scale and
inverse-transformed. Nothing in the transform is learned. It is a fixed
feature of the clip at the estimated rate, and no gradient flows through it
back to the rate estimator.

To save memory, the transform runs on the clip downsampled by area averaging
(`WAVELET_DOWNSCALE`). The resulting complex field is then upsampled
bilinearly back to the input resolution, so the masks and every later stage
work at full resolution.

The complex field is used in three ways: its **magnitude** drives the
pulsatility mask, its **phase** drives the phase masks, and its **spatial and
temporal phase gradients** give the kinematic features.

## Pulsatility Mask

The pulsatility mask is a spatial attention map. It shows where in each
channel the video pulses at the heart rate, as opposed to where it is static
or varies for other reasons such as motion, lighting or noise.

**Physics first.** The starting point is a hand-defined *pulsatility index*
for each pixel: the log of the wavelet band power at the cardiac harmonics
divided by the pixel's total temporal variance,

$$
\pi(x) = \log \sum_k |W_k(x,t)|^2 \;-\; \log \operatorname{Var}_t\, I(x,t).
$$

This index measures how much of what a pixel does is the heartbeat. A static
background pixel scores low. A pixel whose variation is mostly cardiac scores
high.

**The network nudges.** A small separable 3-D convolutional network reads
the index and the wavelet magnitudes at every scale. Inputs are normalised per
channel, so 8-bit colour and 16-bit infrared or depth look alike. The network
outputs a **bounded correction** to the index, capped by a $\tanh$ at a fixed
number of nats. The logits are averaged over the window's frames, and a softmax
over the pixels turns them into a probability distribution that sums to one:

$$
M_p(x) = \operatorname{softmax}_x\!\Big(\overline{\pi}(x) + \beta \tanh\!\big(c(x)/\beta\big)\Big).
$$

Three design choices follow from failures seen during development:

- **One map per window.** The vessels do not move within a 10-second window.
  A mask that changed from frame to frame swept the weighted average across
  intensity gradients and added a slow drift to every waveform derived from
  it.
- **A softmax over pixels**, not a per-pixel sigmoid. The mask cannot vanish
  or be rescaled. It can only redistribute attention.
- **A bounded correction.** In early versions the network's correction was
  unbounded and the map collapsed onto one to three pixels. With the bound,
  the mask can be at most a fixed factor sharper than the physical index
  allows. It covers the pulsing neck and leaves vessel selection to the phase
  masks.

Light regularisation keeps the map spatially smooth (total variation) and
keeps the raw correction inside its bound. Ideally the pulsatility mask
highlights the exposed skin of the neck and suppresses background, hair and
clothing.

## Phase Mask

The neck's two large vessels, the carotid artery and the internal jugular
vein, both pulse at the cardiac frequency, but with a clear phase offset
between them. The phase masker uses this offset to decide, for each channel,
which pulsatile pixels belong to which trace.

**Relative phase.** The absolute phase of a pixel rotates continuously with
time and means nothing on its own. What separates an artery from a vein is
phase *relative to the channel's own pulse*. At each harmonic and frame, the
field is compared against a reference: the pulsatility-weighted mean field of
that channel. This gives each pixel a unit phasor that stays still over the
beat. Pixels moving with the dominant pulse sit near relative phase zero, and
pixels pulsing at a different point in the cycle sit elsewhere on the circle.

**Phase bins.** Each trace has a learned phase centre $\theta_s$ for each
channel. The centres are learned per channel because which vessel dominates
a channel's pulse can differ between colour, infrared and depth. A trace's
logit at a pixel is a fixed-gain cosine bin of the window-averaged relative
phase at the fundamental,

$$
\ell_s(x) = G\, R(x) \cos\!\big(\bar\phi(x) - \theta_s\big),
$$

where $R$ is the coherence of the relative phase over the window: 1 for a
pixel whose phase holds steady and 0 for one whose phase wanders. A
**background** class with a logit of zero collects pixels that pulse at
neither trace's phase.

**The network nudges.** As with the pulsatility mask, a small separable 3-D
convolutional network adds a bounded correction to each class's logit. Its
inputs are five circular-statistics features per harmonic: the relative
phase's cosine and sine, its coherence across the harmonic's scales, its
stability over time, and its agreement with neighbouring pixels. The
correction is constrained so that it cannot favour one class across the whole
image; it can only move pixels between classes. Its bound is below the phase
bins' separation, so **where the phase is coherent, the phase decides; where it
is not, the network decides.**

A single softmax over the `S` traces plus the background produces the masks.
This softmax is what makes the traces compete for pixels. The output is
effectively a soft clustering: for every channel, how much each pixel
belongs to the arterial trace, the venous trace, or neither.

**What we expect, and what is enforced.** No mask is labelled as artery or
vein. Which mask becomes which trace is decided only by the waveform predicted
through it and the loss on that waveform. Physiologically, we expect the
arterial mask to cover most of the pulsatile skin, weighted towards the
carotid, and the venous mask to concentrate over the jugular. The background
should absorb pulsatile pixels that fit neither. Regularisers encourage:

- the traces to claim most of the pulsatile mass, with the background keeping
  no more than an even share;
- each trace's mask to sit on pixels that are coherent across scales, steady
  over time and consistent with their neighbours;
- spatially smooth masks.

As with the pulsatility mask, the corrections are bounded and every class
keeps a small share of every pixel. Without this, one trace's mask collapsed
to nothing during long training runs.

## Kinematics

The kinematics module reads how the pulse **travels** across the skin. It
works directly from the complex wavelet field and contains no learned
parameters.

At each harmonic, the scales are first blended into one complex field (a
power-weighted mean). A travelling wave has a phase that decreases along its
direction of travel. Per pixel, frame and harmonic, the phase gradients give:

- **wavevector**, $\mathbf{k} = -\nabla_{x,y}\,\phi$: the spatial rate of phase
  change;
- **wavenumber**, $|\mathbf{k}|$;
- **angular frequency**, $\omega = \partial_t \phi$;
- **phase speed**, $v = \omega / |\mathbf{k}|$: how fast the phase front moves
  across the image, in pixels per frame;
- **heading**, $\mathbf{k}/|\mathbf{k}|$: the direction of travel;
- **amplitude**, $|W|$.

Phase gradients are computed as $\operatorname{Im}(\partial W \cdot
\overline{W}) / |W|^2$, so no phase unwrapping is needed.

**Why speed.** By the Bramwell–Hill relation, pulse wave velocity rises with
vessel stiffness, which in turn rises with distending pressure. The apparent
speed of the pulse is therefore a candidate cue to the pressure *level*,
which the intensity waveform alone cannot provide. This is the main
physics-based feature intended to help the models predict absolute values. Note
that the phase speed measured here is the speed of a phase pattern in the
image plane, in pixels per frame. It is a proxy for pulse wave velocity, not a
calibrated measurement of it.

The kinematics reach the prediction by two routes, in both models:

1. **Per frame:** the kinematic features are averaged under each trace's mask
   and passed to the waveform predictor, alongside the intensity.
2. **Per window:** the magnitude of the phase speed is averaged under each
   trace's mask and passed to the statistics head as a level feature.

## Statistics Head

The waveform path (masks, then CardioConv's mixer or CardioHydra's trunk) is
**shape-only** by construction. It works on normalised intensities and
standardised features, and its output is z-scored over the window. Whatever
it discarded about level and amplitude is restored by the statistics head.

**Inputs.** The statistics head works on window-level summaries:

- per channel, the clip's minimum, maximum and mean intensity;
- per channel and trace, the wavelet power pooled under the trace's mask, per
  harmonic;
- per channel and trace, the mask's share of the pulsatile pixels;
- which channels are present;
- the estimated heart rate;
- per channel, trace and harmonic, the pooled phase speed (from Kinematics).

Quantities with very different ranges are log-compressed, so 8-bit colour and
16-bit infrared and depth fall in the same band. The head deliberately uses no
batch or layer normalisation, because either would erase the between-sample
differences it exists to read.

**Outputs.** Per trace, a small MLP predicts a **mean** $\mu_s$ and a
**spread** $\sigma_s$. The final waveform is the shape rescaled to them:

$$
\hat y_s(t) = \mu_s + \sigma_s\, z_s(t), \qquad
\sigma_s = (1 + |\mu_s|)\,\operatorname{softplus}(\cdot),
$$

where $z_s(t)$ is the shape-only waveform z-scored over the window. Before
training, the bias of the mean readout is set to a physiological prior for
each trace, so the head starts from a plausible pressure level and learns
deviations from it. The spread is expressed relative to the mean, so it starts
in proportion whether the trace sits near 90 mmHg (ABP) or near 8 mmHg (CVP).

Because the pooled power and speed are averaged under the masks, the gradient
of the level prediction also reaches the masks, pushing them towards pixels
that are informative about level. The heart rate and the mask's share are
read without gradient.

## Training objective

Both models are trained end to end, with one loss per trace on the final
waveform: concordance correlation (CCC), which rewards both shape and level,
plus $L_1$ terms on the window's mean, soft maximum and soft minimum (a
proxy for systolic and diastolic values). Added to this are the label-free
regularisers of the two maskers and the rate estimator's waveform term.
Samples that fail the rate gate do not count towards the loss.

## CardioConv

CardioConv turns the masked video into waveforms with simple, mostly linear
operations.

1. **Candidate waveforms.** For each channel, the clip is min-max scaled.
   Then, at every frame, its pixels are averaged under the combined weight
   pulsatility mask × trace mask:

   $$
   y_{c,s}(t) = \frac{\sum_x M_p(x)\, M_s(x)\, \tilde I_c(x,t)}{\sum_x M_p(x)\, M_s(x)}.
   $$

   Normalising by the mask's area means the candidate depends on *which*
   pixels are chosen, not on how much mask there is. The kinematic features
   are pooled under the same weights and become further candidates. This
   step has no learned parameters. All of the spatial selection is done by
   the masks.

2. **Channel mixer.** Each candidate is standardised over the window. For each
   trace, a temporal 1-D convolution over that trace's candidates from every
   channel, followed by a pointwise readout, gives one shape-only waveform. The
   mixer learns how to combine channels, each with its own polarity, phase lag
   and quality. Each trace is mixed only from its own candidates. The
   kinematic branch is initialised to zero, so the mixer starts as a pure
   intensity filter bank and learns to use the kinematics.

The mixer is deliberately **linear**, with no nonlinearity between temporal
layers. A ReLU in that position rectified the zero-mean candidates, removing
half of every beat, and the predicted waveform's rate came out about 20 bpm
low.

Finally, the **statistics head** (above) sets each waveform's level and spread.

## CardioHydra

CardioHydra uses the same front end and statistics head as CardioConv. It
replaces the candidate-and-mixer stage with a learned sequence model for each
trace.

1. **Per-frame features.** At every frame, for each channel and trace, a richer
   set of quantities is averaged under pulsatility mask × trace mask. These
   are the log wavelet power per harmonic, the phase per harmonic (as the cosine
   and sine of the pooled complex mean), the kinematic features per harmonic,
   and the log intensity. The mask's share of the pulsatile pixels and the
   channel's presence are added. All channels' features are concatenated into
   one feature vector per frame, per trace.

   The intensity is the only feature that keeps the clip's full bandwidth.
   Everything taken from the wavelet field is restricted to the harmonics of
   the average heart rate, which smooths out an irregular beat. The intensity
   path is what lets the trunk see one.

2. **State-space trunk.** Each trace has its own trunk: a linear embedding,
   a stack of residual **bidirectional Mamba** blocks
   (LayerNorm → BiMamba3, with the forward and backward passes summed) and a
   linear readout per frame. Mamba is a selective state-space model. It models
   long-range temporal structure across the whole window in time linear in its
   length, and running it in both directions gives every frame context from
   the full window. This is the same building block as PhysMamba3 elsewhere in
   this repository.

3. **Statistics head**, as above.

The Mamba layers train at a reduced learning rate (one tenth of the rest of
the model). At the full rate, the CVP phase mask collapsed partway through
training on Neckflix. CardioHydra runs on CUDA only, because the Mamba3 kernels
are written in Triton.

## Current settings

These values reflect the code at the time of writing. They are still being
tuned, and the concepts above do not depend on them.

| Setting | Value |
| --- | --- |
| Input interface | 30 fps, 10 s windows, 72 × 72 px, channels R, G, B, IR, depth |
| Traces | ABP, CVP |
| Rate estimator | RhythmFormer, on the clip downscaled ×2; 1 bpm grid, 30–200 bpm |
| Rate gate tolerance | 8 bpm |
| Wavelet | complex Morlet `cmor1.5-1.0`; 3 harmonics × 3 scales at ±10 bpm; clip downscaled ×2 |
| Pulsatility correction bound | 1.5 nats |
| Phase bin gain / correction bound | 2 / 0.9 nats |
| CardioConv mixer | grouped temporal conv, 15 taps, 16 hidden channels per trace |
| CardioHydra trunk | width 128, 2 BiMamba3 blocks per trace; Mamba LR × 0.1 |
| Statistics head | one hidden layer of 64 per trace |
| Loss | per trace CCC + $L_1$ on mean, max, min; mask priors; rate waveform term |
