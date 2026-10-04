# CardioConv: a learned heart rate and an analytic wavelet bank

Design for removing CardioConv's label-rate stopgap. The model estimates
the heart rate from the video with a small learned module, the wavelet
stage becomes a closed-form Morlet bank written in torch so the estimate
receives a gradient from the waveform loss, and training gates each sample
on the estimate being right before anything downstream of the bank runs.

Baseline to compare against: `runs/neckflix_cardioconv` (participant 33
held out, 30 epochs, batch 4, the Neckflix 72 px RGBID interface, about
80 s per epoch). The new model runs the identical command and never sees
the label rate at test time.

## What the user asked for

- The heart rate module is completely learnable; nothing from the label
  reaches the model at test time.
- The wavelet stage keeps its restriction to scales around the harmonics
  of the rate, because transforming every scale was too slow.
- Training does not run the rest of the network on a sample whose rate the
  module has not yet learned. The gate acts per sample, at ±6 bpm.
- The rate target during training is the fused rate of the cardiac traces,
  since ABP and CVP alone are confused by their second harmonics.
- The estimator is the spectral classifier (approach A), reading the
  shrunk clip.

## Decisions made in the design conversation

| Decision | Choice | Why |
| --- | --- | --- |
| Shrink | fixed area average by 5, shared by estimator and bank; not learned | the field stays a property of the clip; averaging 25 pixels is a 5x SNR gain the current bilinear sample throws away |
| Estimator input | the shrunk clip | the rate is spectral; 14x14 pixels of neck is plenty and it costs almost nothing |
| Estimator output | softmax-weighted bin centre over a fixed 1 bpm grid, 30–200 bpm | bounded by construction, differentiable, independent of window length and frame rate |
| Estimator loss | cross-entropy to a 2 bpm Gaussian around the target, plus L1 in bpm on the readout | CE keeps the distribution unimodal and sharp; L1 trains the readout's position directly and its logged value is readable in bpm |
| Target rate | the evaluation's fused rate when two or more cardiac traces are present, else the single trace's | the fused spectrum suppresses the second-harmonic peak CVP alone often wins on |
| Gate | per sample, 6 bpm, training only | 6 bpm is the half-width the evaluation's SNR counts as signal around the rate; the bank's buffer is 10 bpm, so a passing sample's fundamental sits at least 4 bpm inside the band |
| Scale grid | unchanged: 3 harmonics, ±10 bpm, 1 bpm steps, 63 scales, harmonic-major | every module downstream sees the same field layout |
| Dependencies | `ptwt` and `pywavelets` removed | the analytic bank needs neither; nothing else imports them |

## The model

### Shrink

`F.interpolate(mode="area")` by `1 / DOWNSCALE`, in the top-level forward,
once, gradient-free (the input carries no gradient). It replaces the
bilinear interpolation inside the wavelet stage, which at a factor of 5
kept roughly one pixel in 25. The shrunk clip `small` `(B, C, T, h, w)`
feeds the estimator, the bank, the channel predictor and the statistics
head, as `small` does today.

### Rate estimator

New module `neural_methods/model/cardioconv/rate_estimator.py`, class
`RateEstimator`. Input: `small` and `present` `(B, C)`. Output: the rate
`(B,)` in bpm and the logits `(B, n_bins)`.

1. **Spatial attention.** Each pixel's trace loses its mean over the
   window. Two per-pixel statistics, `log1p` of the mean intensity and
   `log1p` of the temporal standard deviation, go through a 2D conv shared
   across channels (`Conv2d(2, 8, 3)`, ReLU, `Conv2d(8, 1, 1)`) to one logit
   per pixel per channel, softmaxed over the pixels. The attention-weighted
   sum of the mean-removed traces gives one trace per channel, `(B, C, T)`.
2. **Spectra.** Each trace is Hann-windowed and FFT'd, zero padded to the
   next power of two at or above `T`. Power is normalised to unit sum inside
   the 30–200 bpm band, `log1p`-compressed, and linearly resampled onto the
   fixed grid `BINS_BPM = 30, 31, ..., 200` (171 bins). An absent channel
   is a zero plane.
3. **Head.** `Conv1d(C, 16, 5)`, ReLU, `Conv1d(16, 16, 5)`, ReLU,
   `Conv1d(16, 1, 1)`, replicate-padded, over the frequency axis with the
   channels as input planes: a learned weighted sum of log spectra, which is
   a learned geometric mean, the same fusion the label side does by hand.
   One logit per bin.
4. **Readout.** `rate = softmax(logits) @ BINS_BPM`.

The estimator's loss terms, computed in the top-level forward from the
logits and the target (see "Terms" below), reach only the estimator's
parameters: the target is a constant and nothing downstream feeds back
into the logits except through the rate, which is the point of the design.

### Analytic Morlet bank

`wavelet_transformer.py` keeps its class name `WaveletTransformer` and its
call shape, `(small, rate) -> field (B, C, T, h, w, K)`, and loses its
`@torch.no_grad()`, the ptwt call, the Python loop over samples and the
shrink (now the top level's). `heart_rate_scales` in `utils.py` becomes a
torch function returning centre frequencies `(B, K)` in cycles per frame
from the rate tensor; `buffer_steps` stays.

For the complex Morlet `cmor1.5-1.0` (bandwidth `B_w = 1.5`, centre
`C = 1.0`), the wavelet at scale `a` (in frames) has the spectrum

    Psi_a(f) = sqrt(a) * exp(-pi^2 * B_w * (a f - C)^2),   f in cycles/frame

centred at `f = C / a`, which is the rate harmonic the scale was chosen
for, so `a = C / f_k` and the spectrum is a closed form of the rate. No
table, no truncation, and `d(field) / d(rate)` exists.

Forward:

1. the pixel traces lose their mean and are zero padded to a fixed length
   `n = next_fast_len(T + 2 * PAD)`, where `PAD` covers four standard
   deviations of the slowest kernel the model can be asked for (the
   fundamental at `RATE_RANGE[0] - buffer_bpm`), so the result is a linear,
   not circular, convolution as ptwt's was;
2. one `fft` over the padded traces, `(B, C, h, w, n)`;
3. per harmonic (3 iterations), the 21 spectra `(B, 21, n)` multiply the
   traces' FFT, an `ifft` returns the complex field for those scales, and
   the window `[PAD, PAD + T)` is cropped out;
4. the harmonics are concatenated on the last axis, harmonic-major, as today.

The sign convention (conjugate the wavelet, as pywt does for a complex
wavelet) is matched to ptwt so `field.angle()` means the same to the phase
masker. Memory: autograd retains the traces' FFT and the spectra, both
small; the `(B, C, h, w, 21, n)` product is forward-only and about 700 MB
at the baseline batch per harmonic, against 2 GB in one shot. The field is
the size it is today.

### The gate and the top-level forward

`CardioConv.forward(video, target_rate=None)`:

1. `present`, `small` (area shrink).
2. `rate, logits = self.rate_estimator(small, present)`.
3. Training (`target_rate` given): `has_target = isfinite(target_rate)`;
   `passed = has_target & (|rate - target_rate| <= tolerance)`. The
   estimator's terms are computed over `has_target` samples. Evaluation
   (`target_rate is None`): every sample passes and no estimator term is
   computed (the regulariser dict still carries every declared term, as
   zeros).
4. If no sample passes, the one with the smallest `|rate - target_rate|`
   runs anyway and stays masked out of the loss, so every parameter is in
   the graph on every step (distributed training refuses unused ones).
5. The bank, maskers, predictor, mixer and statistics head run on the
   passing samples only, selected by index on `small`, `present` and `rate`.
   Mask priors are computed over those samples.
6. The output `(B, S, T)` is zeros for failing samples; `self.sample_mask`
   holds `passed` `(B,)` for the wrapper.

`NEEDS_HEART_RATE` and the leakage docstring go. The class declares
`NEEDS_RATE_TARGET = True`.

### Terms

Two terms join `REGULARISERS`, reported identically under every trace like
the pulsatility priors (the trainer's total is the mean over traces, so
each counts once):

| Term | Value | Gradient reaches |
| --- | --- | --- |
| `rate` | cross-entropy between `softmax(logits)` and a Gaussian of 2 bpm standard deviation centred on the target, normalised over the bins, mean over `has_target` samples | the estimator |
| `rate_error` | mean over `has_target` samples of `|rate - target_rate|`, in bpm | the estimator |

A weighted L1 term's logged value is `weight * bpm`, so the loss log reads
in bpm once divided by the weight. Both are zero when no sample has a
target.

## The shared contract and config

### Wrapper (`src/models.py`)

- `label_heart_rate` becomes `label_rate(batch, traces, fs)`: per sample,
  `clean` and `spectrum` every present cardiac trace, `fuse` them when
  there are two or more, `rate_of` the result; NaN for a sample with none.
  All four functions are `src.evaluation.rate`'s; nothing is copied.
- `MultiTraceModel.forward`: for a backbone declaring `NEEDS_RATE_TARGET`,
  in training mode (`self.training`) compute the target and call
  `backbone(video, target_rate=target)`; in evaluation call
  `backbone(video)`. After the forward, if the backbone has a
  `sample_mask`, AND it into every trace's `label_mask` in the returned
  dict. `PerSignalLoss` already yields exactly zero for a trace no sample
  carries, so nothing in the loss module changes.
- The leakage paragraphs in `label_heart_rate`, `MultiTraceModel` and
  `_build_cardioconv` go.

### Config (`src/model_config.py`)

- `CardioConvConfig` gains `RATE_TOLERANCE_BPM: float = 6.0`, validated as
  positive and strictly less than the bank's buffer (10 bpm), with the
  refusal naming both numbers.
- `REGULARISATION` accepts `RATE` and `RATE_ERROR` because the backbone
  declares them; the config class does not list them.
- `_build_cardioconv` passes `rate_tolerance_bpm=cfg.RATE_TOLERANCE_BPM`.

### Config file

`configs/experiment_configs/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H72W72.yaml`:
the leakage comment goes; `RATE_TOLERANCE_BPM: 6.0` joins the switches;
`REGULARISATION` gains `RATE: 1.0` and `RATE_ERROR: 0.1`. The HPC copies
under `configs/hpc_configs_200/` and `configs/hpc_configs_300/` get the
same edit.

### Docs

- `docs/adding_a_model.md`: the paragraph on the `NEEDS_HEART_RATE`
  stopgap goes; the regulariser bullet says a term may be supervised by a
  target the wrapper supplies, not only label-free.
- `docs/plans/2026-09-18-model-regularisation-design.md`: the "depends on"
  row for a regulariser becomes "a model's own tensors, or a target the
  wrapper supplies".

### Dependencies

`uv remove ptwt pywavelets`, once a grep confirms the wavelet stage was the
only importer (it is today). This touches the shared venv, so it runs when
the user says so, not before.

## Out of scope

- The missing `configs/original_model_config/` file and README command for
  CardioConv (the migration's unfinished proof of life).
- Logging the estimated rate per window into the test records; the
  evaluation's `rates.csv` already compares the predicted waveform's rate
  with the fused reference, which is the estimator's accuracy at one remove.
- Any change to the maskers, predictor, mixer or statistics head beyond
  running them on the passing subset.

## Verification

- A throwaway script in the session scratchpad compares the analytic bank
  with ptwt on a synthetic trace at one rate, before ptwt is removed:
  magnitude and phase agree to within the table quantisation. Deleted
  afterwards; not a test, per the repo's rule.
- The proof run is the baseline's command with a new runs directory:

      uv run python scripts/run.py --datasets neckflix \
        --config configs/experiment_configs/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H72W72.yaml \
        --epochs 30 --batch-size 4 --num-workers 4 \
        --test-participant-dataset neckflix --test-participant-id 33 \
        --runs-dir runs/neckflix_cardioconv_rate

  Comparison: the two `losses.csv` side by side (`rate_error / 0.1` is the
  training bpm error; the CCC columns are the waveform result), the
  `rates.csv` reference and predicted rate columns per recording, and the
  per-signal level errors in `signals.csv`.
- The user runs it.
