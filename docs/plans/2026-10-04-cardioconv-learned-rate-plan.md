# CardioConv learned heart rate: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CardioConv estimates the heart rate from the video with a learned
spectral module, the wavelet stage becomes a closed-form Morlet bank in
torch so that estimate receives a gradient, and training gates each sample
on the estimate being within 6 bpm of the fused label rate.

**Architecture:** Four files change inside
`neural_methods/model/cardioconv/` (a new `rate_estimator.py`, a rewritten
`wavelet_transformer.py`, `utils.py` and the top-level `cardioconv.py`);
the wrapper in `src/models.py` supplies a training-only target rate and
honours a per-sample mask; `src/model_config.py` gains one switch; three
config files, the template and two docs follow. No trainer or loss-module
change: an all-masked trace already yields exactly zero loss, and the new
terms ride the existing regulariser path.

**Tech Stack:** PyTorch (`torch.fft`), einops, the repo's `uv` environment.
`ptwt` and `pywavelets` are removed at the end.

**Spec:** `docs/plans/2026-10-04-cardioconv-learned-rate-design.md`

## Global Constraints

- All tensor reshaping is einops (`rearrange` / `reduce` / `einsum`), never `view` / `permute` / `reshape`. Indexing, `unbind`, `stack`, `cat` and `[..., None]` are fine.
- PEP 8 names; `uv run ruff check neural_methods/model/cardioconv src` is the check.
- No tests. Verification is a throwaway script in the scratchpad plus the run command in the last task. Do not create anything under `tests/`.
- Dependencies go through `uv add` / `uv remove`, never pip. The removal in Task 5 touches the shared venv and runs only when the user says so.
- The scale grid is unchanged: 3 harmonics, ±10 bpm, 1 bpm steps, 63 scales, harmonic-major.
- Every config key is required (`src.config.build`), so a new `MODEL` key must be added to every CardioConv config file in the same commit.
- Legacy code is deleted, not adapted: no `NEEDS_HEART_RATE`, no `heart_rate=` keyword, no ptwt fallback.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- The scratchpad is `C:\Users\20759193\AppData\Local\Temp\claude\c--Users-20759193-source-repos-remote-physiology\7fe63458-9abe-46fd-875d-c0fccb3c8f10\scratchpad`; written `SCRATCH` below.

## Review Focus

Inputs the spec implies but no step's check exercises, most likely to bite first. Each has its check added to the owning task.

1. **A window shorter than the pad or not a power of two** (e.g. 128 frames at 30 Hz): the bank must return `T` frames of field, not crop wrongly. Task 1's script runs `T=128` as well as `T=300`.
2. **A batch where every sample fails the gate** early in training: one sample must still run, every parameter must receive a gradient, the loss must be the rate terms only. Task 3's smoke script forces this by passing a target 50 bpm away.
3. **A sample with no cardiac label** (`label_mask` False for every trace): its target is NaN, it must be excluded from both rate terms and the gate without producing NaN anywhere. Task 3's smoke script sets one sample's masks False.
4. **Evaluation mode**: no target, every sample runs, no NaN, output shape `(B, S, T)`. Task 3's smoke script runs `model.eval()` once.
5. **Mixed precision** (`PRECISION: bfloat16`): `torch.fft` rejects half precision, so the estimator and the bank must cast to float32 themselves. Task 3's smoke script runs one forward under `torch.autocast("cpu", dtype=torch.bfloat16)`.

---

### Task 0: Branch

**Files:** none.

- [ ] **Step 1: Branch off main**

```bash
git checkout -b cardioconv-learned-rate
```

- [ ] **Step 2: Commit the spec and this plan**

```bash
git add docs/plans/2026-10-04-cardioconv-learned-rate-design.md docs/plans/2026-10-04-cardioconv-learned-rate-plan.md
git commit -m "docs(plans): CardioConv learned heart rate, design and plan

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 1: The analytic Morlet bank and the shared shrink

**Files:**
- Modify: `neural_methods/model/cardioconv/utils.py` (replace `heart_rate_scales`, add `shrink`, drop numpy)
- Rewrite: `neural_methods/model/cardioconv/wavelet_transformer.py`
- Modify: `neural_methods/model/cardioconv/cardioconv.py:107,128` (shrink at the top level; bank takes `small`)
- Throwaway: `SCRATCH/check_bank.py`

**Interfaces:**
- Produces `utils.shrink(video: Tensor, factor: int) -> Tensor`, `(B, C, T, H, W) -> (B, C, T, H // factor, W // factor)`, area average, float32.
- Produces `utils.centre_frequencies(rate: Tensor, fs: float, n_harmonics: int, buffer_bpm: float, step_bpm: float) -> Tensor`, `(B,) bpm -> (B, K)` cycles per frame, harmonic-major.
- Produces `WaveletTransformer(fs, n_harmonics, bandwidth=1.5, centre=1.0, buffer_bpm=10.0, step_bpm=1.0, min_bpm=30.0)` with `.n_scales`, `.buffer_bpm`, and `forward(small, rate) -> field (B, C, T, h, w, K)` complex64, differentiable in `rate`.
- Consumes nothing from later tasks. `cardioconv.py` still takes `heart_rate` from the wrapper after this task; that changes in Task 3.

- [ ] **Step 1: Replace the scale helper and add the shrink in `utils.py`**

Delete `import numpy as np` and the `heart_rate_scales` function. Add `import torch.nn.functional as F` is already there. Add, after `buffer_steps`:

```python
def centre_frequencies(rate: torch.Tensor, fs: float, n_harmonics: int, buffer_bpm: float,
                       step_bpm: float) -> torch.Tensor:
    """``(B,)`` bpm -> ``(B, K)`` wavelet centre frequencies in cycles per frame.

    Harmonic ``k`` contributes the rates ``k * rate +- buffer_bpm`` in
    ``step_bpm`` steps. Ordered by harmonic, the fundamental first, so the
    last axis of the transform folds into ``(harmonic, step)``. A torch
    function of the rate, so the field carries ``d(field) / d(rate)``.
    """
    offsets = torch.linspace(-buffer_bpm, buffer_bpm, buffer_steps(buffer_bpm, step_bpm),
                             device=rate.device, dtype=rate.dtype)
    rates = torch.cat([k * rate[:, None] + offsets for k in range(1, n_harmonics + 1)], dim=1)
    return rates / 60.0 / fs


def shrink(video: torch.Tensor, factor: int) -> torch.Tensor:
    """``(B, C, T, H, W)`` -> ``(B, C, T, H // factor, W // factor)``, each output
    pixel the mean of its ``factor x factor`` block, in float32. An area
    average, not a sample: pulse is weak and averaging the block is the
    gain a shrink is for."""
    b = video.shape[0]
    flat = rearrange(video.float(), "b c t h w -> (b c) t h w")
    small = F.interpolate(flat, scale_factor=1.0 / factor, mode="area")
    return rearrange(small, "(b c) t h w -> b c t h w", b=b)
```

Update the module docstring's first line to: `"""Functions CardioConv's modules share: the wavelet centre frequencies of a heart rate, the spatial shrink, circular statistics of wrapped phase, and presence-masked reduction."""`

- [ ] **Step 2: Rewrite `wavelet_transformer.py`**

Replace the whole file with:

```python
"""Heart-rate-conditioned, pixel-wise complex Morlet transform, in closed form."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from neural_methods.model.cardioconv.utils import buffer_steps, centre_frequencies


class WaveletTransformer(nn.Module):
    """The complex wavelet field of a shrunk clip at the harmonics of its heart rate.

    The continuous wavelet transform with the complex Morlet
    ``psi(t) = (pi B)^-1/2 exp(2 pi i C t) exp(-t^2 / B)`` has, at scale ``a``
    frames, the spectrum ``sqrt(a) exp(-pi^2 B (a f - C)^2)`` in cycles per
    frame: one Gaussian centred at ``f = C / a``. So the transform is the
    traces' FFT times that Gaussian, inverse-transformed, and the scale
    enters analytically. That is what lets the rate the scales come from
    be learned: the field carries ``d(field) / d(rate)``. (pywt reaches the
    same quantity by resampling a tabulated integral of the wavelet by
    integer index, which has no gradient in the scale.)

    Each pixel's trace loses its mean and is zero padded on both sides by
    four standard deviations of the slowest kernel the model can meet, so
    the convolution is linear, not circular. One FFT serves every scale;
    the scales are run per harmonic to bound the forward-only product.
    Nothing here is learned. The field stays complex: power is its
    ``abs()`` and phase its ``angle()``, each taken by the stage that wants it.

    Args:
      fs: frame rate of the clip, in Hz.
      n_harmonics: harmonics of the heart rate the scales cover.
      bandwidth, centre: the Morlet's ``B`` and ``C``; ``cmor1.5-1.0``.
      buffer_bpm, step_bpm: the scales sit ``+-buffer_bpm`` around each
        harmonic, ``step_bpm`` apart.
      min_bpm: the lowest rate the model can be asked for; sets the pad.
    """

    def __init__(self, fs: float, n_harmonics: int, bandwidth: float = 1.5, centre: float = 1.0,
                 buffer_bpm: float = 10.0, step_bpm: float = 1.0, min_bpm: float = 30.0):
        super().__init__()
        self.fs = fs
        self.n_harmonics = n_harmonics
        self.bandwidth = bandwidth
        self.centre = centre
        self.buffer_bpm = buffer_bpm
        self.step_bpm = step_bpm
        self.steps = buffer_steps(buffer_bpm, step_bpm)
        self.n_scales = n_harmonics * self.steps
        slowest = (min_bpm - buffer_bpm) / 60.0 / fs                 # cycles per frame
        sigma_frames = (centre / slowest) * math.sqrt(bandwidth / 2.0)
        self.pad = math.ceil(4.0 * sigma_frames)

    def forward(self, small: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, h, w)`` shrunk clip and ``(B,)`` bpm -> the complex field
        ``(B, C, T, h, w, K)``, ``K = n_scales``, harmonic-major."""
        b, c, t, h, w = small.shape
        traces = rearrange(small.float(), "b c t h w -> b (c h w) t")
        traces = traces - traces.mean(dim=-1, keepdim=True)
        n = 1 << (t + 2 * self.pad - 1).bit_length()
        spectrum = torch.fft.fft(F.pad(traces, (self.pad, n - t - self.pad)), dim=-1)
        freqs = torch.fft.fftfreq(n, device=small.device, dtype=torch.float32)
        scales = self.centre / centre_frequencies(rate.float(), self.fs, self.n_harmonics,
                                                  self.buffer_bpm, self.step_bpm)        # (B, K) frames
        fields = []
        for a in scales.split(self.steps, dim=1):                                       # (B, steps)
            kernel = a.sqrt()[..., None] * torch.exp(
                -(math.pi ** 2) * self.bandwidth * (a[..., None] * freqs - self.centre) ** 2)
            product = spectrum[:, :, None, :] * kernel[:, None, :, :]                   # (B, P, steps, n)
            fields.append(torch.fft.ifft(product, dim=-1)[..., self.pad:self.pad + t])
        field = torch.cat(fields, dim=2)                                                # (B, P, K, T)
        return rearrange(field, "b (c h w) k t -> b c t h w k", c=c, h=h, w=w)
```

- [ ] **Step 3: Shrink at the top level in `cardioconv.py`**

Change the import line to `from neural_methods.model.cardioconv.utils import shrink` (add it beside the existing `require_min_frame` import). In `__init__`, the bank line becomes:

```python
        self.wavelet_transformer = WaveletTransformer(fs, N_HARMONICS)
```

In `forward`, replace `field, small = self.wavelet_transformer(video, heart_rate)` with:

```python
        small = shrink(video, DOWNSCALE)
        field = self.wavelet_transformer(small, heart_rate)
```

Delete the `from neural_methods.model.cardioconv.utils import buffer_steps, heart_rate_scales` style imports nothing else uses: grep `heart_rate_scales` across the package and confirm zero hits.

- [ ] **Step 4: Write the throwaway comparison script**

`SCRATCH/check_bank.py`:

```python
"""Throwaway: the analytic bank against ptwt at one rate, plus a gradient check."""
import math
import sys

import numpy as np
import pywt
import torch
from ptwt import cwt

sys.path.insert(0, r"C:\Users\20759193\source\repos\remote-physiology")
from neural_methods.model.cardioconv.wavelet_transformer import WaveletTransformer  # noqa: E402

FS, RATE = 30.0, 72.0
torch.manual_seed(0)
for T in (300, 128):
    t = torch.arange(T) / FS
    trace = (torch.sin(2 * math.pi * RATE / 60 * t) + 0.4 * torch.sin(2 * math.pi * 2 * RATE / 60 * t + 0.7)
             + 0.1 * torch.randn(T))
    bank = WaveletTransformer(FS, 3)
    small = trace[None, None, :, None, None]                      # (1, 1, T, 1, 1)
    rate = torch.tensor([RATE], requires_grad=True)
    field = bank(small, rate)[0, 0, :, 0, 0, :]                   # (T, K)

    wavelet = pywt.ContinuousWavelet("cmor1.5-1.0")
    offsets = np.linspace(-10, 10, 21)
    bpm = np.concatenate([k * RATE + offsets for k in (1, 2, 3)])
    scales = wavelet.center_frequency * FS / (bpm / 60.0)
    ref, _ = cwt((trace - trace.mean())[None, :], scales, wavelet, sampling_period=1.0 / FS)
    ref = torch.as_tensor(np.asarray(ref))[:, 0, :].T.to(torch.complex64)   # (T, K)

    centre = slice(T // 4, 3 * T // 4)
    ratio = (field.abs() / ref.abs().clamp_min(1e-6))[centre]
    phase = torch.angle(field[centre] * ref[centre].conj())
    print(f"T={T}: magnitude ratio median {ratio.median():.3f} (min {ratio.min():.3f}, max {ratio.max():.3f}); "
          f"phase offset median {phase.median():.3f} rad, spread {phase.std():.3f} rad")
    field.abs().sum().backward()
    print(f"T={T}: d|field|/d(rate) = {rate.grad.item():.4g}, field shape {tuple(bank(small, rate.detach()).shape)}")
```

- [ ] **Step 5: Run it and settle sign and normalisation**

Run: `uv run python "SCRATCH/check_bank.py"` (expand `SCRATCH`).

Expected: for both `T`, magnitude ratio median within 5 % of 1.000 and phase offset median within 0.1 rad with spread under 0.1 rad; `rate.grad` finite and non-zero; shape `(1, 1, T, 1, 1, 63)`.

Decision rules if not:
- phase offset median near ±π with small spread: pywt's convention has the opposite sign; negate the kernel (`kernel = -kernel`) and rerun.
- phase offset grows linearly with time (spread large, median meaningless): the conjugation is wrong; replace `freqs` by `-freqs` in the kernel and rerun.
- magnitude ratio median off by a constant: pywt's normalisation differs by that constant; multiply the kernel by it so the masks and the statistics head see power on today's scale, record the constant and why in the class docstring.

- [ ] **Step 6: Lint**

Run: `uv run ruff check neural_methods/model/cardioconv`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add neural_methods/model/cardioconv/utils.py neural_methods/model/cardioconv/wavelet_transformer.py neural_methods/model/cardioconv/cardioconv.py
git commit -m "feat(cardioconv): closed-form Morlet bank in torch, area shrink at the top level

The wavelet stage computes the complex Morlet spectrum analytically from the
rate tensor, so the field carries a gradient to the rate; ptwt's table lookup
could not. The shrink moves to the top level and becomes an area average.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The rate estimator

**Files:**
- Create: `neural_methods/model/cardioconv/rate_estimator.py`
- Throwaway: `SCRATCH/check_estimator.py`

**Interfaces:**
- Produces `RateEstimator(in_channels: int, fs: float, low_bpm=30.0, high_bpm=200.0, hidden=16, attention_hidden=8)` with buffer `.bins` `(n_bins,)` float32 bpm and `forward(small, present) -> (rate (B,), logits (B, n_bins))`, both float32.
- Produces `rate_terms(logits, rate, target, bins, sigma_bpm=2.0) -> dict` with keys `"rate"` (cross-entropy) and `"rate_error"` (mean absolute bpm), both `()` tensors, zero when no sample has a finite target.
- Consumes `utils.EPS`.

- [ ] **Step 1: Write the module**

```python
"""Shrunk clip -> the heart rate, as a softmax over 1 bpm bins."""

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce

from neural_methods.model.cardioconv.utils import EPS

#: The two per-pixel statistics the spatial attention reads.
N_STATISTICS = 2


def interpolate_bins(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``(..., n)`` values on a uniform grid -> ``(..., m)`` at fractional
    indices ``positions`` ``(m,)``, by linear interpolation; positions past
    either end take the end value."""
    low = positions.floor().long().clamp(0, values.shape[-1] - 2)
    frac = (positions - low).clamp(0.0, 1.0)
    return values[..., low] * (1.0 - frac) + values[..., low + 1] * frac


def rate_terms(logits: torch.Tensor, rate: torch.Tensor, target: torch.Tensor, bins: torch.Tensor,
               sigma_bpm: float = 2.0) -> dict:
    """The estimator's two terms over the samples with a finite target.

    ``rate`` is the cross-entropy between the bin distribution and a Gaussian
    of ``sigma_bpm`` around the target, normalised over the bins: it makes
    the distribution unimodal and sharp. ``rate_error`` is the mean absolute
    error of the readout in bpm: it trains the readout's position directly,
    and weighted it logs as ``weight * bpm``. Both are zero when no sample
    has a target.
    """
    has = torch.isfinite(target)
    zero = logits.new_zeros(())
    if not bool(has.any()):
        return {"rate": zero, "rate_error": zero}
    goal = target[has]
    prior = torch.exp(-0.5 * ((bins[None, :] - goal[:, None]) / sigma_bpm) ** 2)
    prior = prior / prior.sum(dim=-1, keepdim=True)
    cross_entropy = -(prior * logits[has].log_softmax(dim=-1)).sum(dim=-1).mean()
    return {"rate": cross_entropy, "rate_error": (rate[has] - goal).abs().mean()}


class RateEstimator(nn.Module):
    """Which rate the clip pulses at, read off a learned fusion of per-channel spectra.

    Three parts. *Attention*: each pixel's trace loses its mean; the log of
    its mean intensity and of its temporal standard deviation go through a
    2-D convolution shared by every channel to one logit per pixel, softmaxed
    over the pixels, and the weighted sum of the traces is one trace per
    channel. *Spectra*: each trace is Hann-windowed, FFT'd at the next power
    of two, and its power resampled onto a fixed grid of 1 bpm bins,
    normalised to unit sum over the grid and ``log1p``-compressed with the
    bin count as gain, so a flat spectrum reads ``log 2`` everywhere and a
    peaked one is O(1) at its peak whatever the channel's bit depth. The
    fixed grid is what makes the head independent of window length and
    frame rate. An absent channel is a zero plane. *Head*: a 1-D convolution
    over frequency with the channels as input planes, a learned weighted sum
    of log spectra, which is a learned geometric mean, the same fusion the
    label side does by hand. The rate is the softmax-weighted bin centre:
    bounded by construction, differentiable, and the only thing the rest of
    the model receives.

    Args:
      in_channels: video channels ``C``.
      fs: frame rate, in Hz.
      low_bpm, high_bpm: the grid, inclusive, 1 bpm apart.
      hidden: width of the head.
      attention_hidden: width of the attention's hidden layer.
    """

    def __init__(self, in_channels: int, fs: float, low_bpm: float = 30.0, high_bpm: float = 200.0,
                 hidden: int = 16, attention_hidden: int = 8):
        super().__init__()
        self.fs = fs
        self.register_buffer("bins", torch.arange(low_bpm, high_bpm + 1.0, dtype=torch.float32))
        self.attention = nn.Sequential(
            nn.Conv2d(N_STATISTICS, attention_hidden, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_hidden, 1, kernel_size=1),
        )
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def traces(self, small: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, h, w)`` -> ``(B, C, T)`` attention-pooled, mean-removed traces."""
        b = small.shape[0]
        clip = small.float()
        centred = clip - reduce(clip, "b c t h w -> b c 1 h w", "mean")
        statistics = torch.stack([
            torch.log1p(reduce(clip, "b c t h w -> b c h w", "mean").clamp_min(0.0)),
            torch.log1p(centred.std(dim=2)),
        ], dim=2)                                                                   # (B, C, 2, h, w)
        logits = self.attention(rearrange(statistics, "b c f h w -> (b c) f h w"))
        weights = rearrange(logits, "(b c) 1 h w -> b c (h w)", b=b).softmax(dim=-1)
        return einsum(rearrange(centred, "b c t h w -> b c t (h w)"), weights, "b c t p, b c p -> b c t")

    def spectra(self, traces: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` -> ``(B, C, n_bins)`` log power on the bpm grid."""
        t = traces.shape[-1]
        window = torch.hann_window(t, device=traces.device, dtype=traces.dtype)
        n = 1 << (t - 1).bit_length()
        power = torch.fft.rfft(traces * window, n=n, dim=-1).abs() ** 2         # (B, C, n // 2 + 1)
        bin_hz = self.fs / n
        positions = self.bins / 60.0 / bin_hz                                   # fractional rfft bins
        on_grid = interpolate_bins(power, positions)
        on_grid = on_grid / on_grid.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return torch.log1p(on_grid * self.bins.numel())

    def forward(self, small: torch.Tensor, present: torch.Tensor) -> tuple:
        """``(B, C, T, h, w)`` shrunk clip and ``(B, C)`` presence -> the rate
        ``(B,)`` in bpm and the logits ``(B, n_bins)`` it was read from."""
        spectra = self.spectra(self.traces(small)) * rearrange(present.float(), "b c -> b c 1")
        logits = rearrange(self.head(spectra), "b 1 k -> b k")
        return logits.softmax(dim=-1) @ self.bins, logits
```

- [ ] **Step 2: Write the throwaway check**

`SCRATCH/check_estimator.py`:

```python
"""Throwaway: an untrained estimator's spectra peak at the true rate, and the terms train it."""
import math
import sys

import torch

sys.path.insert(0, r"C:\Users\20759193\source\repos\remote-physiology")
from neural_methods.model.cardioconv.rate_estimator import RateEstimator, rate_terms  # noqa: E402

torch.manual_seed(0)
FS, T, B, C = 30.0, 300, 4, 5
rates = torch.tensor([48.0, 72.0, 95.0, 130.0])
t = torch.arange(T) / FS
pulse = torch.sin(2 * math.pi * rates[:, None] / 60 * t)                    # (B, T)
small = 100 + 3 * pulse[:, None, :, None, None] + torch.randn(B, C, T, 14, 14)
present = torch.ones(B, C, dtype=torch.bool)
present[0, 4] = False                                                        # one absent channel

model = RateEstimator(C, FS)
spectra = model.spectra(model.traces(small))                                 # (B, C, n_bins)
peaks = model.bins[spectra.mean(dim=1).argmax(dim=-1)]
print("spectral peaks (bpm):", peaks.tolist(), "true:", rates.tolist())

optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
target = rates.clone()
target[3] = float("nan")                                                      # one sample without a label
for step in range(200):
    rate, logits = model(small, present)
    terms = rate_terms(logits, rate, target, model.bins)
    loss = terms["rate"] + 0.1 * terms["rate_error"]
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
rate, logits = model(small, present)
print("after 200 steps:", [f"{r:.1f}" for r in rate.tolist()], "error on labelled samples (bpm):",
      f"{(rate[:3] - rates[:3]).abs().mean():.2f}")
print("T=128 shape:", tuple(model(small[:, :, :128], present)[1].shape))
```

- [ ] **Step 3: Run it**

Run: `uv run python "SCRATCH/check_estimator.py"`
Expected: spectral peaks within 3 bpm of the true rates before any training (the signal is in the spectrum); after 200 steps the mean error on the three labelled samples is under 2 bpm; no NaN; the `T=128` logits are `(4, 171)`.

- [ ] **Step 4: Lint**

Run: `uv run ruff check neural_methods/model/cardioconv`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add neural_methods/model/cardioconv/rate_estimator.py
git commit -m "feat(cardioconv): spectral heart-rate estimator over a fixed bpm grid

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The gate, the wrapper contract and the config

**Files:**
- Modify: `neural_methods/model/cardioconv/cardioconv.py` (whole forward, constants, docstring)
- Modify: `src/models.py:86-105` (`label_heart_rate` -> `label_rate`), `:212-225` (`MultiTraceModel.forward`), `:268-274` (`_build_cardioconv`)
- Modify: `src/model_config.py:142-157` (`CardioConvConfig`)
- Modify: `configs/experiment_configs/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H72W72.yaml`, `configs/hpc_configs_200/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H200W200.yaml`, `configs/hpc_configs_300/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H300W300.yaml`
- Modify: `configs/_model_config_template.yaml:22,35`
- Throwaway: `SCRATCH/check_model.py`

**Interfaces:**
- Consumes `WaveletTransformer.forward(small, rate)` and `.buffer_bpm` (Task 1), `shrink` (Task 1), `RateEstimator`, `rate_terms` (Task 2).
- Produces `CardioConv(in_channels, traces, fs, pulsatility_masker=True, phase_masker=True, trace_independent=True, supervise_masks=True, rate_tolerance_bpm=6.0)` with `NEEDS_RATE_TARGET = True`, `REGULARISERS` including `"rate"` and `"rate_error"`, `forward(video, target_rate=None) -> (B, S, T)`, and after a forward `.sample_mask` `(B,)` bool and `.regularisers()`.
- Produces `cardioconv.BUFFER_BPM = 10.0`.
- Produces `src.models.label_rate(batch, traces, fs) -> (B,) float32 bpm, NaN where no cardiac trace is present`.

- [ ] **Step 1: Rewrite the top of `cardioconv.py`**

Replace the module docstring's items 1 and 2 and the "heart rate is given" paragraph so the list reads:

```
1. :class:`~neural_methods.model.cardioconv.rate_estimator.RateEstimator`
   reads the heart rate off a learned fusion of the channels' spectra;
2. :class:`~neural_methods.model.cardioconv.wavelet_transformer.WaveletTransformer`
   gives the complex wavelet field of every pixel at the harmonics of that
   rate, in closed form, so the rate receives the waveform loss's gradient
   through the field (not learned itself);
3. ... (pulsatility masker, as before, renumbered through 7)
```

and replace the "**The heart rate is given, not estimated.**" paragraph with:

```
**The rate is learned, and training gates on it.** The wrapper hands the
model, in training mode only, the rate of each sample's fused cardiac
labels (``src.models.label_rate``). The estimator's two terms are computed
against it, and a sample whose estimate is more than ``rate_tolerance_bpm``
from it does not go through the bank or anything after: its output is zero
and ``sample_mask`` tells the wrapper to mask it out of the loss. Until the
estimator is close, the field would be noise and the masks would learn
from noise. One sample always runs, so every parameter stays in the graph.
At test time nothing from the label reaches the model and every sample runs.
```

Delete the sentence in the "Differences from the CardioHydra repository's CardioConv" paragraph about the lazy first layer if it survives unchanged; keep the rest.

Replace the constants block from `#: Harmonics of the heart rate...` through `PHASE_PRIORS = (...)` with:

```python
#: Harmonics of the heart rate the wavelet scales cover.
N_HARMONICS = 3

#: The wavelet scales sit this far either side of each harmonic, in bpm. The
#: training gate's tolerance must stay inside it.
BUFFER_BPM = 10.0

#: The spatial shrink before the estimator and the wavelet; every mask lives on this grid.
DOWNSCALE = 5

#: The smallest frame that leaves the masks a 3x3 grid, the spatial
#: neighbourhood their convolutions and coherence features read.
MIN_FRAME = 3 * DOWNSCALE

PULSATILITY_PRIORS = ("sparsity", "pulsatility_tv", "pulsatility_smoothness")
PHASE_PRIORS = ("harmonic_coherence", "spatial_coherence", "temporal_consistency",
                "phase_tv", "phase_smoothness")
RATE_TERMS = ("rate", "rate_error")
```

`RATE_RANGE` is deleted: the estimator's grid bounds the rate.

- [ ] **Step 2: Rewrite the class**

Imports: add `from neural_methods.model.cardioconv.rate_estimator import RateEstimator, rate_terms` and make the utils import `from neural_methods.model.cardioconv.utils import shrink`.

```python
class CardioConv(nn.Module):
    REGULARISERS = PULSATILITY_PRIORS + PHASE_PRIORS + RATE_TERMS
    NEEDS_RATE_TARGET = True

    def __init__(self, in_channels: int, traces: tuple, fs: float, pulsatility_masker: bool = True,
                 phase_masker: bool = True, trace_independent: bool = True,
                 supervise_masks: bool = True, rate_tolerance_bpm: float = 6.0):
        """Definition of CardioConv.

        Args:
          in_channels: the number of input channels.
          traces: the traces to predict, in the order of the output's axis 1.
          fs: frame rate of the clip, in Hz.
          pulsatility_masker: learn the pulsatility mask; ``False`` is the
            ablation, a mask of ones.
          phase_masker: learn the per-trace masks; ``False`` is the ablation,
            masks of ones, which leaves the traces nothing spatial to differ by.
          trace_independent: mix each trace from its own candidates only.
          supervise_masks: let the statistics head's pooled power train the masks.
          rate_tolerance_bpm: in training, a sample whose estimated rate is
            further than this from its target skips everything after the
            estimator. Must be inside ``BUFFER_BPM``.
        """
        super().__init__()
        self.traces = tuple(traces)
        self.fs = fs
        self.rate_tolerance_bpm = rate_tolerance_bpm
        n_traces = len(self.traces)
        self.rate_estimator = RateEstimator(in_channels, fs)
        self.wavelet_transformer = WaveletTransformer(fs, N_HARMONICS, buffer_bpm=BUFFER_BPM)
        self.pulsatility_masker = (
            PulsatilityMasker(self.wavelet_transformer.n_scales) if pulsatility_masker else None)
        self.phase_masker = PhaseMasker(n_traces) if phase_masker else None
        self.channel_predictor = ChannelPredictor()
        self.channel_mixer = ChannelMixer(in_channels, n_traces, trace_independent)
        self.statistics_head = StatisticsHead(in_channels, n_traces, N_HARMONICS,
                                              supervise_masks=supervise_masks)
        self.sample_mask = None

    def output_layers(self):
        """Each trace's activation-free mean readout, in traces order."""
        return tuple(self.statistics_head.means)

    def regularisers(self) -> dict:
        return self._regularisers

    def forward(self, video: torch.Tensor, target_rate: torch.Tensor = None) -> torch.Tensor:
        """``(B, C_in, T, H, W)`` raw clip, and in training the ``(B,)`` target
        bpm (NaN where a sample has none) -> ``(B, S, T)``. Sets
        ``sample_mask``: the samples whose output counts."""
        require_min_frame("CardioConv", MIN_FRAME, video.shape[-2], video.shape[-1])
        present = reduce(video.abs(), "b c t h w -> b c", "max") > 0
        small = shrink(video, DOWNSCALE)
        rate, logits = self.rate_estimator(small, present)
        zero = small.new_zeros(())

        if target_rate is None:
            passed = torch.ones_like(rate, dtype=torch.bool)
            run = passed
            terms = dict.fromkeys(RATE_TERMS, zero)
        else:
            error = (rate - target_rate).abs()
            passed = torch.isfinite(target_rate) & (error <= self.rate_tolerance_bpm)
            run = passed
            if not bool(passed.any()):
                # Nothing passed: run the closest sample anyway, masked out of
                # the loss, so every parameter is in the graph on every step.
                run = torch.zeros_like(passed)
                run[torch.nan_to_num(error, nan=float("inf")).argmin()] = True
            terms = rate_terms(logits, rate, target_rate, self.rate_estimator.bins)
        self.sample_mask = passed

        index = run.nonzero().squeeze(1)
        subset, priors = self.predict(small[index], present[index], rate[index])
        out = small.new_zeros(video.shape[0], len(self.traces), video.shape[2])
        out[index] = subset
        self._regularisers = {trace: {**own, **terms} for trace, own in zip(self.traces, priors)}
        return out

    def predict(self, small: torch.Tensor, present: torch.Tensor, rate: torch.Tensor) -> tuple:
        """Everything after the estimator, on the samples that run: ``(b, C, T, h, w)``
        shrunk clip, ``(b, C)`` presence and ``(b,)`` bpm -> ``(b, S, T)`` and
        each trace's mask priors, ``[{term: ()}]`` in traces order."""
        field = self.wavelet_transformer(small, rate)
        power = field.abs()
        zero = small.new_zeros(())

        if self.pulsatility_masker is None:
            pulsatility = torch.ones_like(small)
            shared = dict.fromkeys(PULSATILITY_PRIORS, zero)
        else:
            pulsatility, shared = self.pulsatility_masker(power, present)
        if self.phase_masker is None:
            phase_masks = small.new_ones(*small.shape, len(self.traces))
            per_trace = [dict.fromkeys(PHASE_PRIORS, zero) for _ in self.traces]
        else:
            phase_masks, per_trace = self.phase_masker(field.angle(), present)
        # The pulsatility priors belong to every trace alike, so each trace
        # reports them: the trainer's total is the mean over traces.
        priors = [{**shared, **own} for own in per_trace]

        candidates = self.channel_predictor(small, pulsatility, phase_masks)
        mixed = self.channel_mixer(candidates)
        out = self.statistics_head(mixed, power, small, pulsatility, phase_masks, rate, present)
        return out, priors
```

- [ ] **Step 3: The wrapper in `src/models.py`**

Add `from src.signal_transforms import is_cardiac` beside the other `src` imports. Replace `label_heart_rate` (the whole function, lines 89 to 105) with:

```python
def label_rate(batch: dict, traces, fs: float) -> torch.Tensor:
    """``(B,)`` bpm: each sample's heart rate read from its cardiac labels,
    fused when it has more than one.

    The target a backbone declaring ``NEEDS_RATE_TARGET`` (CardioConv) trains
    its rate estimator against, and gates on. It is computed in training mode
    only; nothing from the label reaches the model at test time. The rate is
    the evaluation's own (``src.evaluation.rate``): every present cardiac
    trace is cleaned and its spectrum taken, two or more are fused the way
    the evaluation's ``FUSED`` source is, and the largest in-band bin is the
    rate. A sample with no cardiac trace present gets NaN, which the backbone
    leaves out of its terms and its gate.
    """
    cardiac = [t for t in traces if is_cardiac(t)]
    rates = []
    for i in range(len(batch["labels"][traces[0]])):
        powers, freqs = [], None
        for t in cardiac:
            if bool(batch["label_mask"][t][i]):
                trace = batch["labels"][t][i].detach().float().cpu().numpy()
                freqs, power = rate.spectrum(rate.clean(trace, fs), fs)
                powers.append(power)
        if not powers:
            rates.append(float("nan"))
            continue
        fused = powers[0] if len(powers) == 1 else rate.fuse(freqs, powers)
        rates.append(rate.rate_of(freqs, fused))
    return torch.tensor(rates, dtype=torch.float32)
```

Replace `MultiTraceModel.forward` with:

```python
    def forward(self, batch: dict) -> dict:
        video = self.prepare_frames(batch)
        if self.training and getattr(self.backbone, "NEEDS_RATE_TARGET", False):
            target = label_rate(batch, self.traces, self.backbone.fs).to(video.device)
            out = self.backbone(video, target_rate=target)
        else:
            out = self.forward_video(video)
        predictions = {trace: out[:, i] for i, trace in enumerate(self.traces)}
        result = {**batch, "predictions": predictions,
                  "regularisers": self.collect_regularisers()}
        # A backbone that gates samples says which outputs count; the loss
        # already masks a trace per sample, so the gate joins that mask.
        sample_mask = getattr(self.backbone, "sample_mask", None)
        if sample_mask is not None:
            result["label_mask"] = {trace: mask & sample_mask
                                    for trace, mask in batch["label_mask"].items()}
        return result
```

Add to the `MultiTraceModel` class docstring, after the sentence ending "from ``regularisers()`` after each forward.":

```
    A backbone that declares ``NEEDS_RATE_TARGET`` takes
    ``forward(video, target_rate=None)`` and is handed, in training mode
    only, each sample's label rate (:func:`label_rate`); one that sets a
    ``sample_mask`` ``(B,)`` after its forward has that mask joined into
    every trace's ``label_mask``, so a sample it chose not to predict is
    out of the loss.
```

Replace `_build_cardioconv` with:

```python
def _build_cardioconv(cfg: CardioConvConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "CardioConv", cardioconv.MIN_FRAME)
    if not cfg.RATE_TOLERANCE_BPM < cardioconv.BUFFER_BPM:
        raise ConfigError(
            f"CardioConv: RATE_TOLERANCE_BPM must be inside the wavelet buffer of "
            f"{cardioconv.BUFFER_BPM:g} bpm either side of each harmonic, got "
            f"{cfg.RATE_TOLERANCE_BPM:g}")
    backbone = cardioconv.CardioConv(
        in_channels=len(interface.CHANNELS), traces=interface.TRACES, fs=interface.FS,
        pulsatility_masker=cfg.PULSATILITY_MASKER, phase_masker=cfg.PHASE_MASKER,
        trace_independent=cfg.TRACE_INDEPENDENT, supervise_masks=cfg.SUPERVISE_MASKS,
        rate_tolerance_bpm=cfg.RATE_TOLERANCE_BPM)
    return _wrap(backbone, interface, cfg)
```

Grep `NEEDS_HEART_RATE`, `label_heart_rate`, `heart_rate=` across `src/` and `neural_methods/` and confirm zero hits.

- [ ] **Step 4: The config class in `src/model_config.py`**

Replace `CardioConvConfig` with:

```python
@dataclass
class CardioConvConfig(ModelConfig):
    """CardioConv's ablations (every one ``True`` in the model as designed)
    and its training gate."""
    PULSATILITY_MASKER: bool = True   # learn the pulsatility mask; false is a mask of ones
    PHASE_MASKER: bool = True         # learn the per-trace masks; false is masks of ones
    TRACE_INDEPENDENT: bool = True    # mix each trace from its own candidates only
    SUPERVISE_MASKS: bool = True      # the statistics head's pooled power trains the masks
    RATE_TOLERANCE_BPM: float = 0.0   # a sample whose estimated rate is further than this
                                      # from its label rate skips the stages after the estimator

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        super().validate(interface, where)
        other = [t for t in interface.TRACES if not is_cardiac(t)]
        if other:
            raise ConfigError(
                f"{where}: CardioConv reads the clip at the harmonics of the heart "
                f"rate, so it predicts cardiac traces only; TRACES has {other}")
        if self.RATE_TOLERANCE_BPM <= 0:
            raise ConfigError(
                f"{where}: RATE_TOLERANCE_BPM must be positive, got {self.RATE_TOLERANCE_BPM}")
```

- [ ] **Step 5: The three config files**

In each of the three CardioConv YAML files, delete the three `# LABEL LEAKAGE ...` comment lines at the top, and make the `MODEL` section:

```yaml
MODEL:
  NAME: CardioConv
  PULSATILITY_MASKER: true
  PHASE_MASKER: true
  TRACE_INDEPENDENT: true
  SUPERVISE_MASKS: true
  RATE_TOLERANCE_BPM: 6.0        # training gate on the learned heart rate: the
                                 # evaluation's SNR half-width, inside the 10 bpm wavelet buffer
  REGULARISATION:                # the mask priors, and the rate estimator's terms
    SPARSITY: 0.022
    PULSATILITY_TV: 0.44
    PULSATILITY_SMOOTHNESS: 1.0
    HARMONIC_COHERENCE: 0.058
    SPATIAL_COHERENCE: 0.018
    TEMPORAL_CONSISTENCY: 0.031
    PHASE_TV: 0.50
    PHASE_SMOOTHNESS: 1.0
    RATE: 1.0                    # cross-entropy to the fused label rate
    RATE_ERROR: 0.1              # L1 in bpm; the logged column is 0.1 x bpm
```

In `configs/_model_config_template.yaml` line 22, append `RATE_TOLERANCE_BPM` to CardioConv's switch list; at line 35, the parenthetical becomes `(CardioConv's mask priors and rate terms; no`.

- [ ] **Step 6: The smoke script**

`SCRATCH/check_model.py`:

```python
"""Throwaway: one training step and one eval forward of the wrapped model on random data,
covering every gate branch, a sample without labels, eval mode and bfloat16 autocast."""
import sys

import torch

sys.path.insert(0, r"C:\Users\20759193\source\repos\remote-physiology")
from neural_methods.model.cardioconv.utils import shrink  # noqa: E402
from src.model_config import load_config  # noqa: E402
from src.models import build_model  # noqa: E402
from src.trainer import init_output_bias  # noqa: E402

CONFIG = r"configs/experiment_configs/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H72W72.yaml"
interface, model_cfg, _ = load_config(CONFIG)
model = build_model(model_cfg, interface)
init_output_bias(model, interface)
torch.manual_seed(0)
B, T = 4, 300


def batch(rate_bpm):
    t = torch.arange(T) / interface.FS
    pulse = torch.sin(2 * torch.pi * rate_bpm / 60 * t)
    frames = {ch: 120 + 5 * pulse + torch.randn(B, T, 72, 72) for ch in interface.CHANNELS}
    labels = {"ABP": 90 + 20 * pulse.expand(B, T).clone(), "CVP": 8 + 3 * pulse.expand(B, T).clone()}
    label_mask = {"ABP": torch.tensor([True, True, True, False]),
                  "CVP": torch.tensor([True, True, False, False])}
    return {"frames": frames, "labels": labels, "label_mask": label_mask,
            "channel_mask": {ch: torch.ones(B, dtype=torch.bool) for ch in interface.CHANNELS},
            "label_stats": {}, "metadata": {}}


def check(out, name):
    pred = torch.stack(list(out["predictions"].values()))
    assert pred.shape == (2, B, T), pred.shape
    assert torch.isfinite(pred).all(), f"{name}: NaN in predictions"
    for trace, terms in out["regularisers"].items():
        for term, value in terms.items():
            assert torch.isfinite(value), f"{name}: {trace}/{term} is {value}"
    rate_terms = {k: round(float(v), 3) for k, v in out["regularisers"]["ABP"].items()
                  if k.startswith("rate")}
    print(f"{name}: sample_mask={model.backbone.sample_mask.tolist()}, rate terms={rate_terms}")


model.train()
# 1. ordinary step: the untrained estimator is unlikely to be within 6 bpm, so this
#    is also the "nothing passed" branch; every parameter must get a gradient.
out = model(batch(72.0))
check(out, "train, untrained")
total = sum(v for terms in out["regularisers"].values() for v in terms.values())
total.backward()
missing = [n for n, p in model.named_parameters() if p.grad is None]
assert not missing, f"no gradient: {missing}"
print("every parameter has a gradient")

# 2. force the gate open: hand the backbone its own estimate as the target.
video = model.prepare_frames(batch(72.0))
with torch.no_grad():
    rate, _ = model.backbone.rate_estimator(
        shrink(video, 5), torch.ones(B, len(interface.CHANNELS), dtype=torch.bool))
target = rate.clone()
target[3] = float("nan")                                     # the unlabelled sample
out = model.backbone(video, target_rate=target)
assert model.backbone.sample_mask.tolist() == [True, True, True, False], model.backbone.sample_mask
assert torch.isfinite(out).all() and (out[3] == 0).all()
print("gate open for labelled samples, closed for the unlabelled one; its output is zero")

# 3. eval mode: no target, every sample runs.
model.eval()
with torch.no_grad():
    out = model(batch(72.0))
check(out, "eval")
assert model.backbone.sample_mask.all()

# 4. bfloat16 autocast on CPU: the FFTs must have cast to float32 themselves.
model.train()
with torch.autocast("cpu", dtype=torch.bfloat16):
    out = model(batch(72.0))
check(out, "train, bfloat16 autocast")
print("ok")
```

- [ ] **Step 7: Run it**

Run: `uv run python "SCRATCH/check_model.py"`
Expected: every assertion passes; the first forward prints a sample mask of all `False` (or whichever samples the untrained estimator happens to hit) and finite rate terms; "every parameter has a gradient"; the forced-open forward prints the `[True, True, True, False]` mask; eval runs with all `True`; bfloat16 runs. If the first forward has every sample passing by luck, the "nothing passed" branch is still exercised by replacing `batch(72.0)` with a target the backbone cannot meet: call `model.backbone(video, target_rate=torch.full((B,), 500.0))` and assert the mask is all `False` and the output finite.

- [ ] **Step 8: Lint**

Run: `uv run ruff check neural_methods/model/cardioconv src`
Expected: `All checks passed!`

- [ ] **Step 9: Commit**

```bash
git add neural_methods/model/cardioconv/cardioconv.py src/models.py src/model_config.py configs/experiment_configs/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H72W72.yaml configs/hpc_configs_200/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H200W200.yaml configs/hpc_configs_300/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H300W300.yaml configs/_model_config_template.yaml
git commit -m "feat(cardioconv): learned heart rate, per-sample training gate, fused label-rate target

The wrapper hands CardioConv the fused rate of each sample's cardiac labels
in training only; the estimator trains against it and a sample more than
RATE_TOLERANCE_BPM off skips every stage after the estimator and leaves the
loss through its label mask. No label reaches the model at test time.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Docs

**Files:**
- Modify: `docs/adding_a_model.md:362-366` (the regulariser bullet), `:381-386` (the stopgap paragraph)
- Modify: `docs/plans/2026-09-18-model-regularisation-design.md:13-16` (the split table)

- [ ] **Step 1: `docs/adding_a_model.md`**

Delete the paragraph beginning "One thing in CardioConv is a stopgap and not part of the form:" through "Do not build on it." (lines 381 to 386).

Replace the `regularisers()` bullet (lines 362 to 366) with:

```markdown
- **`regularisers()`** returns `{trace: {term: () tensor}}`, every declared
  term under every trace. A term that belongs to one trace (a mask prior on
  that trace's mask) goes under it; a term the traces share goes, identical,
  under each — the total is the mean over traces, so it counts once. A term
  is usually label-free; it may instead be supervised by a target the
  wrapper supplies, as CardioConv's rate terms are by `label_rate`, in
  training mode only (`MultiTraceModel`'s docstring has the contract).
```

- [ ] **Step 2: The regularisation design note**

In the table under "The split", change the Regulariser row's "Depends on" cell from `a model's own intermediate tensors` to `a model's own intermediate tensors, or a target the wrapper supplies in training (CardioConv's rate)`.

- [ ] **Step 3: Commit**

```bash
git add docs/adding_a_model.md docs/plans/2026-09-18-model-regularisation-design.md
git commit -m "docs: CardioConv's rate is learned; a model term may be supervised

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Remove ptwt and pywavelets, hand over the run

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv remove`)

**Interfaces:** none.

- [ ] **Step 1: Confirm nothing imports them**

Run: `rg -n "ptwt|pywt|PyWavelets" --glob "!.venv/**" --glob "!vendor/**" --glob "!runs/**" --glob "!external/**" .`
Expected: hits only in `pyproject.toml` and `uv.lock` (and this plan and the spec).

- [ ] **Step 2: Ask the user before touching the venv**

The removal syncs the shared environment. Stop here and ask: "Ready to run `uv remove ptwt pywavelets`; it re-syncs the shared venv, which can disturb a running job. Go ahead?" Proceed only on a yes. If the answer is no or absent, leave both dependencies in place, say so in the handover, and finish the rest of this task.

- [ ] **Step 3: Remove**

```bash
uv remove ptwt pywavelets
```

Expected: `pyproject.toml` loses the `ptwt>=1.0.1` line (and `pywavelets` if listed directly), `uv.lock` updates, no error.

- [ ] **Step 4: Confirm the model still imports**

Run: `uv run python -c "from src.models import build_model; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: drop ptwt and pywavelets; the Morlet bank is closed-form torch

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 6: Delete the throwaway scripts**

```bash
rm "SCRATCH/check_bank.py" "SCRATCH/check_estimator.py" "SCRATCH/check_model.py"
```

- [ ] **Step 7: Hand over the proof run**

Give the user this command and the comparison recipe; the user runs it (memory: `user-runs-verification-themselves`):

```bash
uv run python scripts/run.py --datasets neckflix \
  --config configs/experiment_configs/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H72W72.yaml \
  --epochs 30 --batch-size 4 --num-workers 4 \
  --test-participant-dataset neckflix --test-participant-id 33 \
  --runs-dir runs/neckflix_cardioconv_rate
```

Comparison against `runs/neckflix_cardioconv/CARDIOCONV_NECKFLIX.33_202609250025`:
- `losses.csv`: `ABP/rate_error / 0.1` is the training bpm error per epoch; `ABP/ccc` and `CVP/ccc` against the baseline's; `seconds` against the baseline's ~80 s.
- `epoch_30/test_records/neckflix/<recording>/<camera>/rates.csv`: `ref_hr` vs `pred_hr` per source, against the baseline's.
- `signals.csv` per recording: the level errors.

---

## Self-review

**Spec coverage.** Shrink (Task 1), estimator and its two terms (Task 2), bank per harmonic with fixed pad and matched convention (Task 1), gate with the always-run fallback and `sample_mask` (Task 3), `label_rate` with fusion and training-only call (Task 3), config switch validated positive in the class and under the buffer in the builder (Task 3), three config files and template (Task 3), docs (Task 4), dependency removal gated on the user (Task 5), proof run (Task 5). Out-of-scope items untouched.

**Placeholders.** None; every step is literal code or a literal command.

**Type consistency.** `WaveletTransformer.forward(small, rate)` in Task 1 is what `CardioConv.predict` calls in Task 3; `RateEstimator.forward -> (rate, logits)` and `rate_terms(logits, rate, target, bins)` in Task 2 match their calls in Task 3; `label_rate(batch, traces, fs)` matches the wrapper; `cardioconv.BUFFER_BPM` is defined in Task 3 Step 1 and read in Step 3.

**Review focus.** Items 1 and 5 are checked in Task 1 Step 5 (`T=128`) and Task 3 Step 7 (bfloat16); items 2, 3 and 4 in Task 3 Step 7.
