"""Heart-rate-conditioned, pixel-wise complex Morlet transform, in closed form."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from neural_methods.model.cardioconv.utils import buffer_steps, centre_frequencies, shrink, upscale


class WaveletTransformer(nn.Module):
    """The complex wavelet field of a clip at the harmonics of its heart rate.

    The transform is taken on the clip shrunk by ``downscale`` (an area
    average, :func:`~neural_methods.model.cardioconv.utils.shrink`) and the
    field is upscaled bilinearly back to the clip's frame, so every stage
    after this one works at the input's resolution while the transform, the
    expensive step, runs on ``downscale ** 2`` fewer pixels.

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
      downscale: the area shrink before the transform; 1 is none.
    """

    def __init__(self, fs: float, n_harmonics: int, bandwidth: float = 1.5, centre: float = 1.0,
                 buffer_bpm: float = 10.0, step_bpm: float = 1.0, min_bpm: float = 30.0,
                 downscale: int = 1):
        super().__init__()
        self.fs = fs
        self.downscale = downscale
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

    def forward(self, video: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, H, W)`` clip and ``(B,)`` bpm -> the complex field
        ``(B, C, T, H, W, K)``, ``K = n_scales``, harmonic-major."""
        small = shrink(video, self.downscale)
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
            # Cloned: the crop is a view, and a view would keep the whole padded
            # inverse transform alive until the harmonics are joined.
            fields.append(torch.fft.ifft(product, dim=-1)[..., self.pad:self.pad + t].clone())
        field = torch.cat(fields, dim=2)                                                # (B, P, K, T)
        field = rearrange(field, "b (c h w) k t -> b c t h w k", c=c, h=h, w=w)
        return upscale(field, video.shape[-2:])
