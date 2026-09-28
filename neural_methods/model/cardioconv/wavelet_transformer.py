"""Heart-rate-conditioned, pixel-wise continuous wavelet transform."""

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ptwt import cwt

from neural_methods.model.cardioconv.utils import buffer_steps, heart_rate_scales


class WaveletTransformer(nn.Module):
    """The complex wavelet field of a clip at the harmonics of its heart rate.

    The clip is shrunk spatially, each pixel's trace loses its mean, and a
    complex Morlet transform is taken at the scales
    :func:`~neural_methods.model.cardioconv.utils.heart_rate_scales` puts
    around each harmonic. The field stays complex: power is its ``abs()`` and
    phase its ``angle()``, each taken by the stage that wants it. Nothing here
    is learned, and nothing carries a gradient.

    Args:
      fs: frame rate of the clip, in Hz.
      n_harmonics: harmonics of the heart rate the scales cover.
      downscale: the spatial shrink before the transform.
      wavelet: the pywt name of the complex wavelet.
      buffer_bpm, step_bpm: the scales sit ``+-buffer_bpm`` around each
        harmonic, ``step_bpm`` apart.
    """

    def __init__(self, fs: float, n_harmonics: int, downscale: int, wavelet: str = "cmor1.5-1.0",
                 buffer_bpm: float = 10.0, step_bpm: float = 1.0):
        super().__init__()
        self.fs = fs
        self.n_harmonics = n_harmonics
        self.downscale = downscale
        self.buffer_bpm = buffer_bpm
        self.step_bpm = step_bpm
        # The wavelet object, not its name, so ptwt conjugates a complex one correctly.
        self.wavelet = pywt.ContinuousWavelet(wavelet)
        self.n_scales = n_harmonics * buffer_steps(buffer_bpm, step_bpm)

    @torch.no_grad()
    def forward(self, video: torch.Tensor, heart_rate: torch.Tensor) -> tuple:
        """``(B, C, T, H, W)`` clip and ``(B,)`` finite bpm -> the complex field
        ``(B, C, T, h, w, K)`` and the shrunk clip ``(B, C, T, h, w)`` it was
        taken from, ``h = H // downscale``."""
        b, c = video.shape[:2]
        small = F.interpolate(rearrange(video.float(), "b c t h w -> (b c) t h w"),
                              scale_factor=1.0 / self.downscale, mode="bilinear", align_corners=False)
        small = rearrange(small, "(b c) t h w -> b c t h w", b=b)
        height, width = small.shape[-2:]

        fields = []
        for clip, rate in zip(small, heart_rate.tolist()):
            scales = heart_rate_scales(rate, self.fs, self.n_harmonics, self.wavelet.center_frequency,
                                       self.buffer_bpm, self.step_bpm)
            traces = rearrange(clip, "c t h w -> (c h w) t")
            traces = traces - traces.mean(dim=-1, keepdim=True)
            coefficients, _ = cwt(traces, scales, self.wavelet, sampling_period=1.0 / self.fs)
            fields.append(rearrange(coefficients.to(torch.complex64), "k (c h w) t -> c t h w k",
                                    c=c, h=height, w=width))
        return torch.stack(fields), small
