"""Wavelet magnitude -> where each channel pulses, as one spatial attention map per window."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch.utils.checkpoint import checkpoint

from neural_methods.model.cardioconv.utils import EPS, present_mean


class PulsatilityMasker(nn.Module):
    """Where the clip pulses at the heart rate: one distribution over the
    pixels of each channel, per window, expanded over the frames.

    Physics decides, the network nudges. The *pulsatility index* is the log
    of the field's band power at the harmonics over the pixel's variance in
    the window: how much of what the pixel does is the heartbeat. It is the
    logit, at unit gain (proportional to the band-power-to-variance ratio:
    a static background gets nothing, a pulsing pixel its share). A
    separable 3-D convolution (scales mixed pointwise, then 3x3 in space,
    then ``temporal_kernel`` frames in time, replicate-padded) reads the
    index and the magnitudes at every scale, scaled per sample and channel
    by their mean so 8-bit colour and 16-bit infrared or depth read the same
    and log-compressed, and adds a correction centred over the pixels and
    bounded to ``+-correction_bound`` nats by a tanh: the weights of any two
    pixels may move against each other by a factor of at most
    ``exp(2 correction_bound)`` relative to the index's (e^3, about 20, at
    1.5). The logits are averaged over the frames and a softmax over the
    pixels turns them into weights that sum to one: a mask that cannot
    vanish and cannot be rescaled, and, with the correction bounded, cannot
    be sharper than the index allows, so it covers the pulsing neck and
    leaves the vessels to the phase masks. Every channel goes through the
    same network on its own; the phase is never read.

    Before 2026-10-11 the index entered through a learned gain, the
    correction was unbounded and an entropy prior rewarded concentration:
    the correction grew to 44 nats of spread and the map collapsed to one to
    three pixels per channel (experiment 6b, CardioConv).

    The mask is one map per window, from the frames' mean logit: a vessel
    does not move in ten seconds, and a mask that moved put a slow wander
    into the masked mean. Priors: ``pulsatility_tv``, the spatial total
    variation of the density (``h w`` times the weight, so 1 is uniform);
    ``pulsatility_saturation``, the mean of ``relu(|c| / bound - 1) ** 2``
    over the raw correction ``c``, zero inside the bound and a pull back
    towards it beyond, where the tanh has no gradient left.

    Args:
      n_scales: the wavelet field's scale count.
      correction_bound: the network's largest move of a pixel's logit, in nats.
      hidden: channels after the scale-mixing layer.
      temporal_kernel: frames the temporal smoothing sees; odd.
    """

    def __init__(self, n_scales: int, correction_bound: float, hidden: int = 16,
                 temporal_kernel: int = 15):
        super().__init__()
        self.correction_bound = correction_bound
        self.net = nn.Sequential(
            nn.Conv3d(n_scales + 1, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=(temporal_kernel, 1, 1),
                      padding=(temporal_kernel // 2, 0, 0), padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )

    def forward(self, field: torch.Tensor, small: torch.Tensor, present: torch.Tensor) -> tuple:
        """``(B, C, T, h, w, K)`` complex field, ``(B, C, T, h, w)`` shrunk clip
        and ``(B, C)`` presence -> the mask ``(B, C, T, h, w)`` (one map per
        window, expanded over the frames), summing to one over ``(h, w)``,
        and its priors ``{term: ()}``."""
        b, _, _, h, w, _ = field.shape
        # Features, not a path: the mask trains through the pooling weights
        # downstream. The network is checkpointed, so a full-resolution grid
        # keeps no hidden activations between the passes.
        with torch.no_grad():
            power = field.abs()
            scale = reduce(power, "b c t h w k -> b c 1 1 1 1", "mean").clamp_min(EPS)
            variance = small.float().var(dim=2, keepdim=True)                         # (B, C, 1, h, w)
            band = reduce(power ** 2, "b c t h w k -> b c t h w", "sum")
            # Floors relative to the channel, the band's far below the
            # variance's, so a static pixel (no band power, no variance)
            # scores well under a pulsing one instead of one over one.
            band_floor = 1e-6 * reduce(band, "b c t h w -> b c 1 1 1", "mean")
            variance_floor = 1e-3 * reduce(variance, "b c 1 h w -> b c 1 1 1", "mean")
            index = (torch.log(band + band_floor + EPS)
                     - torch.log(variance + variance_floor + EPS))                    # (B, C, T, h, w)
            x = torch.cat([torch.log1p(power / scale), index[..., None]], dim=-1)
            x = rearrange(x, "b c t h w f -> (b c) f t h w")
        net = checkpoint(self.net, x, use_reentrant=False)
        # One mask per window: the logits are averaged over the frames before
        # the softmax. A mask that moved from frame to frame swept the
        # weighted mean across the clip's intensity gradients and put a slow
        # wander into every candidate waveform (2026-10-10, experiment 2).
        correction = reduce(net, "(b c) 1 t h w -> b c h w", "mean", b=b)
        correction = correction - reduce(correction, "b c h w -> b c 1 1", "mean")
        bound = self.correction_bound
        logits = reduce(index, "b c t h w -> b c h w", "mean") + bound * torch.tanh(correction / bound)
        weight = rearrange(logits, "b c h w -> b c (h w)").softmax(dim=-1)
        mask = rearrange(weight, "b c (h w) -> b c h w", h=h)
        density = mask * (h * w)
        tv = (reduce((density[..., 1:, :] - density[..., :-1, :]).abs(), "b c h w -> b c", "mean")
              + reduce((density[..., 1:] - density[..., :-1]).abs(), "b c h w -> b c", "mean"))
        saturation = reduce(F.relu(correction.abs() / bound - 1.0) ** 2, "b c h w -> b c", "mean")
        priors = {"pulsatility_tv": present_mean(tv, present),
                  "pulsatility_saturation": present_mean(saturation, present)}
        return rearrange(mask, "b c h w -> b c 1 h w").expand(-1, -1, field.shape[2], -1, -1), priors
