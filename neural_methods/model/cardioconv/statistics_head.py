"""The level and spread of each trace, and the waveform rescaled to them."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, reduce

from neural_methods.model.cardioconv.utils import EPS, softplus_inverse

#: Shrinks the mean readout's initial weights, so the bias the trainer seeds
#: with the trace's physiological prior is the starting prediction and the
#: features only a small perturbation around it. Not zero, which would cut
#: the gradient to the hidden layer (and through it to the masks) at step 0.
#: It matters most for a low-level trace: a perturbation that is nothing to
#: ABP at 90 mmHg is a large fraction of CVP at 8.
READOUT_WEIGHT_SCALE = 0.1


class StatisticsHead(nn.Module):
    """Where the waveform's physical level and amplitude come from.

    The waveform path is shape-only by design — min-max scaled pixels under
    soft masks — so what it discards is read back here: per channel the
    shrunk clip's min, max and mean, the wavelet power pooled under each
    trace's mask per harmonic, the mask's mass and the channel's presence,
    and the heart rate. Those summaries arrive on very different scales and
    bit depths (8-bit colour, 16-bit infrared and depth, unbounded power), so
    intensity and power go through ``log1p``, which lands them all in the same
    band whatever the source, and the heart rate is divided by 100. No batch
    or layer normalisation: either would erase the differences between
    samples this head exists to read.

    Per trace a small MLP gives a mean and a spread. The mean is the trace's
    readout, a single-bias linear layer. The spread is relative,
    ``(1 + |mean|) * softplus(raw)``, starting at ``relative_spread`` of that,
    so it needs no prior of its own and starts in proportion for a trace at
    any level (the ``1 +`` keeps it alive for a z-scored trace, whose mean is
    zero); the mean is detached there, so the spread never pulls on it. The
    output is the mixed waveform z-scored over the window, times the spread,
    plus the mean, which makes the mixer's own numeric scale irrelevant.

    Mask mass is always read detached: its gradient could only say "make the
    mask bigger", against the sparsity and total-variation priors. Pooled
    power is normalised by the mask's area, so its gradient says "put the
    mask on better pixels"; ``supervise_masks=False`` detaches that too and
    leaves the head a read-only probe of the masks.

    Args:
      in_channels: video channels ``C``.
      n_traces: traces ``S``.
      n_harmonics: harmonic groups the scale axis folds into.
      hidden: width of each trace's hidden layer.
      supervise_masks: let the pooled-power path train the masks.
      relative_spread: the spread the head starts at, as a fraction of the mean.
    """

    def __init__(self, in_channels: int, n_traces: int, n_harmonics: int, hidden: int = 64,
                 supervise_masks: bool = True, relative_spread: float = 0.15):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.supervise_masks = supervise_masks
        n_features = in_channels * (3 + n_traces * (n_harmonics + 1) + 1) + 1
        self.trunks = nn.ModuleList(
            nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU()) for _ in range(n_traces))
        self.means = nn.ModuleList(nn.Linear(hidden, 1) for _ in range(n_traces))
        self.spreads = nn.ModuleList(nn.Linear(hidden, 1) for _ in range(n_traces))
        with torch.no_grad():
            for mean, spread in zip(self.means, self.spreads):
                mean.weight.mul_(READOUT_WEIGHT_SCALE)
                spread.weight.mul_(READOUT_WEIGHT_SCALE)
                spread.bias.fill_(softplus_inverse(relative_spread))

    def features(self, power, small, pulsatility, phase_masks, heart_rate, present) -> torch.Tensor:
        """The ``(B, n_features)`` summary; absent channels contribute zeros."""
        here = rearrange(present.to(small.dtype), "b c -> b c 1")
        intensity = torch.stack([reduce(small, "b c t h w -> b c", kind)
                                 for kind in ("min", "max", "mean")], dim=-1)
        intensity = torch.log1p(intensity.clamp_min(0.0)) * here                      # (B, C, 3)

        weight = rearrange(pulsatility, "b c t h w -> b c t h w 1") * phase_masks      # (B, C, T, h, w, S)
        mass = reduce(weight.detach(), "b c t h w s -> b c s", "mean") * here
        if not self.supervise_masks:
            weight = weight.detach()
        area = reduce(weight, "b c t h w s -> b c s 1", "sum").clamp_min(EPS)
        pooled = einsum(power, weight, "b c t h w k, b c t h w s -> b c s k") / area
        harmonics = reduce(pooled, "b c s (n j) -> b c s n", "mean", n=self.n_harmonics)
        harmonics = torch.log1p(harmonics) * rearrange(here, "b c 1 -> b c 1 1")

        return torch.cat([
            rearrange(intensity, "b c f -> b (c f)"),
            rearrange(harmonics, "b c s n -> b (c s n)"),
            rearrange(mass, "b c s -> b (c s)"),
            rearrange(here, "b c 1 -> b c"),
            rearrange(heart_rate.to(small.dtype), "b -> b 1") / 100.0,
        ], dim=-1)

    def forward(self, mixed, power, small, pulsatility, phase_masks, heart_rate, present) -> torch.Tensor:
        """``(B, S, T)`` shape-only waveforms -> ``(B, S, T)`` in each trace's
        physical units. The rest are the stages' outputs: ``(B, C, T, h, w, K)``
        power, ``(B, C, T, h, w)`` shrunk clip and pulsatility mask,
        ``(B, C, T, h, w, S)`` trace masks, ``(B,)`` bpm, ``(B, C)`` presence."""
        features = self.features(power, small, pulsatility, phase_masks, heart_rate, present)
        hidden = [trunk(features) for trunk in self.trunks]
        mean = torch.cat([layer(h) for layer, h in zip(self.means, hidden)], dim=-1)          # (B, S)
        spread = torch.cat([layer(h) for layer, h in zip(self.spreads, hidden)], dim=-1)
        spread = (1.0 + mean.detach().abs()) * F.softplus(spread)
        shape = mixed - mixed.mean(dim=-1, keepdim=True)
        shape = shape / shape.std(dim=-1, keepdim=True).clamp_min(EPS)
        return shape * rearrange(spread, "b s -> b s 1") + rearrange(mean, "b s -> b s 1")
