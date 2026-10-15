"""Mask-pooled per-frame features -> one shape-only waveform per trace."""

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce

from neural_methods.model.cardiohydra.mamba_block import MambaBlock
from neural_methods.model._shared_modules.utils import (
    EPS, KINEMATIC_FEATURES, kinematic_features, mask_area,
)

#: Pooled features per channel and harmonic: power, the phase's cosine and
#: sine, and the kinematics (wave amplitude, wavenumber, speed and the
#: heading's two components).
PER_HARMONIC = 3 + len(KINEMATIC_FEATURES)

#: Pooled features per channel beside those: intensity, mask mass, presence.
PER_CHANNEL = 3


class SSMPredictor(nn.Module):
    """Each trace's waveform, from what its mask sees at every frame.

    Per channel, trace and frame, the pixels are averaged under
    ``pulsatility * trace mask``: the wavelet power and the unit phase per
    harmonic, the kinematics per harmonic (amplitude, wavenumber, signed
    speed, heading), and the clip's intensity. Power, amplitude,
    speed and intensity go through ``log1p``, so 8-bit colour and 16-bit
    infrared or depth land in one band. The mask's mass and the channel's
    presence complete the frame's features. Each trace then has its own
    trunk: a linear embedding, bidirectional Mamba3 blocks, and a linear
    readout per frame.

    Intensity is the one feature with the clip's full bandwidth. Everything
    read off the field is confined to the harmonics of the mean heart rate,
    which attenuates an irregular beat; the intensity path is what lets the
    trunk see one.

    The averages are normalised by the mask's area and stay attached, so the
    waveform loss trains both maskers through the weights. Mass is read
    detached: its gradient could only say "make the mask bigger".

    Args:
      in_channels: video channels ``C``.
      n_traces: traces ``S``, one trunk each.
      n_harmonics: harmonic groups the scale axis folds into.
      d_model: width of each trunk.
      n_blocks: Mamba blocks per trunk.
      d_state, expand, headdim, dt_init: each block's ``BiMamba3`` arguments.
    """

    def __init__(self, in_channels: int, n_traces: int, n_harmonics: int, d_model: int,
                 n_blocks: int, d_state: int, expand: int, headdim: int, dt_init: float):
        super().__init__()
        self.n_harmonics = n_harmonics
        n_features = in_channels * (n_harmonics * PER_HARMONIC + PER_CHANNEL)
        self.embeds = nn.ModuleList(nn.Linear(n_features, d_model) for _ in range(n_traces))
        self.trunks = nn.ModuleList(
            nn.Sequential(*[MambaBlock(d_model, d_state, expand, headdim, dt_init)
                            for _ in range(n_blocks)])
            for _ in range(n_traces))
        self.heads = nn.ModuleList(nn.Linear(d_model, 1) for _ in range(n_traces))

    def features(self, field, power, clip, weight, weight_fine, kinematics,
                 present) -> torch.Tensor:
        """The ``(B, S, T, n_features)`` per-frame summary under each trace's mask."""
        n = self.n_harmonics
        centre = rearrange(field, "b c t h w (n j) -> b c t h w n j", n=n).mean(dim=-1)
        fields = torch.cat([
            torch.log1p(reduce(power, "b c t h w (n j) -> b c t h w n", "mean", n=n)),
            *kinematic_features(kinematics),
            centre.real,
            centre.imag,
        ], dim=-1)                                                                # (B, C, T, h, w, 8N)

        area = rearrange(mask_area(weight), "b c s -> b s 1 c 1")
        pooled = einsum(fields, weight, "b c t h w f, b c h w s -> b s t c f") / area
        scalars, real, imag = pooled.split([(1 + len(KINEMATIC_FEATURES)) * n, n, n], dim=-1)
        # The pooled complex mean, as a unit vector: its angle is the phase.
        # The floor sits inside the root, whose gradient at zero (an absent
        # channel) is not finite.
        length = torch.sqrt(real ** 2 + imag ** 2 + EPS ** 2)
        # The clip is pooled on its own grid, under the masks read there.
        intensity = (rearrange(einsum(torch.log1p(clip.clamp_min(0.0)), weight_fine,
                                      "b c t h w, b c h w s -> b s t c"), "b s t c -> b s t c 1")
                     / rearrange(mask_area(weight_fine), "b c s -> b s 1 c 1"))

        here = rearrange(present.to(pooled.dtype), "b c -> b 1 1 c 1")
        mass = rearrange(reduce(weight.detach(), "b c h w s -> b s c", "sum"),
                         "b s c -> b s 1 c 1").expand(-1, -1, pooled.shape[2], -1, -1) * here   # share of the pulsatile pixels
        out = torch.cat([scalars, real / length, imag / length, intensity, mass,
                         here.expand_as(mass)], dim=-1)
        return rearrange(out, "b s t c f -> b s t (c f)")

    def forward(self, field, power, clip, weight, weight_fine, kinematics,
                present) -> torch.Tensor:
        """The stages' outputs -> ``(B, S, T)`` shape-only waveforms:
        ``(B, C, T, h, w, K)`` complex field and its power and the kinematics
        ``{name: (B, C, T, h, w, N)}``, with the pooling weight ``(B, C, h, w, S)``
        on their grid; the ``(B, C, T, H, W)`` clip with the weight
        ``(B, C, H, W, S)`` on its own; ``(B, C)`` presence."""
        features = self.features(field, power, clip, weight, weight_fine, kinematics, present)
        waveforms = [head(trunk(embed(own)))
                     for embed, trunk, head, own
                     in zip(self.embeds, self.trunks, self.heads, features.unbind(dim=1))]
        return rearrange(torch.cat(waveforms, dim=-1), "b t s -> b s t")
