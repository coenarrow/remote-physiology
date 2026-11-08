"""``propagation.png``: per channel a grid of arrows, one per block, each
pointing the way the pulse travels there, read from the fall of the
window-averaged arrival phase."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from src.visualise.layout import grey_frame, page_grid
from src.visualise.scores import coarse_weight

#: The box over which the mean phasor is smoothed before its gradient is
#: read, the phase coherence below which a pixel does not vote on the
#: direction, and the arrow's length in grid cells.
PHASE_SMOOTHING = 3
PHASE_COHERENCE_FLOOR = 0.6              # a noise pixel's R on a 10 s window reaches 0.6 (experiment 13)
ARROW_BLOCK = 2                          # grid cells per arrow
ARROW_NEIGHBOURHOOD = 3                  # blocks over which arrows must agree to stay opaque
#: The traces' arrow colours (red, blue, green, orange), in ``TRACES`` order;
#: an arrow fainter than the floor is not drawn.
TRACE_TINTS = np.asarray(((1.0, 0.15, 0.15), (0.15, 0.6, 1.0), (0.2, 1.0, 0.2), (1.0, 0.6, 0.1)))
ARROW_ALPHA_FLOOR = 0.2
ARROW_STYLE = "-|>,head_width=0.3,head_length=0.45"
ARROW_HEAD_PT = 4.0                      # the arrowhead's scale in points (matplotlib's mutation_scale)


def relative_phase(captured: dict, channel_index: int) -> tuple:
    """The fundamental's phase relative to the channel's own pulse, averaged
    over the window: ``(mean phasor (h, w) complex, coherence R (h, w))``.
    The reference is the pulsatility-weighted mean field per frame, as in
    the phase masker; the per-frame relative phasors are averaged, so a
    pixel whose phase holds has ``R`` near 1 and the mean phasor's angle is
    its arrival phase. The phasor is box-averaged (weighted by pulsatility)
    so the arrival is read from the region, not pixel speckle; the
    coherence is the per-pixel one, before smoothing."""
    field = captured["harmonic"][0, channel_index]                           # (T, h, w) complex
    pulse = captured["pulsatility_coarse"][0, channel_index]                 # (h, w)
    reference = (field * pulse).sum(dim=(-2, -1), keepdim=True)              # (T, 1, 1)
    relative = field * reference.conj()
    relative = relative / (relative.abs() + 1e-12)                           # unit phasors
    mean = relative.mean(dim=0)                                              # (h, w) complex
    coherence = mean.abs()
    k = PHASE_SMOOTHING
    weighted = torch.view_as_real(mean * pulse)                              # (h, w, 2)
    smooth = F.avg_pool2d(rearrange(weighted, "h w r -> r 1 h w"), k, stride=1, padding=k // 2,
                          count_include_pad=False)
    norm = F.avg_pool2d(pulse[None, None], k, stride=1, padding=k // 2, count_include_pad=False)
    smooth = rearrange(smooth / (norm + 1e-12), "r 1 h w -> h w r")
    return torch.view_as_complex(smooth.contiguous()), coherence


def phase_gradient_2d(field: torch.Tensor, dim: int) -> torch.Tensor:
    """``Im(dW conj W) / |W|^2`` along ``dim`` of a complex map: the phase's
    derivative, no unwrapping (the model's helper, for a 2-D map)."""
    (difference,) = torch.gradient(field, dim=dim)
    return (difference * field.conj()).imag / (field.real ** 2 + field.imag ** 2 + 1e-12)


def propagation_field(mean_phasor: torch.Tensor, coherence: torch.Tensor, weight: torch.Tensor) -> tuple:
    """Per block of ``ARROW_BLOCK`` cells: the unit direction in which the
    arrival phase falls (the way the pulse travels), the block's share of the
    weight, and how far the block's neighbours agree with it (the resultant
    length of the unit directions over ``ARROW_NEIGHBOURHOOD`` blocks, mass-
    weighted). ``(dx, dy, mass, agreement)`` on the block grid, image axes."""
    b = ARROW_BLOCK
    votes = weight * (coherence >= PHASE_COHERENCE_FLOOR)
    k_x = -phase_gradient_2d(mean_phasor, dim=-1)
    k_y = -phase_gradient_2d(mean_phasor, dim=-2)

    def pool(x):
        return F.avg_pool2d(x[None, None], b, ceil_mode=True)[0, 0]
    mass = pool(votes)
    mx = pool(k_x * votes) / (mass + 1e-12)
    my = pool(k_y * votes) / (mass + 1e-12)
    norm = torch.sqrt(mx ** 2 + my ** 2) + 1e-12
    dx, dy = mx / norm, my / norm
    n = ARROW_NEIGHBOURHOOD

    def smooth(x):
        return F.avg_pool2d(x[None, None], n, stride=1, padding=n // 2, count_include_pad=False)[0, 0]
    agreement = torch.sqrt(smooth(dx * mass) ** 2 + smooth(dy * mass) ** 2) / (smooth(mass) + 1e-12)
    return dx, dy, mass / (mass.max() + 1e-12), agreement


def draw_propagation(out: Path, captured: dict, channels: list, traces: list) -> None:
    """``propagation.png``: per channel the mean frame on the field's grid
    and a grid of arrows pointing the way the pulse travels, coloured by
    the trace that holds each block (``TRACE_TINTS`` in ``TRACES`` order)
    and faded by the block's mass and its disagreement with its
    neighbours; one page-wide row, the channel as the only label."""
    clip = captured["clip"][0]
    weight = coarse_weight(captured)                                         # (C, h, w, S)
    h, w = weight.shape[1:3]
    b = ARROW_BLOCK
    with page_grid(out, channels, 1, h / w) as axes:
        for c, ax in enumerate(axes[0]):
            grey = grey_frame(clip[c], size=(h, w))
            rgb = np.repeat(grey[..., None], 3, axis=-1) * 0.7 + 0.15
            ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
            total = weight[c].sum(dim=-1)                                    # (h, w)
            if float(total.max()) <= 0:
                continue
            mean, coherence = relative_phase(captured, c)
            dx, dy, mass, agreement = propagation_field(mean, coherence, total)
            hb, wb = mass.shape
            # Each block's arrow takes the colour of the trace with the most
            # of the block's pulsatile mass.
            block_share = F.avg_pool2d(rearrange(weight[c], "h w s -> s 1 h w"), b, ceil_mode=True)[:, 0]   # (S, hb, wb)
            dominant = block_share.argmax(dim=0).cpu().numpy()
            alpha = (np.sqrt(mass.cpu().numpy()) * agreement.cpu().numpy()).clip(0, 1)
            ys, xs = np.mgrid[0:hb, 0:wb] * b + (b - 1) / 2
            u, v = dx.cpu().numpy(), dy.cpu().numpy()
            for iy in range(hb):
                for ix in range(wb):
                    a = float(alpha[iy, ix])
                    if a < ARROW_ALPHA_FLOOR or not np.isfinite(u[iy, ix]):
                        continue
                    half = b * 0.6
                    start = (xs[iy, ix] - u[iy, ix] * half, ys[iy, ix] - v[iy, ix] * half)
                    end = (xs[iy, ix] + u[iy, ix] * half, ys[iy, ix] + v[iy, ix] * half)
                    tint = TRACE_TINTS[dominant[iy, ix]]
                    # The trace's colour with a white edge, so it reads on any grey.
                    ax.annotate("", xy=end, xytext=start,
                                arrowprops={"arrowstyle": ARROW_STYLE, "mutation_scale": ARROW_HEAD_PT,
                                            "color": (1, 1, 1, a * 0.9), "lw": 1.1, "shrinkA": 0, "shrinkB": 0})
                    ax.annotate("", xy=end, xytext=start,
                                arrowprops={"arrowstyle": ARROW_STYLE, "mutation_scale": ARROW_HEAD_PT,
                                            "color": (*tint, a), "lw": 0.6, "shrinkA": 0, "shrinkB": 0})
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
