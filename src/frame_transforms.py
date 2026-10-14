"""Frame resizing: the one thing the dataset does to a frame.

The cache holds raw pixel values; the dataset (``src.inputs``) resizes each
window to the interface's ``RESIZE`` and hands the raw plane on. Normalising
is the backbone's own first stage (``neural_methods.model._shared_modules``), never
the dataset's. Every reshape is einops.
"""

import math

import torch
from einops import einsum


def axis_weights(n_in: int, n_out: int) -> torch.Tensor:
    """``(n_out, n_in)`` resampling weights along one axis, each row summing to one.

    Shrinking, each output pixel is the exact mean of the source span it
    covers, ``n_in / n_out`` pixels wide, its two edge pixels weighted by the
    fraction they overlap: an area average at any factor, whole or not. Grown,
    there is no span to average, so it is linear interpolation between pixel
    centres (``align_corners=False``, edges clamped).
    """
    scale = n_in / n_out
    weights = torch.zeros(n_out, n_in, dtype=torch.float64)
    if scale >= 1.0:
        for i in range(n_out):
            lo, hi = i * scale, (i + 1) * scale
            for j in range(math.floor(lo), min(math.ceil(hi), n_in)):
                weights[i, j] = min(hi, j + 1) - max(lo, j)
    else:
        for i in range(n_out):
            centre = min(max((i + 0.5) * scale - 0.5, 0.0), n_in - 1)
            j = min(math.floor(centre), n_in - 2) if n_in > 1 else 0
            frac = centre - j
            weights[i, j] += 1.0 - frac
            if n_in > 1:
                weights[i, j + 1] += frac
    return (weights / weights.sum(dim=1, keepdim=True)).float()


def resize_video(plane: torch.Tensor, size) -> torch.Tensor:
    """Spatial resize of ``(T, H, W)`` to ``size = (H', W')``: an area average
    along an axis that shrinks, linear along one that grows
    (:func:`axis_weights`). A plane already at ``size`` is returned as is.

    Until 2026-10-12 this was bilinear without antialiasing, which at 128 ->
    72 blends one to four source pixels per output pixel in a repeating
    pattern: per-pixel noise varied in a lattice and a weak pulse drowned in
    it (``cardioconv_cardiohydra_development_notes/experiment_9.md``).
    """
    height, width = size
    if tuple(plane.shape[-2:]) == (height, width):
        return plane
    rows = axis_weights(plane.shape[-2], height).to(plane.device)
    cols = axis_weights(plane.shape[-1], width).to(plane.device)
    return einsum(rows, plane.float(), cols, "y h, t h w, x w -> t y x")
