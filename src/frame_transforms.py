"""Frame resizing: the one thing the dataset does to a frame.

The cache holds raw pixel values; the dataset (``src.inputs``) resizes each
window to the interface's ``RESIZE`` and hands the raw plane on. Normalising
is the backbone's own first stage (``neural_methods.model._shared_modules``), never
the dataset's. Every reshape is einops.
"""

import torch
import torch.nn.functional as F
from einops import rearrange


def resize_video(plane: torch.Tensor, size) -> torch.Tensor:
    """Antialiased bilinear resize of ``(T, H, W)`` to ``size = (H', W')``.

    Shrinking, ``antialias=True`` widens the tent kernel with the factor, so
    every output pixel averages the same amount of source and no source pixel
    is skipped; growing, it is plain linear interpolation between pixel
    centres (``align_corners=False``). A plane already at ``size`` is returned
    as is.

    History: until 2026-10-12 this was bilinear *without* antialiasing, which
    at 128 -> 72 blends one to four source pixels per output pixel in a
    repeating pattern, so per-pixel noise varied in a lattice and a weak pulse
    drowned in it (``cardioconv_cardiohydra_development_notes/experiment_9.md``).
    From 2026-10-14 to 2026-10-18 it was an exact area average applied as a
    dense einsum, which on the native-resolution HPC cache cost 7-30x the
    time of this call and made the loader the cluster's bottleneck.
    """
    height, width = size
    if tuple(plane.shape[-2:]) == (height, width):
        return plane
    frames = rearrange(plane.float(), "t h w -> t 1 h w")
    resized = F.interpolate(frames, size=(height, width), mode="bilinear",
                            align_corners=False, antialias=True)
    return rearrange(resized, "t 1 h w -> t h w")
