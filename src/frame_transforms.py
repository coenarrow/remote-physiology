"""Frame resizing: the one thing the dataset does to a frame.

The cache holds raw pixel values; the dataset (``src.inputs``) resizes each
window to the interface's ``RESIZE`` and hands the raw plane on. Normalising
is the backbone's own first stage (``neural_methods.model.modules``), never
the dataset's. Every reshape is einops.
"""

import torch
import torch.nn.functional as F
from einops import rearrange


def resize_video(plane: torch.Tensor, size) -> torch.Tensor:
    """Bilinear spatial resize of ``(T, H, W)`` to ``size = (H', W')``.

    Frames are folded into the batch axis so ``interpolate`` sees plain 2-D
    images; a plane already at ``size`` is returned as is.
    """
    height, width = size
    if tuple(plane.shape[-2:]) == (height, width):
        return plane
    frames = rearrange(plane, "t h w -> t 1 h w")
    resized = F.interpolate(frames, size=(height, width), mode="bilinear",
                            align_corners=False)
    return rearrange(resized, "t 1 h w -> t h w")
