"""Frame resizing: the one thing the dataset does to a frame.

The cache holds raw pixel values; the dataset (``src.inputs``) resizes each
window to the interface's ``RESIZE`` and hands the raw plane on. Normalising
is the backbone's own first stage (``neural_methods.model._shared_modules``), never
the dataset's. Every reshape is einops.

The filters are the Neckflix cacher's (``dataset/cachers/neckflix``,
``video.py``), so a window read from a cache at another size matches one read
from a cache written at the interface's size: OpenCV ``INTER_AREA`` for a
shrink of colour or infrared, ``INTER_LINEAR`` for a grow, ``INTER_NEAREST``
for depth either way. The area filter is the exact mean of the source pixels
each output pixel covers, edge pixels weighted by their overlap, so no source
pixel is skipped, every output pixel averages the same area and neighbouring
outputs carry independent noise, which the cardio models' per-cell
statistics assume. Nearest on depth keeps the sensor's invalid-depth zeros
from being blended into false distances.

History: until 2026-10-12 this was bilinear *without* antialiasing, which at
128 -> 72 blends one to four source pixels per output pixel in a repeating
pattern, so per-pixel noise varied in a lattice and a weak pulse drowned in
it (``cardioconv_cardiohydra_development_notes/experiment_9.md``). From
2026-10-14 to 2026-10-18 it was the same area average as a dense einsum,
which on the native-resolution HPC cache cost 7-30x this call and made the
loader the cluster's bottleneck.
"""

import cv2
import numpy as np
import torch

# Every DataLoader worker would otherwise start its own OpenCV thread pool,
# and twenty of them on one node fight each other; one thread per call.
cv2.setNumThreads(0)

#: Modality whose frames are a distance field, resized by nearest neighbour.
NEAREST_MODALITIES = frozenset({"depth"})


def resize_video(plane: torch.Tensor, size, modality: str = "rgb") -> torch.Tensor:
    """Resize ``(T, H, W)`` to ``size = (H', W')`` with the cacher's filter
    for ``modality``: area average when shrinking, linear when growing,
    nearest neighbour for depth. A plane already at ``size`` is returned as
    is; the result is float32."""
    height, width = size
    if tuple(plane.shape[-2:]) == (height, width):
        return plane
    if modality in NEAREST_MODALITIES:
        interpolation = cv2.INTER_NEAREST
    elif height < plane.shape[-2] or width < plane.shape[-1]:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    frames = np.ascontiguousarray(plane.detach().cpu().numpy(), dtype=np.float32)
    out = np.empty((frames.shape[0], height, width), dtype=np.float32)
    for t in range(frames.shape[0]):
        cv2.resize(frames[t], (width, height), dst=out[t], interpolation=interpolation)
    return torch.from_numpy(out).to(plane.device)
