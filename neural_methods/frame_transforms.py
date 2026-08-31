"""Consumer-side frame preprocessing for the raw-pixel zarr cache.

The zarr loader deliberately emits raw pixel values (design spec, "Out of
scope"), so the ``DATA_TYPE`` transforms that used to live in ``BaseLoader``
now run here, on ``(B, C, T, H, W)`` torch tensors, immediately before a
backbone sees them. Semantics match ``BaseLoader.diff_normalize_data`` /
``standardized_data`` exactly, except that statistics are per-sample rather
than per-cached-chunk — the natural reading once a batch holds several windows.

Every reshape is einops.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, reduce

#: ``DATA_TYPE`` names understood here, matching the YAML vocabulary.
DATA_TYPES = ("Raw", "Standardized", "DiffNormalized")

_EPS = 1e-7


def _per_sample_std(video: torch.Tensor) -> torch.Tensor:
    """Std over everything but the batch axis, kept broadcastable as (B,1,1,1,1)."""
    flat = rearrange(video, "b c t h w -> b (c t h w)")
    return rearrange(flat.std(dim=1), "b -> b 1 1 1 1")


def _per_sample_mean(video: torch.Tensor) -> torch.Tensor:
    return reduce(video, "b c t h w -> b 1 1 1 1", "mean")


def standardized(video: torch.Tensor) -> torch.Tensor:
    """Z-score a clip with its own mean/std (``BaseLoader.standardized_data``)."""
    centred = video - _per_sample_mean(video)
    out = centred / _per_sample_std(video).clamp_min(_EPS)
    return torch.nan_to_num(out)


def diff_normalized(video: torch.Tensor) -> torch.Tensor:
    """Frame-to-frame difference normalised by its own std, zero-padded to T.

    ``d_t = (x_{t+1} - x_t) / (x_{t+1} + x_t + 1e-7)``, divided by the clip's std,
    with a zero frame appended so the temporal length is unchanged — the same
    construction (and the same padding-at-the-end convention) as
    ``BaseLoader.diff_normalize_data``.
    """
    later, earlier = video[:, :, 1:], video[:, :, :-1]
    diff = (later - earlier) / (later + earlier + _EPS)
    diff = diff / _per_sample_std(diff).clamp_min(_EPS)
    diff = torch.nan_to_num(diff)
    pad = torch.zeros_like(video[:, :, :1])
    return torch.cat([diff, pad], dim=2)


_TRANSFORMS = {
    "Raw": lambda video: video,
    "Standardized": standardized,
    "DiffNormalized": diff_normalized,
}


def resize_video(video: torch.Tensor, size) -> torch.Tensor:
    """Bilinear spatial resize of ``(B, C, T, H, W)`` to ``size=(H', W')``.

    The cache is written at one resolution; models that want another (DeepPhys
    at 72, say) resize here rather than forcing a second cache. Frames are
    folded into the batch axis so ``interpolate`` sees plain 2-D images.
    """
    height, width = size
    if tuple(video.shape[-2:]) == (height, width):
        return video
    flat = rearrange(video, "b c t h w -> (b t) c h w")
    resized = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
    return rearrange(resized, "(b t) c h w -> b c t h w", b=video.shape[0])


def apply_data_types(video: torch.Tensor, data_types) -> torch.Tensor:
    """Concatenate the named transforms along the channel axis.

    Upstream toolbox semantics: ``DATA_TYPE: ['DiffNormalized', 'Standardized']``
    means a 2C-channel input whose blocks are the two transforms of the same
    clip — which is exactly what DeepPhys/TS-CAN split back apart. A single
    entry leaves the channel count alone.
    """
    if not data_types:
        return video
    unknown = [name for name in data_types if name not in _TRANSFORMS]
    if unknown:
        raise ValueError(f"Unknown DATA_TYPE(s) {unknown}; known: {list(DATA_TYPES)}")
    blocks = [_TRANSFORMS[name](video) for name in data_types]
    return torch.cat(blocks, dim=1) if len(blocks) > 1 else blocks[0]


class FrameTransform(torch.nn.Module):
    """Resize-then-``DATA_TYPE`` pipeline, carried by the model that needs it.

    Holding it on the model (rather than in the trainer) keeps the promise that
    a model consumes exactly what the loader emits: raw pixels in, predictions
    out, whatever preprocessing the architecture happens to want in between.
    """

    def __init__(self, data_types=("Standardized",), size=None):
        super().__init__()
        self.data_types = tuple(data_types or ())
        unknown = [name for name in self.data_types if name not in _TRANSFORMS]
        if unknown:
            raise ValueError(f"Unknown DATA_TYPE(s) {unknown}; known: {list(DATA_TYPES)}")
        self.size = tuple(size) if size else None

    @property
    def channel_multiplier(self) -> int:
        """How many channel blocks the transform emits per input channel."""
        return max(len(self.data_types), 1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if self.size is not None:
            video = resize_video(video, self.size)
        return apply_data_types(video, self.data_types)

    def extra_repr(self) -> str:
        return f"data_types={list(self.data_types)}, size={self.size}"
