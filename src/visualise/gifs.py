"""``pulse_magnified_<channel>.gif``: the Eulerian-magnified pulse over the
window's frames."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

GIF_FRAME_STEP = 3                       # every third frame is drawn
GIF_SCALE = 3                            # nearest-neighbour upscale of the grid
#: Magnified GIF: the band-passed pulse's peak-to-peak swing as a fraction
#: of the grey range, after magnification.
MAGNIFICATION_SWING = 0.8


def save_gif(out: Path, frames: list, fs: float) -> None:
    """``frames`` (PIL images, every ``GIF_FRAME_STEP``-th frame of the
    window) as a looping GIF at the interface's frame rate."""
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * GIF_FRAME_STEP / fs), loop=0)


def to_image(rgb: torch.Tensor) -> Image.Image:
    """An ``(h, w, 3)`` tensor in ``[0, 1]`` as a PIL image, upscaled by
    ``GIF_SCALE`` without smoothing."""
    image = Image.fromarray((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    return image.resize((image.width * GIF_SCALE, image.height * GIF_SCALE), Image.NEAREST)


def draw_magnified_gif(out: Path, captured: dict, channel_index: int, fs: float) -> None:
    """``pulse_magnified_<channel>.gif``: the channel's frames on the field's
    grid with the fundamental's band-passed signal added back many times
    over, brightening in red and darkening in blue, the pulsatility map as
    its alpha. The magnification is set so the pulsatile region's swing is
    ``MAGNIFICATION_SWING`` of the grey range."""
    field = captured["harmonic"][0, channel_index]                           # (T, h, w) complex
    pulse = captured["pulsatility_coarse"][0, channel_index]
    alpha = (pulse / (pulse.max() + 1e-12)).clamp(0, 1)
    signal = field.real                                                     # (T, h, w), zero-mean band-passed
    scale = float(torch.quantile((signal.abs() * alpha).flatten(), 0.99)) + 1e-12
    swing = (signal / scale * MAGNIFICATION_SWING).clamp(-1, 1) * alpha      # (T, h, w)
    clip = captured["clip"][0, channel_index]
    frames = []
    for i in range(0, clip.shape[0], GIF_FRAME_STEP):
        grey = F.interpolate(clip[i][None, None], size=field.shape[-2:], mode="area")[0, 0]
        grey = (grey - clip.min()) / (clip.max() - clip.min() + 1e-6)
        lift = swing[i]
        rgb = torch.stack([grey + lift.clamp(min=0), grey - lift.abs() * 0.5, grey - lift.clamp(max=0)], dim=-1)
        frames.append(to_image(rgb))
    save_gif(out, frames, fs)
