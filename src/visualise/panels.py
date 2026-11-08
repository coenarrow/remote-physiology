"""The masks over the anatomy: ``pulsatility_mask.png`` (one row, the
pulsatility map over each channel's mean frame) and ``phase_masks.png``
(the same row, then each trace's mask weighted by pulsatility: how the
model splits the pulsatile region between the traces)."""

from pathlib import Path

import numpy as np
import torch

from src.visualise.layout import grey_frame, page_grid

#: The map's colour map and the alpha of its brightest pixel.
MAP_CMAP = "magma"
MAP_ALPHA = 0.9


def window_maps(captured: dict) -> tuple:
    """``(pulsatility (C, h, w), masks (C, h, w, S))``, the trace masks
    weighted by pulsatility, time-averaged."""
    pulsatility = captured["pulsatility"][0].mean(dim=1)
    masks = (captured["pulsatility"][0, ..., None] * captured["phase_masks"][0]).mean(dim=1)
    return pulsatility, masks


def draw_map_over(ax, grey: np.ndarray, image: torch.Tensor) -> None:
    """``image`` scaled to its own maximum, in ``MAP_CMAP`` over ``grey``
    with its value as its alpha, so the mass sits on the anatomy and the
    empty parts of the map show the frame through."""
    weight = image.cpu().numpy()
    weight = weight / (weight.max() + 1e-12)
    ax.imshow(grey, cmap="gray", vmin=0.0, vmax=1.0)
    ax.imshow(weight, cmap=MAP_CMAP, vmin=0.0, vmax=1.0, alpha=np.clip(weight, 0.0, 1.0) * MAP_ALPHA)


def mask_grid(out: Path, captured: dict, channels: list, rows: list) -> None:
    """A page grid of maps over the mean frames, one row per
    ``(label, maps (C, h, w))`` pair; the labels are shown only when there
    is more than one row."""
    clip = captured["clip"][0]                                               # (C, T, H, W)
    aspect = clip.shape[-2] / clip.shape[-1]
    labels = [label for label, _ in rows] if len(rows) > 1 else None
    with page_grid(out, channels, len(rows), aspect, labels) as axes:
        for c in range(len(channels)):
            grey = grey_frame(clip[c])
            for r, (_, maps) in enumerate(rows):
                draw_map_over(axes[r, c], grey, maps[c])


def draw_pulsatility_mask(out: Path, captured: dict, channels: list) -> None:
    """``pulsatility_mask.png``: the window's pulsatility map over each
    channel's mean frame, one row, the channel as the only label."""
    pulsatility, _ = window_maps(captured)
    mask_grid(out, captured, channels, [("Pulsatility", pulsatility)])


def draw_phase_masks(out: Path, captured: dict, channels: list, traces: list) -> None:
    """``phase_masks.png``: the pulsatility row of ``pulsatility_mask.png``
    and below it, per trace, the trace's mask weighted by pulsatility: how
    the pulsatile region is split between the traces (the background mask
    is left out)."""
    pulsatility, masks = window_maps(captured)
    rows = [("Pulsatility", pulsatility)] + [(trace, masks[..., s]) for s, trace in enumerate(traces)]
    mask_grid(out, captured, channels, rows)
