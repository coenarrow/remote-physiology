"""The page layout the thesis figures share: a grid of panels, one column
per channel, as wide as an A4 text block, drawn under seaborn's paper
style with the channel's name as the only label."""

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

#: Full channel names for the column titles; a letter without one is shown as is.
CHANNEL_NAMES = {"R": "Red", "G": "Green", "B": "Blue", "I": "Infrared", "D": "Depth"}
#: A row of panels is as wide as an A4 page's text block.
FIGURE_WIDTH_IN = 6.3
#: Room above the panels for the column titles.
TITLE_HEIGHT_IN = 0.3
DPI = 200


def grey_frame(clip: torch.Tensor, size: tuple | None = None) -> np.ndarray:
    """A channel's ``(T, H, W)`` clip as its mean frame in ``[0, 1]``,
    area-resized to ``size`` when given (the field's grid)."""
    frame = clip.mean(dim=0)
    if size is not None and tuple(frame.shape) != tuple(size):
        frame = F.interpolate(frame[None, None], size=size, mode="area")[0, 0]
    return ((frame - frame.min()) / (frame.max() - frame.min() + 1e-6)).cpu().numpy()


@contextmanager
def page_grid(out: Path, channels: list, n_rows: int, aspect: float, row_labels: list | None = None):
    """Yields the ``(n_rows, len(channels))`` axes of a figure sized to the
    page: each panel ``aspect`` (height over width) times as tall as it is
    wide, the channel names over the first row, ``row_labels`` down the
    left when given. On exit the ticks are stripped and the figure is
    written to ``out`` cropped to its contents."""
    n_cols = len(channels)
    panel = FIGURE_WIDTH_IN / n_cols
    with sns.axes_style("white"), sns.plotting_context("paper"):
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                                 figsize=(FIGURE_WIDTH_IN, panel * aspect * n_rows + TITLE_HEIGHT_IN))
        for c, channel in enumerate(channels):
            axes[0, c].set_title(CHANNEL_NAMES.get(channel, channel))
        for r, label in enumerate(row_labels or []):
            axes[r, 0].set_ylabel(label)
        yield axes
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        sns.despine(fig, left=True, bottom=True)
        fig.tight_layout(pad=0.3)
        fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
