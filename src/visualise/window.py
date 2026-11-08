"""Everything written for one window: the folder of images."""

from pathlib import Path

from src.visualise.gifs import draw_magnified_gif
from src.visualise.kinematics import draw_kinematics
from src.visualise.panels import draw_phase_masks, draw_pulsatility_mask
from src.visualise.propagation import draw_propagation

#: The magnified pulse is drawn on each of these channels that is present.
MAGNIFIED_CHANNELS = ("G", "I")


def draw_window(folder: Path, captured: dict, batch: dict, channels: list, traces: list, fs: float) -> None:
    """``pulsatility_mask.png``, ``phase_masks.png``, ``kinematics.png``,
    ``propagation.png`` and the magnified-pulse GIFs of one
    window into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    draw_pulsatility_mask(folder / "pulsatility_mask.png", captured, channels)
    draw_phase_masks(folder / "phase_masks.png", captured, channels, traces)
    draw_kinematics(folder / "kinematics.png", captured, batch, channels, traces, fs)
    draw_propagation(folder / "propagation.png", captured, channels, traces)
    for channel in MAGNIFIED_CHANNELS:
        if channel in channels:
            draw_magnified_gif(folder / f"pulse_magnified_{channel}.gif", captured, channels.index(channel), fs)
