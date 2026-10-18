"""Show what a CardioConv or CardioHydra run looked at, window by window.

    uv run scripts/visualise.py --checkpoint runs/dev_cardio/exp2_cardioconv_synth \\
        --dataset synthetic_neck --windows 6

``--checkpoint`` is a run directory written by ``scripts/run.py`` (its latest
epoch, or ``--epoch``), one of its ``epoch_NN`` folders or a ``model.pt``.
Every strided window of ``--dataset`` is a candidate (one participant's with
``--test-participant-id``); ``--windows`` evenly spaced ones are run with the
backbone's ``capture`` switch on, and for each one the script writes, under
``<epoch dir>/visuals/<dataset>/<recording>_<perspective>_<start frame>/``:

- ``panel.png``: one row per input channel with the window's mean frame, the
  window's pulsatility map, each trace's mask weighted by pulsatility, and
  the background mask; below them the predicted and label waveforms per
  trace and the estimated heart rate. A synthetic store's true vessel map
  (``vessel_ids``: artery, vein) is drawn as contours on every map.
- ``overlay.png``: the same maps drawn over each channel's mean frame, the
  map's value as its alpha, so the mass is read against the anatomy.
- ``masks.gif``: the depth channel's frames with the window's pulsatility
  and trace masks over them, and beside them the fundamental's wrapped
  phase on the field's grid (a cyclic colour map, the pulsatility as its
  alpha): a travelling pulse shows bands sweeping along the vessel, a
  standing pulsation the whole region flashing in step.
- ``propagation.png``: per channel, the mean frame and a grid of arrows,
  one per block, each pointing the way the pulse travels there: the
  direction in which the arrival phase (the window-averaged phase of each
  pixel relative to the channel's own pulse at the fundamental) falls,
  averaged over the block. An arrow is coloured by the trace that holds its
  block (ABP red, CVP blue) and fades with the block's pulsatile mass and
  with how far its neighbours disagree with it, so a region whose arrows
  all point one way stands out and a region of noise fades. No numbers; the
  length is fixed.
- ``pulse_magnified_<channel>.gif``: Eulerian magnification of the pulse,
  the fundamental's band-passed signal added back onto the frame many times
  over (brightening in red, darkening in blue), so the sub-pixel pulsation
  becomes a visible wave; whether it sweeps along the neck or beats in
  place is the propagation question seen directly.
- ``kinematics.png``: what the kinematics stage reads off the field at the
  fundamental, one row per channel, every map on the field's grid over the
  channel's mean frame with the pulsatility map as its alpha and on a
  fixed physical scale (so maps compare across channels and windows, unlike
  the masks): the log amplitude; the wavelength ``2 pi / |k|`` in input
  pixels (a travelling pulse's wavelength is metres, far wider than the
  frame, so the map says whether ``|k|`` is small and smooth on the vessels
  or speckle); the signed phase speed on a symmetric log scale with the
  trace masks' contours (do the ABP and CVP pixels carry different speeds?);
  and the heading as arrows, one per block, length by pulsatile weight. The
  last row plots the mask-pooled signed speed per trace over the window's
  frames, one line per channel, against the label waveform: a wave speed
  modulates within the beat, a noise ratio does not.

and one ``scores.csv`` per dataset: per window and channel, the pulsatility
map's effective pixel count and each trace's and the background's share of
the pulsatile mass (also printed in each map's title, because every map is
drawn at its own colour scale); per trace the kinematics pooled under its
mask at the fundamental (the speed the statistics head reads, in pixels per
frame; the median wavelength; the heading's circular mean and resultant
length, 1 when every pulsatile pixel's wave travels the same way); the
window's label level (ABP mean and systolic) so the pooled speed can be
read against pressure; and, when the dataset carries ``vessel_ids``:
per window and channel, the share of the pulsatility distribution that falls
on the neck (``neck_mask``) and on the vessel pixels, each against that
region's share of the grid (chance), and the share of each trace's pulsatile
mass inside the true artery, the true vein and the background. The
pulsatility mask is meant to find where the neck pulses, so its mass on the
neck should sit well above chance; only the phase masks separate the vessels,
so the arterial trace's mask should sit on the artery and the venous one on
the vein. Nothing is trained or scored for agreement here;
``scripts/evaluate.py`` does that.

``--windows 0`` runs every window; with ``--no-images`` only ``scores.csv``
is written, which is how the pooled speed of a whole participant is
collected. Whenever more than a dozen windows were run,
``speed_vs_pressure.png`` beside ``scores.csv`` scatters each channel's
pooled ABP speed against the window's label ABP mean: the Bramwell-Hill
claim that the speed is a level cue is a slope there or it is nothing.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from einops import rearrange, reduce
from PIL import Image
from torch.utils.data import default_collate

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError                                # noqa: E402
from src.dataset_config import load_dataset_configs              # noqa: E402
from src.datasets import Split, hold_out_participant, load_stores  # noqa: E402
from src.experiment import (                                     # noqa: E402
    epoch_checkpoint, limit_windows, load_checkpoint, rebuild, test_windows,
)
from src.models import build_model                               # noqa: E402
from src.trainer import move_to_device                           # noqa: E402

VISUALS_DIR = "visuals"
GIF_FRAME_STEP = 3
GIF_SCALE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, metavar="PATH")
    parser.add_argument("--epoch", type=int, metavar="N")
    parser.add_argument("--dataset", required=True, metavar="NAME")
    parser.add_argument("--test-participant-id", metavar="ID")
    parser.add_argument("--windows", type=int, default=6, metavar="N",
                        help="evenly spaced windows to draw (default 6; 0 is every window)")
    parser.add_argument("--no-images", action="store_true",
                        help="write scores.csv (and speed_vs_pressure.png) only, no per-window images")
    parser.add_argument("--no-gpu", action="store_true")
    return parser


def vessel_maps(store: Path, shape: tuple) -> dict | None:
    """``{"artery", "vein", "neck"}`` boolean maps on the mask grid, nearest-
    neighbour resized from the store's root arrays; ``None`` without them."""
    group = zarr.open_group(str(store), mode="r")
    if "vessel_ids" not in group:
        return None
    ids = torch.from_numpy(np.asarray(group["vessel_ids"][:], dtype=np.float32))[None, None]
    ids = F.interpolate(ids, size=shape, mode="nearest")[0, 0]
    maps = {"artery": ids == 1, "vein": ids == 2}
    if "neck_mask" in group:
        neck = torch.from_numpy(np.asarray(group["neck_mask"][:], dtype=np.float32))[None, None]
        maps["neck"] = F.interpolate(neck, size=shape, mode="nearest")[0, 0] > 0
    return maps


def health(captured: dict, channels: list, traces: list) -> list:
    """Per channel: the pulsatility map's effective pixel count (exp of its
    entropy) and each trace's and the background's share of the pulsatile
    mass. The panels draw every map at its own colour scale, so a map that
    has all but vanished still shows bright pixels; these numbers say how
    much mass is there."""
    pulsatility = captured["pulsatility"][0, :, 0]                            # (C, h, w), one map per window
    weight = rearrange(pulsatility, "c h w -> c (h w)")
    entropy = -torch.special.xlogy(weight, weight).sum(dim=-1)                # 0 log 0 = 0
    shares = reduce(pulsatility[..., None] * captured["phase_masks"][0, :, 0], "c h w s -> c s", "sum")
    background = reduce(pulsatility * captured["background"][0, :, 0], "c h w -> c", "sum")
    rows = []
    for c, channel in enumerate(channels):
        row = {"channel": channel, "pulsatility_effective_px": float(entropy[c].exp())}
        for s, trace in enumerate(traces):
            row[f"{trace}_share"] = float(shares[c, s])
        row["background_share"] = float(background[c])
        rows.append(row)
    return rows


def share_text(share: float) -> str:
    """A share as a percentage, or as a power of ten once it is too small to read."""
    return f"{100 * share:.2g}%" if share >= 1e-3 else f"1e{np.log10(max(share, 1e-45)):.0f}"


def scores(captured: dict, maps: dict, channels: list, traces: list) -> list:
    """Per channel: pulsatile mass on the vessels against chance, and each
    trace's pulsatile mass split over the true artery, vein and background."""
    pulsatility = captured["pulsatility"][0].mean(dim=1)                      # (C, h, w)
    weight = captured["pulsatility"][0, ..., None] * captured["phase_masks"][0]   # (C, T, h, w, S)
    claimed = weight.mean(dim=1)                                              # (C, h, w, S)
    vessels = maps["artery"] | maps["vein"]
    neck = maps.get("neck", torch.ones_like(vessels))
    rows = []
    for c, channel in enumerate(channels):
        # The pulsatility mask is meant to find where the neck pulses, not the
        # vessels (only the phase masks separate those), so its score is its
        # mass on the neck against the neck's share of the grid.
        row = {"channel": channel,
               "pulsatility_on_neck": float(pulsatility[c][neck].sum()),
               "neck_share_of_grid": float(neck.float().mean()),
               "pulsatility_on_vessels": float(pulsatility[c][vessels].sum()),
               "vessel_share_of_grid": float(vessels.float().mean())}
        for s, trace in enumerate(traces):
            total = float(claimed[c, ..., s].sum()) or 1.0
            for region in ("artery", "vein"):
                row[f"{trace}_mass_in_{region}"] = float(claimed[c, ..., s][maps[region]].sum()) / total
            row[f"{trace}_mass_in_background"] = 1.0 - row[f"{trace}_mass_in_artery"] - row[f"{trace}_mass_in_vein"]
        rows.append(row)
    return rows

#: Fixed scales of the kinematic maps, so panels compare across windows.
WAVELENGTH_RANGE_PX = (3.0, 1000.0)      # log scale: the grid cell to five frames
SPEED_RANGE_PX_PER_FRAME = 100.0         # symmetric log scale, linear within 1
QUIVER_BLOCK = 3                         # field-grid cells per arrow
QUIVER_GAIN = 3.0                        # arrow length per unit heading resultant, in blocks


def coarse_weight(captured: dict) -> torch.Tensor:
    """``(C, h, w, S)``: each trace's pooling weight on the field's grid,
    pulsatility times trace mask, one map per window."""
    return captured["pulsatility_coarse"][0, ..., None] * captured["phase_masks_coarse"][0]


def weighted_median(values: torch.Tensor, weights: torch.Tensor) -> float:
    """The weighted median of ``values`` (any shape) under ``weights`` of the
    same shape."""
    flat, w = values.reshape(-1), weights.reshape(-1)
    order = torch.argsort(flat)
    cum = torch.cumsum(w[order], dim=0)
    index = int(torch.searchsorted(cum, cum[-1] / 2).clamp(max=len(flat) - 1))
    return float(flat[order][index])


def kinematic_summary(captured: dict, channels: list, traces: list) -> list:
    """Per channel and trace, the kinematics under the trace's mask at the
    fundamental: the pooled speed the statistics head reads (``expm1`` of its
    ``log1p`` mean, pixels per frame), the mask-weighted median wavelength
    (input pixels) and the heading's weighted circular mean (degrees, image
    axes) and resultant length."""
    weight = coarse_weight(captured)                                          # (C, h, w, S)
    n_channels, n_traces = weight.shape[0], weight.shape[-1]
    n_harmonics = captured["pooled_speed"].shape[-1] // (n_channels * n_traces)
    pooled = rearrange(captured["pooled_speed"][0], "(c s n) -> c s n", c=n_channels, s=n_traces, n=n_harmonics)
    wavelength = 2 * np.pi / captured["wavenumber"][0, ..., 0]               # (C, T, h, w)
    dx, dy = captured["direction_x"][0, ..., 0], captured["direction_y"][0, ..., 0]
    frames = wavelength.shape[1]
    rows = []
    for c, channel in enumerate(channels):
        row = {"channel": channel}
        for s, trace in enumerate(traces):
            w = weight[c, ..., s]
            area = float(w.sum())
            if area <= 0:                                                     # an absent channel
                for key in ("pooled_speed_px_per_frame", "median_wavelength_px", "heading_deg", "heading_resultant"):
                    row[f"{trace}_{key}"] = float("nan")
                continue
            row[f"{trace}_pooled_speed_px_per_frame"] = float(torch.expm1(pooled[c, s, 0]))
            row[f"{trace}_median_wavelength_px"] = weighted_median(
                wavelength[c], w[None].expand(frames, -1, -1))
            mean_x = float((dx[c] * w).sum() / (area * frames))
            mean_y = float((dy[c] * w).sum() / (area * frames))
            row[f"{trace}_heading_deg"] = float(np.degrees(np.arctan2(mean_y, mean_x)))
            row[f"{trace}_heading_resultant"] = float(np.hypot(mean_x, mean_y))
        rows.append(row)
    return rows


def symlog(x: np.ndarray, limit: float) -> np.ndarray:
    """Signed ``log1p``, scaled so ``limit`` maps to 1."""
    return np.sign(x) * np.log1p(np.abs(x)) / np.log1p(limit)


def draw_kinematics(out: Path, captured: dict, batch: dict, channels: list, traces: list,
                    fs: float) -> None:
    """``kinematics.png``: per channel the fundamental's log amplitude,
    wavelength, signed phase speed and heading on the field's grid, each over
    the channel's mean frame with the pulsatility map as alpha and on a
    fixed scale; below, the mask-pooled signed speed per trace over the
    window's frames against the label waveform."""
    clip = captured["clip"][0]                                               # (C, T, H, W)
    pulse = captured["pulsatility_coarse"][0]                                # (C, h, w)
    h, w = pulse.shape[-2:]
    alpha_all = (pulse / (pulse.amax(dim=(-2, -1), keepdim=True) + 1e-12)).clamp(0, 1).cpu().numpy()
    amplitude = captured["power"][0, ..., 0].mean(dim=1)                      # (C, h, w)
    wavelength = (2 * np.pi / captured["wavenumber"][0, ..., 0]).median(dim=1).values   # (C, h, w)
    speed = captured["phase_speed"][0, ..., 0]                                # (C, T, h, w)
    speed_map = speed.median(dim=1).values
    weight = coarse_weight(captured)                                          # (C, h, w, S)
    dx = (captured["direction_x"][0, ..., 0] * pulse[:, None]).mean(dim=1)   # (C, h, w), pulsatility-weighted
    dy = (captured["direction_y"][0, ..., 0] * pulse[:, None]).mean(dim=1)
    titles = ["log amplitude", "wavelength (px, log)", "signed speed (px/frame, symlog)", "heading"]
    n_cols = len(titles)
    fig, axes = plt.subplots(len(channels) + 1, n_cols, figsize=(3.4 * n_cols, 3.0 * (len(channels) + 1)))
    summary = kinematic_summary(captured, channels, traces)
    lo, hi = np.log10(WAVELENGTH_RANGE_PX)
    colours = ("red", "deepskyblue", "lime", "orange")
    for c, channel in enumerate(channels):
        frame = clip[c].mean(dim=0)[None, None]
        grey = F.interpolate(frame, size=(h, w), mode="area")[0, 0]
        grey = ((grey - grey.min()) / (grey.max() - grey.min() + 1e-6)).cpu().numpy()
        alpha = alpha_all[c] * 0.9
        amp = torch.log10(amplitude[c] + 1e-9).cpu().numpy()
        amp = (amp - amp.max() + 3) / 3                                       # three decades below the channel's peak
        wl = (np.log10(np.clip(wavelength[c].cpu().numpy(), *WAVELENGTH_RANGE_PX)) - lo) / (hi - lo)
        sp = symlog(speed_map[c].cpu().numpy(), SPEED_RANGE_PX_PER_FRAME)
        for col, (image, cmap, vmin, vmax) in enumerate(((amp, "magma", 0, 1), (wl, "viridis", 0, 1),
                                                         (sp, "RdBu_r", -1, 1))):
            ax = axes[c, col]
            ax.imshow(grey, cmap="gray", vmin=0, vmax=1)
            ax.imshow(np.clip(image, vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
            if col == 2:
                for s, colour in zip(range(len(traces)), colours):
                    m = weight[c, ..., s].cpu().numpy()
                    ax.contour(m / (m.max() + 1e-12), levels=[0.3], colors=colour, linewidths=0.6)
            ax.axis("off")
        ax = axes[c, 3]
        ax.imshow(grey, cmap="gray", vmin=0, vmax=1)
        b = QUIVER_BLOCK
        ys, xs = np.mgrid[b // 2:h:b, b // 2:w:b]
        def block(x):
            pooled = F.avg_pool2d(x[None, None], b, ceil_mode=True)[0, 0].cpu().numpy()
            return pooled[:ys.shape[0], :ys.shape[1]]
        u, v, a = block(dx[c]), block(dy[c]), block(torch.from_numpy(alpha_all[c]))
        # Arrow length is the block's resultant times its pulsatile weight, so
        # an incoherent block draws a dot; QUIVER_GAIN keeps a resultant of
        # a few tenths visible, and the length is capped at the block.
        gain = QUIVER_GAIN * a
        ax.quiver(xs, ys, u * gain, -v * gain, color="yellow", angles="xy", scale_units="xy",
                  scale=1.0 / b, width=0.004, headwidth=4)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.axis("off")
        row = summary[c]
        notes = [f"{channel}: {titles[0]}" if c == 0 else channel,
                 (titles[1] if c == 0 else "") + "\n"
                 + " / ".join(f"{t} {row[f'{t}_median_wavelength_px']:.0f}" for t in traces),
                 (titles[2] if c == 0 else "") + "\n"
                 + " / ".join(f"{t} {row[f'{t}_pooled_speed_px_per_frame']:+.1f}" for t in traces),
                 (titles[3] if c == 0 else "") + "\n"
                 + " / ".join(f"{t} R {row[f'{t}_heading_resultant']:.2f}" for t in traces)]
        for col, note in enumerate(notes):
            axes[c, col].set_title(note.strip("\n"), fontsize=7)
    for col in range(n_cols):
        axes[-1, col].axis("off")
    # The pooled signed speed per trace over the frames, one line per channel.
    t = np.arange(speed.shape[1]) / fs
    area = weight.sum(dim=(1, 2)).clamp_min(1e-9)                             # (C, S)
    pooled_t = torch.einsum("cthw,chws->cst", speed, weight) / area[..., None]   # (C, S, T)
    for s, trace in enumerate(traces):
        ax = fig.add_subplot(len(channels) + 1, n_cols, len(channels) * n_cols + 1 + s)
        for c, channel in enumerate(channels):
            ax.plot(t, symlog(pooled_t[c, s].cpu().numpy(), SPEED_RANGE_PX_PER_FRAME), lw=0.7, label=channel)
        ax.set_ylim(-1, 1)
        ax.set_title(f"{trace}: pooled signed speed (symlog, +-{SPEED_RANGE_PX_PER_FRAME:.0f} px/frame)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, ncol=len(channels), loc="upper right")
        if "labels" in batch and trace in batch["labels"]:
            twin = ax.twinx()
            twin.plot(t, batch["labels"][trace][0].cpu().numpy(), "k", lw=0.6, alpha=0.6)
            twin.tick_params(labelsize=6)
    fig.suptitle(f"{out.parent.name}: kinematics at the fundamental, "
                 f"estimated rate {float(captured['rate'][0]):.1f} bpm", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def draw_speed_vs_pressure(out: Path, rows: list, channels: list, trace: str) -> None:
    """``speed_vs_pressure.png``: each channel's pooled speed for ``trace``
    against the window's label mean of that trace, one point per window."""
    fig, axes = plt.subplots(1, len(channels), figsize=(3.2 * len(channels), 3.2), squeeze=False)
    for ax, channel in zip(axes[0], channels):
        points = [(r[f"label_{trace}_mean"], r[f"{trace}_pooled_speed_px_per_frame"])
                  for r in rows if r["channel"] == channel and np.isfinite(r.get(f"label_{trace}_mean", np.nan))]
        if not points:
            ax.axis("off")
            continue
        x, y = map(np.asarray, zip(*points))
        ax.scatter(x, y, s=8, alpha=0.6)
        if len(x) > 2 and np.std(x) > 0:
            r = np.corrcoef(x, y)[0, 1]
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 2)
            ax.plot(xs, slope * xs + intercept, "r", lw=0.8)
            ax.set_title(f"{channel}: r {r:+.2f}, {slope:+.3f} px/frame per mmHg", fontsize=7)
        else:
            ax.set_title(channel, fontsize=7)
        ax.set_xlabel(f"label {trace} mean (mmHg)", fontsize=7)
        ax.set_ylabel("pooled speed (px/frame)", fontsize=7)
        ax.tick_params(labelsize=6)
    n_windows = len(rows) // max(len(channels), 1)
    fig.suptitle(f"{trace}: mask-pooled phase speed against the window's pressure, {n_windows} windows", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=110)
    plt.close(fig)

#: Propagation arrows: the box over which the mean phasor is smoothed before
#: its gradient is read, the phase coherence below which a pixel does not
#: vote on the direction, and the arrow's length in grid cells.
PHASE_SMOOTHING = 3
PHASE_COHERENCE_FLOOR = 0.6              # a noise pixel's R on a 10 s window reaches 0.6 (experiment 13)
ARROW_BLOCK = 2                          # grid cells per arrow
ARROW_NEIGHBOURHOOD = 3                  # blocks over which arrows must agree to stay opaque
#: Magnified GIF: the band-passed pulse's peak-to-peak swing as a fraction
#: of the grey range, after magnification.
MAGNIFICATION_SWING = 0.8


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
    """``propagation.png``: per channel the mean frame, each trace's
    pulsatile region tinted in its colour, and a grid of arrows pointing the
    way the pulse travels, coloured by the trace that holds each block and
    faded by the block's mass and its disagreement with its neighbours."""
    clip = captured["clip"][0]
    weight = coarse_weight(captured)                                         # (C, h, w, S)
    h, w = weight.shape[1:3]
    b = ARROW_BLOCK
    tints = np.asarray(((1.0, 0.15, 0.15), (0.15, 0.6, 1.0), (0.2, 1.0, 0.2), (1.0, 0.6, 0.1)))
    fig, axes = plt.subplots(1, len(channels), figsize=(5.0 * len(channels), 5.3), squeeze=False)
    for c, (channel, ax) in enumerate(zip(channels, axes[0])):
        frame = F.interpolate(clip[c].mean(dim=0)[None, None], size=(h, w), mode="area")[0, 0]
        grey = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-6)).cpu().numpy()
        rgb = np.repeat(grey[..., None], 3, axis=-1) * 0.7 + 0.15
        mean, coherence = relative_phase(captured, c)
        total = weight[c].sum(dim=-1)                                        # (h, w)
        if float(total.max()) <= 0:
            ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
            ax.set_title(channel, fontsize=9)
            ax.axis("off")
            continue
        ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
        dx, dy, mass, agreement = propagation_field(mean, coherence, total)
        hb, wb = mass.shape
        # Each block's arrow takes the colour of the trace with the most of
        # the block's pulsatile mass.
        block_share = F.avg_pool2d(rearrange(weight[c], "h w s -> s 1 h w"), b, ceil_mode=True)[:, 0]   # (S, hb, wb)
        dominant = block_share.argmax(dim=0).cpu().numpy()
        alpha = (np.sqrt(mass.cpu().numpy()) * agreement.cpu().numpy()).clip(0, 1)
        ys, xs = np.mgrid[0:hb, 0:wb] * b + (b - 1) / 2
        u, v = dx.cpu().numpy(), dy.cpu().numpy()
        for iy in range(hb):
            for ix in range(wb):
                a = float(alpha[iy, ix])
                if a < 0.2 or not np.isfinite(u[iy, ix]):
                    continue
                half = b * 0.6
                start = (xs[iy, ix] - u[iy, ix] * half, ys[iy, ix] - v[iy, ix] * half)
                end = (xs[iy, ix] + u[iy, ix] * half, ys[iy, ix] + v[iy, ix] * half)
                tint = tints[dominant[iy, ix]]
                # The trace's colour with a white edge, so it reads on any grey.
                ax.annotate("", xy=end, xytext=start,
                            arrowprops={"arrowstyle": "-|>,head_width=0.3,head_length=0.45",
                                        "color": (1, 1, 1, a * 0.9), "lw": 2.6, "shrinkA": 0, "shrinkB": 0})
                ax.annotate("", xy=end, xytext=start,
                            arrowprops={"arrowstyle": "-|>,head_width=0.3,head_length=0.45",
                                        "color": (*tint, a), "lw": 1.5, "shrinkA": 0, "shrinkB": 0})
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_title(channel, fontsize=9)
        ax.axis("off")
    legend = ", ".join(f"{trace} {name}" for trace, name in zip(traces, ("red", "blue", "green", "orange")))
    fig.suptitle(f"{out.parent.name}: the direction the pulse travels, one arrow per block ({legend}); "
                 "arrows fade with low pulsatile mass and where neighbours disagree", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=130)
    plt.close(fig)


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
        image = Image.fromarray((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        frames.append(image.resize((image.width * GIF_SCALE, image.height * GIF_SCALE), Image.NEAREST))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * GIF_FRAME_STEP / fs), loop=0)


def draw_panel(out: Path, captured: dict, batch: dict, result: dict, maps: dict | None,
               channels: list, traces: list, fs: float) -> None:
    clip = captured["clip"][0]                                               # (C, T, h, w)
    pulsatility = captured["pulsatility"][0].mean(dim=1)
    masks = (captured["pulsatility"][0, ..., None] * captured["phase_masks"][0]).mean(dim=1)
    background = (captured["pulsatility"][0] * captured["background"][0]).mean(dim=1)
    n_cols = 3 + len(traces)
    fig, axes = plt.subplots(len(channels) + 1, n_cols, figsize=(3.2 * n_cols, 3.0 * (len(channels) + 1)))
    titles = ["mean frame", "pulsatility"] + [f"{t} mask x pulsatility" for t in traces] + ["background x pulsatility"]
    for c, (channel, row) in enumerate(zip(channels, health(captured, channels, traces))):
        images = [clip[c].mean(dim=0), pulsatility[c], *[masks[c, ..., s] for s in range(len(traces))], background[c]]
        # Every map is drawn at its own colour scale; the title says how much
        # mass it holds, so a vanished map cannot pass for a sharp one.
        amounts = ["", f"{row['pulsatility_effective_px']:.0f} px",
                   *[f"share {share_text(row[f'{t}_share'])}" for t in traces],
                   f"share {share_text(row['background_share'])}"]
        for col, (image, title, amount) in enumerate(zip(images, titles, amounts)):
            ax = axes[c, col]
            ax.imshow(image.cpu().numpy(), cmap="gray" if col == 0 else "magma")
            if maps is not None:
                for region, colour in (("artery", "red"), ("vein", "deepskyblue")):
                    ax.contour(maps[region].cpu().numpy().astype(float), levels=[0.5], colors=colour, linewidths=0.6)
            name = f"{channel}: {title}" if c == 0 else channel if col == 0 else ""
            ax.set_title("\n".join(part for part in (name, amount) if part), fontsize=8)
            ax.axis("off")
    for col in range(n_cols):
        axes[-1, col].axis("off")
    t = np.arange(clip.shape[1]) / fs
    for s, trace in enumerate(traces):
        ax = fig.add_subplot(len(channels) + 1, n_cols, (len(channels)) * n_cols + 1 + s)
        ax.plot(t, batch["labels"][trace][0].cpu().numpy(), "k", lw=0.8, label="label")
        ax.plot(t, result["predictions"][trace][0].detach().cpu().numpy(), "r", lw=0.8, label="prediction")
        ax.set_title(trace, fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
    rate = float(captured["rate"][0])
    label_rate = float(batch["heart_rate"][0]) if "heart_rate" in batch else float("nan")
    fig.suptitle(f"{out.parent.name}: estimated rate {rate:.1f} bpm (label {label_rate:.1f})", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))           # leave the suptitle its own line
    fig.savefig(out, dpi=110)
    plt.close(fig)


def draw_overlay(out: Path, captured: dict, channels: list, traces: list) -> None:
    """``panel.png``'s maps drawn over the channel's mean frame: the frame in
    grey, the map in magma with its alpha the map's own normalised value, so
    the mass sits on the anatomy and the empty parts of the map show the
    frame through. Each map is scaled to its own maximum, as in the panel,
    and the titles carry the same amounts."""
    clip = captured["clip"][0]                                               # (C, T, h, w)
    pulsatility = captured["pulsatility"][0].mean(dim=1)
    masks = (captured["pulsatility"][0, ..., None] * captured["phase_masks"][0]).mean(dim=1)
    background = (captured["pulsatility"][0] * captured["background"][0]).mean(dim=1)
    n_cols = 2 + len(traces)
    fig, axes = plt.subplots(len(channels), n_cols, figsize=(3.2 * n_cols, 3.0 * len(channels)))
    axes = np.atleast_2d(axes)
    titles = ["pulsatility"] + [f"{t} mask x pulsatility" for t in traces] + ["background x pulsatility"]
    for c, (channel, row) in enumerate(zip(channels, health(captured, channels, traces))):
        frame = clip[c].mean(dim=0)
        grey = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-6)).cpu().numpy()
        images = [pulsatility[c], *[masks[c, ..., s] for s in range(len(traces))], background[c]]
        amounts = [f"{row['pulsatility_effective_px']:.0f} px",
                   *[f"share {share_text(row[f'{t}_share'])}" for t in traces],
                   f"share {share_text(row['background_share'])}"]
        for col, (image, title, amount) in enumerate(zip(images, titles, amounts)):
            ax = axes[c, col]
            weight = image.cpu().numpy()
            weight = weight / (weight.max() + 1e-12)
            ax.imshow(grey, cmap="gray", vmin=0.0, vmax=1.0)
            ax.imshow(weight, cmap="magma", vmin=0.0, vmax=1.0, alpha=np.clip(weight, 0.0, 1.0) * 0.9)
            name = f"{channel}: {title}" if c == 0 else channel if col == 0 else ""
            ax.set_title("\n".join(part for part in (name, amount) if part), fontsize=8)
            ax.axis("off")
    rate = float(captured["rate"][0])
    fig.suptitle(f"{out.parent.name}: maps over the mean frame, estimated rate {rate:.1f} bpm", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def draw_gif(out: Path, captured: dict, channel_index: int, traces: list, fs: float) -> None:
    """The channel's frames with the window's pulsatility map (green) and
    the first two traces' pulsatile masks (red, blue) over them, at the
    interface's frame rate (the masks are one map per window, so only the
    frame underneath moves)."""
    clip = captured["clip"][0, channel_index]                                # (T, h, w)
    frame = (clip - clip.min()) / (clip.max() - clip.min() + 1e-6)
    weight = captured["pulsatility"][0, channel_index, ..., None] * captured["phase_masks"][0, channel_index]
    def norm(x):
        return (x / (x.amax(dim=(-2, -1), keepdim=True) + 1e-9)).clamp(0, 1)
    pulse = norm(captured["pulsatility"][0, channel_index])
    red = norm(weight[..., 0]) if len(traces) > 0 else torch.zeros_like(pulse)
    blue = norm(weight[..., 1]) if len(traces) > 1 else torch.zeros_like(pulse)
    # Beside it, the fundamental's wrapped phase on the field's grid, cyclic
    # colours, the pulsatility map as its alpha over the (coarse) frame.
    harmonic = captured["harmonic"][0, channel_index]                        # (T, h, w) complex
    coarse_pulse = captured["pulsatility_coarse"][0, channel_index]
    coarse_alpha = (coarse_pulse / (coarse_pulse.max() + 1e-12)).clamp(0, 1)[..., None]
    hue = torch.angle(harmonic) / (2 * np.pi) + 0.5                          # (T, h, w) in [0, 1)
    cyclic = matplotlib.colormaps["hsv"]
    frames = []
    for i in range(0, clip.shape[0], GIF_FRAME_STEP):
        grey = frame[i]
        rgb = torch.stack([grey * 0.5 + 0.5 * red[i], grey * 0.5 + 0.5 * pulse[i], grey * 0.5 + 0.5 * blue[i]], dim=-1)
        image = Image.fromarray((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        image = image.resize((image.width * GIF_SCALE, image.height * GIF_SCALE), Image.NEAREST)
        coarse_grey = F.interpolate(grey[None, None], size=harmonic.shape[-2:], mode="area")[0, 0, ..., None]
        colour = torch.from_numpy(cyclic(hue[i].cpu().numpy())[..., :3]).to(coarse_grey)
        phase_rgb = coarse_grey * (1 - coarse_alpha) + colour * coarse_alpha
        phase = Image.fromarray((phase_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        phase = phase.resize(image.size, Image.NEAREST)
        both = Image.new("RGB", (image.width * 2 + GIF_SCALE, image.height))
        both.paste(image, (0, 0))
        both.paste(phase, (image.width + GIF_SCALE, 0))
        frames.append(both)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * GIF_FRAME_STEP / fs), loop=0)


def main(argv=None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        path = epoch_checkpoint(args.checkpoint, args.epoch)
        checkpoint = load_checkpoint(path)
        setup = rebuild(checkpoint["config"], str(path))
        interface, model_config = setup.interface, setup.model
        configs = load_dataset_configs([args.dataset])
        stores = load_stores(configs)
        if args.test_participant_id is not None:
            stores = hold_out_participant(stores, args.dataset, args.test_participant_id).test
        dataset = test_windows(Split(train={}, test={args.dataset: stores[args.dataset]}), interface)
    except (ValueError, FileNotFoundError) as err:                 # ConfigError is a ValueError
        parser.error(str(err))
    subset = limit_windows(dataset, args.windows) if args.windows > 0 else dataset
    indices = list(subset.indices) if hasattr(subset, "indices") else list(range(len(dataset)))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    model = build_model(model_config, interface)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    backbone = model.backbone
    if not hasattr(backbone, "capture"):
        raise ConfigError(f"{model_config.NAME} keeps no intermediates to show")
    backbone.capture = True

    out_root = path.parent / VISUALS_DIR / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)
    channels, traces = list(interface.CHANNELS), list(interface.TRACES)
    all_scores = []
    for idx in indices:
        i, start = dataset.windows[idx]
        sample = dataset.samples[i]
        item = dataset[idx]
        batch = move_to_device(default_collate([item]), device)
        with torch.no_grad():
            result = model(batch)
        captured = backbone.captured
        name = f"{sample.store.stem}_{sample.perspective}_{int(item['metadata']['start_frame'])}"
        folder = out_root / name
        maps = vessel_maps(sample.store, tuple(captured["pulsatility"].shape[-2:]))
        if maps is not None:
            maps = {k: v.to(device) for k, v in maps.items()}
        vessel_rows = scores(captured, maps, channels, traces) if maps is not None else [{}] * len(channels)
        levels = {}
        for trace in traces:
            label = batch["labels"][trace][0] if "labels" in batch and trace in batch["labels"] else None
            finite = label[torch.isfinite(label)] if label is not None else None
            has = finite is not None and len(finite) > 0
            levels[f"label_{trace}_mean"] = float(finite.mean()) if has else float("nan")
            levels[f"label_{trace}_max"] = float(torch.quantile(finite, 0.95)) if has else float("nan")
        for row, kin_row, vessel_row in zip(health(captured, channels, traces),
                                            kinematic_summary(captured, channels, traces), vessel_rows):
            all_scores.append({"window": name, "rate_bpm": float(captured["rate"][0]), **levels, **row,
                               **{k: v for k, v in kin_row.items() if k != "channel"},
                               **{k: v for k, v in vessel_row.items() if k != "channel"}})
        if args.no_images:
            continue
        folder.mkdir(exist_ok=True)
        draw_panel(folder / "panel.png", captured, batch, result, maps, channels, traces, interface.FS)
        draw_overlay(folder / "overlay.png", captured, channels, traces)
        draw_kinematics(folder / "kinematics.png", captured, batch, channels, traces, interface.FS)
        draw_propagation(folder / "propagation.png", captured, channels, traces)
        depth = channels.index("D") if "D" in channels else 0
        draw_gif(folder / "masks.gif", captured, depth, traces, interface.FS)
        for channel in ("G", "I"):
            if channel in channels:
                draw_magnified_gif(folder / f"pulse_magnified_{channel}.gif", captured,
                                   channels.index(channel), interface.FS)
        print(f"drew {folder}")
    if all_scores:
        with open(out_root / "scores.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_scores[0]))
            writer.writeheader()
            writer.writerows(all_scores)
        print(f"scores: {out_root / 'scores.csv'}")
        if len(indices) > 12 and traces:
            draw_speed_vs_pressure(out_root / "speed_vs_pressure.png", all_scores, channels, traces[0])
            print(f"drew {out_root / 'speed_vs_pressure.png'}")
    return out_root


if __name__ == "__main__":
    main()
