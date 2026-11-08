"""``kinematics.png`` and ``speed_vs_pressure.png``: what the kinematics
stage reads off the field at the fundamental, on fixed physical scales so
maps compare across channels and windows, and the pooled speed against the
window's pressure over many windows."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.visualise.scores import coarse_weight, kinematic_summary

#: Fixed scales of the kinematic maps, so panels compare across windows.
WAVELENGTH_RANGE_PX = (3.0, 1000.0)      # log scale: the grid cell to five frames
SPEED_RANGE_PX_PER_FRAME = 100.0         # symmetric log scale, linear within 1
QUIVER_BLOCK = 3                         # field-grid cells per arrow
QUIVER_GAIN = 3.0                        # arrow length per unit heading resultant, in blocks


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
