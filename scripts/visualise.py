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
  and trace masks over them.

and one ``scores.csv`` per dataset: per window and channel, the pulsatility
map's effective pixel count and each trace's and the background's share of
the pulsatile mass (also printed in each map's title, because every map is
drawn at its own colour scale), and, when the dataset carries
``vessel_ids``:
per window and channel, the share of the pulsatility distribution that falls
on the neck (``neck_mask``) and on the vessel pixels, each against that
region's share of the grid (chance), and the share of each trace's pulsatile
mass inside the true artery, the true vein and the background. The
pulsatility mask is meant to find where the neck pulses, so its mass on the
neck should sit well above chance; only the phase masks separate the vessels,
so the arterial trace's mask should sit on the artery and the venous one on
the vein. Nothing is trained or scored for agreement here;
``scripts/evaluate.py`` does that.
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
                        help="evenly spaced windows to draw (default 6)")
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
    frames = []
    for i in range(0, clip.shape[0], GIF_FRAME_STEP):
        grey = frame[i]
        rgb = torch.stack([grey * 0.5 + 0.5 * red[i], grey * 0.5 + 0.5 * pulse[i], grey * 0.5 + 0.5 * blue[i]], dim=-1)
        image = Image.fromarray((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        frames.append(image.resize((image.width * GIF_SCALE, image.height * GIF_SCALE), Image.NEAREST))
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
    subset = limit_windows(dataset, args.windows)
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
        folder.mkdir(exist_ok=True)
        maps = vessel_maps(sample.store, tuple(captured["pulsatility"].shape[-2:]))
        if maps is not None:
            maps = {k: v.to(device) for k, v in maps.items()}
        vessel_rows = scores(captured, maps, channels, traces) if maps is not None else [{}] * len(channels)
        for row, vessel_row in zip(health(captured, channels, traces), vessel_rows):
            all_scores.append({"window": name, "rate_bpm": float(captured["rate"][0]), **row,
                               **{k: v for k, v in vessel_row.items() if k != "channel"}})
        draw_panel(folder / "panel.png", captured, batch, result, maps, channels, traces, interface.FS)
        draw_overlay(folder / "overlay.png", captured, channels, traces)
        depth = channels.index("D") if "D" in channels else 0
        draw_gif(folder / "masks.gif", captured, depth, traces, interface.FS)
        print(f"drew {folder}")
    if all_scores:
        with open(out_root / "scores.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_scores[0]))
            writer.writeheader()
            writer.writerows(all_scores)
        print(f"scores: {out_root / 'scores.csv'}")
    return out_root


if __name__ == "__main__":
    main()
