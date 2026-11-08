"""Show what a CardioConv or CardioHydra run looked at, window by window.

    uv run scripts/visualise.py --checkpoint runs/dev_cardio/exp2_cardioconv_synth \\
        --dataset synthetic_neck --windows 6

``--checkpoint`` is a run directory written by ``scripts/run.py`` (its latest
epoch, or ``--epoch``), one of its ``epoch_NN`` folders or a ``model.pt``.
Every strided window of ``--dataset`` is a candidate (one participant's with
``--test-participant-id``); ``--windows`` evenly spaced ones are run with the
backbone's ``capture`` switch on, and for each one the script writes, under
``<epoch dir>/visuals/<dataset>/<recording>_<perspective>_<start frame>/``:

- ``pulsatility_mask.png``: one row, the window's pulsatility map over each
  channel's mean frame (grey), the map's value as its alpha, titled with
  the channel's name only; sized to an A4 text block.
- ``phase_masks.png``: that row, then one row per trace with the trace's
  mask weighted by pulsatility: how the pulsatile region is split between
  the traces. The background mask is not drawn.
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
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError                                # noqa: E402
from src.dataset_config import load_dataset_configs              # noqa: E402
from src.datasets import Split, hold_out_participant, load_stores  # noqa: E402
from src.experiment import (                                     # noqa: E402
    epoch_checkpoint, limit_windows, load_checkpoint, rebuild, test_windows,
)
from src.models import build_model                               # noqa: E402
from src.trainer import move_to_device                           # noqa: E402
from src.visualise import VISUALS_DIR                            # noqa: E402
from src.visualise.kinematics import draw_speed_vs_pressure      # noqa: E402
from src.visualise.scores import score_window, vessel_maps, write_scores   # noqa: E402
from src.visualise.window import draw_window                     # noqa: E402

#: ``speed_vs_pressure.png`` is drawn once more windows than this were run.
SCATTER_MIN_WINDOWS = 12


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
        maps = vessel_maps(sample.store, tuple(captured["pulsatility"].shape[-2:]))
        if maps is not None:
            maps = {k: v.to(device) for k, v in maps.items()}
        all_scores.extend(score_window(name, captured, batch, maps, channels, traces))
        if args.no_images:
            continue
        draw_window(out_root / name, captured, batch, channels, traces, interface.FS)
        print(f"drew {out_root / name}")
    if all_scores:
        write_scores(out_root / "scores.csv", all_scores)
        print(f"scores: {out_root / 'scores.csv'}")
        if len(indices) > SCATTER_MIN_WINDOWS and traces:
            draw_speed_vs_pressure(out_root / "speed_vs_pressure.png", all_scores, channels, traces[0])
            print(f"drew {out_root / 'speed_vs_pressure.png'}")
    return out_root


if __name__ == "__main__":
    main()
