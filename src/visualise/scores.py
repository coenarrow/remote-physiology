"""The numbers behind the pictures: per window and channel, how much
pulsatile mass each map holds, the kinematics pooled under each trace's
mask, the window's label levels and, on a synthetic store, how much of
the mass sits on the true vessels. Every map is drawn at its own colour
scale, so these rows are what says whether a map has vanished."""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from einops import rearrange, reduce


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


def vessel_scores(captured: dict, maps: dict, channels: list, traces: list) -> list:
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


def coarse_weight(captured: dict) -> torch.Tensor:
    """``(C, h, w, S)``: each trace's pooling weight on the field's grid,
    pulsatility times trace mask, one map per window."""
    return captured["pulsatility_coarse"][0, ..., None] * captured["phase_masks_coarse"][0]


def weighted_median(values: torch.Tensor, weights: torch.Tensor) -> float:
    """The weighted median of ``values`` (any shape) under ``weights`` of the
    same shape."""
    flat, w = rearrange(values, "... -> (...)"), rearrange(weights, "... -> (...)")
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


def label_levels(batch: dict, traces: list) -> dict:
    """``label_<trace>_mean`` and ``label_<trace>_max`` (the 95th percentile)
    of each trace's finite label samples in the window; NaN without any."""
    levels = {}
    for trace in traces:
        label = batch["labels"][trace][0] if "labels" in batch and trace in batch["labels"] else None
        finite = label[torch.isfinite(label)] if label is not None else None
        has = finite is not None and len(finite) > 0
        levels[f"label_{trace}_mean"] = float(finite.mean()) if has else float("nan")
        levels[f"label_{trace}_max"] = float(torch.quantile(finite, 0.95)) if has else float("nan")
    return levels


def score_window(name: str, captured: dict, batch: dict, maps: dict | None,
                 channels: list, traces: list) -> list:
    """One ``scores.csv`` row per channel of the window ``name``: its
    estimated rate and label levels, then :func:`health`,
    :func:`kinematic_summary` and, with ``maps``, :func:`vessel_scores`."""
    vessel_rows = vessel_scores(captured, maps, channels, traces) if maps is not None else [{}] * len(channels)
    levels = label_levels(batch, traces)
    rows = []
    for row, kin_row, vessel_row in zip(health(captured, channels, traces),
                                        kinematic_summary(captured, channels, traces), vessel_rows):
        rows.append({"window": name, "rate_bpm": float(captured["rate"][0]), **levels, **row,
                     **{k: v for k, v in kin_row.items() if k != "channel"},
                     **{k: v for k, v in vessel_row.items() if k != "channel"}})
    return rows


def write_scores(out: Path, rows: list) -> None:
    """``rows`` as a CSV with the first row's keys as its header."""
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
