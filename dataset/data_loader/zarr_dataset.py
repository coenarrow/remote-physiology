"""Lazy torch datasets over external Neckflix-preprocessor zarr caches.

Ported from CardioHydra src/dataset/base.py; see the design spec at
docs/superpowers/specs/2026-08-30-neckflix-zarr-loader-design.md for the
contract and the sanctioned deviations (1-7). The cache is an external input
produced by ghcr.io/coenarrow/neckflix (>= 1.0.0); this module never writes it.
"""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import zarr

from dataset.data_loader.label_transforms import STAT_NAMES, apply_norm, finite_stats

# Sentinel for root attrs absent from a store (deviation 2).
_MISSING = object()


def _validate_filters(filters):
    """Reject overlapping include/exclude upfront (deviation 6).

    CardioHydra checks lazily inside the per-sample loop, where an overlap can
    pass silently when no sample reaches that attribute; this port validates
    unconditionally at construction.
    """
    for attribute, spec in (filters or {}).items():
        overlap = set(spec.get("include", [])) & set(spec.get("exclude", []))
        if overlap:
            raise ValueError(
                f"Overlapping include/exclude for '{attribute}': {overlap}"
            )


class BaseZarrDataset(ABC, torch.utils.data.Dataset):
    """Abstract lazy dataset over Neckflix-preprocessor zarr stores.

    Subclasses provide only ``channel_map``. Construction is metadata-only:
    no pixel data is read until ``__getitem__``.
    """

    MIN_TOOL_VERSION = (1, 0, 0)  # raw-frame cache format floor

    @property
    @abstractmethod
    def channel_map(self) -> dict[str, tuple[str, int]]:
        """Map channel names to ``(stream_group, channel_index)`` pairs."""
        ...

    def _resolve_streams(self, channels):
        """Map config channel names through ``channel_map``, preserving order."""
        cmap = self.channel_map
        plan = []
        for ch in channels:
            if ch not in cmap:
                raise ValueError(f"Unknown channel: {ch}. Valid: {list(cmap)}")
            plan.append(cmap[ch])
        return plan

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.cache_root = Path(cfg["cache_dir"])
        self.channels = cfg["channels"]
        self.labels = cfg["labels"]
        self.window_size = cfg["window_size"]
        self.window_stride = cfg.get("window_stride", self.window_size)
        self.random_windows = cfg.get("random_windows", False)
        self.filters = cfg.get("filters", {})
        self.label_norm = cfg.get("label_norm", "zscore")
        if self.label_norm not in ("zscore", "minmax"):
            raise ValueError(
                f"label_norm must be 'zscore' or 'minmax', got {self.label_norm!r}"
            )
        _validate_filters(self.filters)

        self.stream_plan = self._resolve_streams(self.channels)
        self.required_streams = sorted({s[0].lower() for s in self.stream_plan})

        self.allow_missing = cfg.get("allow_missing", False)
        self.min_channels = cfg.get("min_channels", 1)
        self.min_labels = cfg.get("min_labels", 1)
        self.present_streams: dict[tuple[str, str], list[str]] = {}
        self.present_labels: dict[tuple[str, str], list[str]] = {}
        self.stream_hw: dict[str, tuple[int, int]] | None = None

        # Pipeline: scan -> discover -> filter -> window
        self.dataset_dict = self._scan_cache()
        self.samples = self.discover_samples()
        self._filter_by_attribute(self.filters)
        self._load_windows()

    def _scan_cache(self) -> dict:
        """Walk external zarr stores in ``cache_root`` into the recording dict.

        Admission gate (see spec): unreadable stores, stores whose root attrs
        lack ``complete is True`` (identity — JSON boolean true only), and
        stores whose ``tool_version`` parses below 1.0.0 (unparseable values
        count as 0) are skipped with a warning. Groups without stream
        sub-groups (root ``events/``) and stream groups without a ``video``
        child are ignored.
        """
        if not self.cache_root.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {self.cache_root}. The zarr cache "
                "is an external input — generate it with the Neckflix "
                "preprocessor (ghcr.io/coenarrow/neckflix) first."
            )

        cache_dict: dict = {}
        for store_path in sorted(self.cache_root.glob("*.zarr")):
            try:
                root = zarr.open_group(str(store_path), mode="r")
            except Exception as err:
                warnings.warn(
                    f"Skipping {store_path.name}: unreadable store ({err})"
                )
                continue
            attrs = dict(root.attrs)
            if attrs.get("complete") is not True:
                warnings.warn(
                    f"Skipping {store_path.name}: no 'complete: true' root attr "
                    "(partial run or pre-Neckflix store); regenerate it."
                )
                continue
            version = str(attrs.get("tool_version", "0"))
            try:
                version_tuple = tuple(int(p) for p in version.split("."))
            except ValueError:
                version_tuple = (0,)
            if version_tuple < self.MIN_TOOL_VERSION:
                warnings.warn(
                    f"Skipping {store_path.name}: tool_version {version!r} predates "
                    "the raw-frame format (needs >= 1.0.0); regenerate it."
                )
                continue

            recording: dict = {"attrs": attrs}
            for perspective_name, perspective_group in root.groups():
                perspective: dict = {}
                for stream_name, stream_group in perspective_group.groups():
                    entries = {name for name, _ in stream_group.groups()}
                    if "video" not in entries:
                        continue  # non-stream group, e.g. a bare trace group
                    perspective[stream_name] = {name: {} for name in entries}
                if perspective:
                    recording[perspective_name] = perspective
            cache_dict[store_path.stem] = recording

        if not any(len(rec) > 1 for rec in cache_dict.values()):
            raise RuntimeError(
                f"No usable zarr stores under {self.cache_root} — every store "
                "was missing, incomplete, or pre-1.0.0. Regenerate the cache "
                "with the Neckflix preprocessor (ghcr.io/coenarrow/neckflix)."
            )
        return cache_dict

    def discover_samples(self) -> list[tuple[str, str]]:
        """Build the sample list, retaining partial samples when allowed.

        Records, for every admitted ``(recording, perspective)``, the streams
        and labels actually present (canonical order) in
        ``self.present_streams`` / ``self.present_labels``. With
        ``allow_missing`` a sample is kept when it has >= ``min_channels``
        present streams and >= ``min_labels`` present labels; otherwise the
        strict all-present rule applies.
        """
        samples: list[tuple[str, str]] = []
        for rec_name, rec_data in sorted(self.dataset_dict.items()):
            for perspective, perspective_data in sorted(rec_data.items()):
                if perspective == "attrs":
                    continue

                streams = perspective_data.keys()
                present_streams = [s for s in self.required_streams if s in streams]
                present_labels = [
                    lab for lab in self.labels
                    if any(lab.lower() in perspective_data[s] for s in present_streams)
                ]

                if self.allow_missing:
                    keep = (len(present_streams) >= self.min_channels
                            and len(present_labels) >= self.min_labels)
                else:
                    # Strict: every required stream present AND every stream in
                    # the perspective carries every configured label.
                    keep = (
                        len(present_streams) == len(self.required_streams)
                        and all(
                            all(lab.lower() in perspective_data[s] for lab in self.labels)
                            for s in streams
                        )
                    )
                if not keep:
                    continue

                self.present_streams[(rec_name, perspective)] = present_streams
                self.present_labels[(rec_name, perspective)] = present_labels
                samples.append((rec_name, perspective))
        return samples

    def _filter_by_attribute(self, filters=None) -> None:
        """Filter ``self.samples`` in place by attribute include/exclude.

        ``filters`` is ``{attribute: {"include": [...], "exclude": [...]}}``:
        a value in ``exclude`` drops the sample; a non-empty ``include``
        whitelists. The pseudo-attribute ``"perspective"`` compares
        ``str()``-coerced values against the sample's perspective key
        (deviation 3). A sample whose store lacks a root attr fails any
        non-empty include and passes an exclude-only filter; one UserWarning
        per affected attribute is emitted after the pass (deviation 2 —
        CardioHydra raises KeyError instead). Overlap validation already
        happened upfront in ``__init__`` (deviation 6).
        """
        filters = filters or {}
        missing: dict[str, set[str]] = {}
        filtered_samples = []
        for recording, perspective in self.samples:
            attrs = self.dataset_dict[recording]["attrs"]
            passed = True
            for attribute, spec in filters.items():
                include = spec.get("include", [])
                exclude = spec.get("exclude", [])
                if attribute == "perspective":
                    value = str(perspective)
                    include = [str(v) for v in include]
                    exclude = [str(v) for v in exclude]
                else:
                    value = attrs.get(attribute, _MISSING)
                    if value is _MISSING:
                        missing.setdefault(attribute, set()).add(recording)
                        if include:            # membership unprovable
                            passed = False
                            break
                        continue               # exclude-only: passes
                if value in exclude or (include and value not in include):
                    passed = False
                    break
            if passed:
                filtered_samples.append((recording, perspective))

        for attribute, stores in sorted(missing.items()):
            warnings.warn(
                f"Filter attribute '{attribute}' missing from store root attrs "
                f"of: {', '.join(sorted(stores))}",
                UserWarning,
            )
        self.samples = filtered_samples

    def attribute_values(self, attribute: str) -> list[str]:
        """Sorted unique values of a root attr (or 'perspective') over the
        current samples — the LOSO fold-enumeration primitive (deviation 5).

        Values are ``str()``-coerced before sorting; samples whose store lacks
        the attribute are silently skipped.
        """
        values: set[str] = set()
        for recording, perspective in self.samples:
            if attribute == "perspective":
                values.add(str(perspective))
                continue
            attrs = self.dataset_dict[recording]["attrs"]
            if attribute in attrs:
                values.add(str(attrs[attribute]))
        return sorted(values)

    def _sample_streams(self, recording_name: str, perspective: str) -> list[str]:
        """Per-sample present streams; full required set if sample unknown."""
        return self.present_streams.get(
            (recording_name, perspective), list(self.required_streams)
        )

    def _sample_traces(self, recording_name: str, perspective: str) -> list[str]:
        """Per-sample present labels lowercased; all labels if sample unknown."""
        labels = self.present_labels.get((recording_name, perspective), self.labels)
        return [lab.lower() for lab in labels]

    def _get_frame_count(self, recording_name: str, perspective: str) -> int:
        """Shortest aligned frame count for a sample, from ``num_frames`` attrs."""
        store_path = self.cache_root / f"{recording_name}.zarr"
        cam = zarr.open_group(str(store_path), mode="r")[perspective]
        length: int | None = None
        for stream_name in self._sample_streams(recording_name, perspective):
            try:
                video = cam[stream_name]["video"]
                n = int(video.attrs["num_frames"])
            except KeyError as err:
                raise RuntimeError(
                    f"{store_path.name}/{perspective}/{stream_name}: missing "
                    "video group or 'num_frames' attr; regenerate this store "
                    "with the Neckflix preprocessor."
                ) from err
            length = n if length is None else min(length, n)
        assert length is not None, f"no streams for {recording_name}/{perspective}"
        return int(length)

    def _load_windows(self) -> None:
        """Build the window index from samples (frame units).

        Strided mode emits ``range(0, frame_count - window_size + 1,
        window_stride)`` starts; random mode emits a single ``None``-start
        entry per sample (start chosen at access time). Samples shorter than
        ``window_size`` are skipped in both modes.
        """
        windows: list[tuple[str, str, int | None]] = []
        for recording_name, perspective in self.samples:
            frame_count = self._get_frame_count(recording_name, perspective)
            if frame_count < self.window_size:
                continue
            if self.random_windows:
                windows.append((recording_name, perspective, None))
                continue
            for start in range(0, frame_count - self.window_size + 1, self.window_stride):
                windows.append((recording_name, perspective, int(start)))
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def _ensure_stream_shapes(self) -> None:
        """Populate ``self.stream_hw`` (canonical (H, W) per required stream).

        Scans cached stores for each required stream's frame shape so absent
        channels can be zero-filled to a size consistent across samples.
        Freezes only once every required stream has a real shape; a stream
        never seen in any store keeps the fallback (first real shape found).
        """
        if getattr(self, "_stream_hw_complete", False):
            return
        found: dict[str, tuple[int, int]] = dict(getattr(self, "_stream_hw_found", {}))
        for rec_name, perspective in self.samples:
            store_path = self.cache_root / f"{rec_name}.zarr"
            if not store_path.exists():
                continue
            root = zarr.open_group(str(store_path), mode="r")
            try:
                cam = root[str(perspective)]
            except KeyError:
                continue
            for stream_name in self.required_streams:
                if stream_name in found:
                    continue
                try:
                    frames = cam[stream_name]["video"]["frames"]  # (C, T, H, W)
                    found[stream_name] = (int(frames.shape[-2]), int(frames.shape[-1]))
                except KeyError:
                    pass
        if not found:
            raise RuntimeError(
                "No cached streams found to infer fill shapes; check cache_dir stores"
            )
        self._stream_hw_found = found
        fallback = next(iter(found.values()))
        self.stream_hw = {s: found.get(s, fallback) for s in self.required_streams}
        self._stream_hw_complete = True

    def _window_trace(self, stream, trace_key: str, start: int, end: int) -> np.ndarray:
        """Slice one trace copy, NaN-right-padded to the window (deviation 7).

        Per-stream trailing-NaN trimming can leave a trace shorter than its
        stream's ``num_frames``; a window overlapping that tail yields a short
        slice, padded here so copies always align. CardioHydra crashes on this.
        """
        data = stream[trace_key]["data"]
        stop = min(end, int(data.shape[0]))
        sliced = np.asarray(data[start:stop], dtype=np.float64)
        if sliced.shape[0] < (end - start):
            pad = np.full((end - start) - sliced.shape[0], np.nan)
            sliced = np.concatenate([sliced, pad])
        return sliced

    @staticmethod
    def _finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
        """Position-wise mean over finite values across trace copies.

        Positions where every copy is non-finite stay NaN (absorbed by the
        deviation-4 guard downstream). Warning-free equivalent of np.nanmean.
        """
        stacked = np.stack(arrays)                       # (n_copies, T)
        finite = np.isfinite(stacked)
        counts = finite.sum(axis=0)                      # (T,)
        sums = np.where(finite, stacked, 0.0).sum(axis=0)
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    def __getitem__(self, idx: int) -> dict:
        """Load one window as the spec's nested dense dict (see class docstring).

        frames: {channel: (1, T, H, W) float32} raw pixels, zeros where the
        stream is absent; labels: {label: (T,) float32} normalised per
        ``label_norm`` with finite-only stats (deviation 4); label_stats:
        physical-unit stats that normalised each window; channel_mask /
        label_mask: scalar bools; metadata: recording_id / camera_id /
        start_frame.
        """
        rec_name, camera_id, start = self.windows[idx]
        self._ensure_stream_shapes()

        present_streams = self._sample_streams(rec_name, str(camera_id))
        present_labels = set(
            self.present_labels.get((rec_name, str(camera_id)), self.labels)
        )

        if start is None:  # random window
            n_frames = self._get_frame_count(rec_name, camera_id)
            max_start = n_frames - self.window_size
            start = (
                int(torch.randint(0, max_start + 1, (1,)).item())
                if max_start > 0
                else 0
            )
        end = start + self.window_size

        store_path = self.cache_root / f"{rec_name}.zarr"
        root = zarr.open_group(str(store_path), mode="r")
        cam_group = root[str(camera_id)]

        # --- Load present streams' frames + label trace copies ---
        stream_frames: dict[str, np.ndarray] = {}
        label_accumulators: dict[str, list[np.ndarray]] = {
            name: [] for name in self.labels
        }
        for stream_name in present_streams:
            stream = cam_group[stream_name]
            video = stream["video"]["frames"]  # (C, T, H, W) raw frames
            stream_frames[stream_name] = np.asarray(video[:, start:end])
            for label_name in self.labels:
                trace_key = label_name.lower()
                if trace_key in stream:
                    label_accumulators[label_name].append(
                        self._window_trace(stream, trace_key, start, end)
                    )

        # --- Dense frames: every channel, zeros where its stream is absent ---
        frames: dict[str, torch.Tensor] = {}
        channel_mask: dict[str, torch.Tensor] = {}
        for ch_name, (s_name, ch_idx) in zip(self.channels, self.stream_plan):
            present = s_name in stream_frames
            if present:
                arr = stream_frames[s_name][ch_idx][np.newaxis]  # (1, T, H, W)
                frames[ch_name] = torch.from_numpy(arr.copy()).float()
            else:
                h, w = self.stream_hw[s_name]
                frames[ch_name] = torch.zeros(
                    (1, self.window_size, h, w), dtype=torch.float32
                )
            channel_mask[ch_name] = torch.tensor(present, dtype=torch.bool)

        # --- Dense labels: finite-only stats + post-norm NaN zeroing (dev. 4) ---
        labels: dict[str, torch.Tensor] = {}
        label_stats: dict[str, dict[str, torch.Tensor]] = {}
        label_mask: dict[str, torch.Tensor] = {}
        for label_name in self.labels:
            arrays = label_accumulators[label_name]
            has_data = bool(arrays) and label_name in present_labels
            if has_data:
                raw = torch.from_numpy(self._finite_mean(arrays)).float()  # (T,)
                finite = torch.isfinite(raw)
                present = bool(finite.any())
            else:
                present = False
            if present:
                stats = finite_stats(raw)
                normed = torch.where(
                    finite, apply_norm(raw, stats, self.label_norm),
                    raw.new_zeros(()),
                )
            else:
                normed = torch.zeros(self.window_size, dtype=torch.float32)
                stats = {
                    name: torch.zeros((), dtype=torch.float32)
                    for name in STAT_NAMES
                }
            labels[label_name] = normed
            label_stats[label_name] = stats
            label_mask[label_name] = torch.tensor(present, dtype=torch.bool)

        recording_id = root.attrs.get("recording", rec_name)

        return {
            "frames": frames,
            "labels": labels,
            "label_stats": label_stats,
            "channel_mask": channel_mask,
            "label_mask": label_mask,
            "metadata": {
                "recording_id": recording_id,
                "camera_id": camera_id,
                "start_frame": start,
            },
        }
