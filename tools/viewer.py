"""Cache viewer — browse a zarr cache in napari.

The side panel lists every store in a cache directory. Picking one shows each
checked perspective's modalities as time-aligned layers, tiled side by side,
and a trace panel below with a cursor that follows the time slider; clicking a
trace seeks the video there. The slider is in seconds on the store's common
clock, so perspectives and modalities with different rates or start times
stay aligned. Frames are read lazily, chunk by chunk, as they are shown, so a
store opens instantly whatever its length. The event camera (``ev``) is shown
as signed polarity counts binned at the "Event bin" width. The layout read is
docs/cache-contract.md.

    uv run python tools/viewer.py                 # pick the cache folder
    uv run python tools/viewer.py <cache-dir>     # open a cache
    uv run python tools/viewer.py <store.zarr>    # open a cache at that store

Keys: PageDown / PageUp step to the next / previous store; the slider's play
button (or ctrl+alt+P) plays the recording.
"""
import argparse
import collections
import functools
import re
from pathlib import Path

import napari
import numpy as np
import zarr
from einops import rearrange
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from napari.utils.colormaps import Colormap
from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QScrollArea,
    QVBoxLayout, QWidget)

#: Groups of a modality that are not traces.
NOT_TRACES = {"timestamps_us", "video", "x", "y", "p"}
COLORMAPS = {"gr": "gray", "ir": "gray", "depth": "turbo", "t": "inferno"}
#: Negative polarity magenta, positive green, no events dark gray (the
#: sensor stays visible against the black canvas).
EVENTS = Colormap([[1, 0, 1, 1], [0.15, 0.15, 0.15, 1], [0, 1, 0, 1]], name="events")


def modality_order(name):
    """RGB first, the event camera last, the rest alphabetical."""
    return name != "rgb", name == "ev", name


def natural_key(name):
    """Sort ``2.zarr`` before ``10.zarr``."""
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", name)]


def data(group, name):
    """The array at ``<name>/data`` (every array in a store lives there)."""
    return group[name]["data"]


def traces_of(modality):
    """Trace group names of a modality, as the store holds them."""
    return sorted(k for k in modality.group_keys() if k not in NOT_TRACES)


def contrast(video):
    """Display range of a ``(C, T, H, W)`` video from its middle chunk."""
    if video.dtype == np.uint8:
        return 0, 255
    middle = video.shape[1] // 2
    sample = np.asarray(video[:, middle:middle + video.chunks[1]], dtype=np.float64)
    sample = sample[np.isfinite(sample) & (sample != 0)]   # 0 = invalid depth
    if not sample.size:
        return 0, 1
    low, high = np.percentile(sample, (0.5, 99.5))
    return float(low), float(high if high > low else low + 1)


class Frames:
    """A lazy ``(T, ...)`` frame stack napari slices like an array.

    ``read_block(b)`` returns frames ``[b * block, (b + 1) * block)``; the
    last few blocks are kept, so playing through a store reads each zarr
    chunk once and a slider step is a cache hit.
    """

    def __init__(self, read_block, shape, dtype, block):
        self.shape, self.dtype, self.block = tuple(shape), np.dtype(dtype), block
        self.ndim, self.size = len(self.shape), int(np.prod(self.shape))
        self._read = functools.lru_cache(maxsize=8)(read_block)

    def __len__(self):
        return self.shape[0]

    def _frame(self, t):
        block, offset = divmod(int(t) % self.shape[0], self.block)
        return self._read(block)[offset]

    def __getitem__(self, key):
        time, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        if isinstance(time, slice):
            frames = [self._frame(t) for t in range(*time.indices(self.shape[0]))]
            stack = np.stack(frames) if frames else np.zeros(
                (0,) + self.shape[1:], self.dtype)
            return stack[(slice(None),) + rest]
        return self._frame(time)[rest]


def frame_layer(video, stamps_us, t0_us, name, modality):
    """napari ``add_image`` arguments for a ``(C, T, H, W)`` video."""
    dt = float(np.median(np.diff(stamps_us))) / 1e6 if len(stamps_us) > 1 else 1.0
    kwargs = {"name": name, "scale": (dt, 1, 1),
              "translate": ((stamps_us[0] - t0_us) / 1e6, 0, 0),
              "contrast_limits": contrast(video)}
    channels, length, height, width = video.shape
    block = video.chunks[1]                  # one read per zarr chunk along T

    def read_block(b):
        chunk = video[:, b * block:(b + 1) * block]
        if channels == 3:
            return rearrange(chunk, "c t h w -> t h w c")
        return chunk[0]

    if channels == 3:
        return (Frames(read_block, (length, height, width, 3), video.dtype, block),
                {**kwargs, "rgb": True})
    return (Frames(read_block, (length, height, width), video.dtype, block),
            {**kwargs, "colormap": COLORMAPS.get(modality, "gray")})


def event_layer(perspective, stamps_us, t0_us, bin_s, name):
    """Signed polarity counts of the ``ev`` modality, one lazy frame per bin."""
    group = perspective["ev"]
    xs, ys, ps = data(group, "x"), data(group, "y"), data(group, "p")
    height = perspective.attrs.get("sensor_height") or int(np.max(ys[:])) + 1
    width = perspective.attrs.get("sensor_width") or int(np.max(xs[:])) + 1
    width_us = bin_s * 1e6
    edges = stamps_us[0] + width_us * np.arange(
        int(np.ceil((stamps_us[-1] - stamps_us[0] + 1) / width_us)) + 1)
    bounds = np.searchsorted(stamps_us, edges)

    def read_block(i):
        lo, hi = bounds[i], bounds[i + 1]
        flat = ys[lo:hi].astype(np.int64) * width + xs[lo:hi]
        signs = np.where(ps[lo:hi] > 0, 1.0, -1.0)
        counts = np.bincount(flat, weights=signs, minlength=height * width)
        return rearrange(counts.astype(np.float32), "(t h w) -> t h w",
                         t=1, h=height)

    frames = Frames(read_block, (len(bounds) - 1, height, width), np.float32, 1)
    return frames, {
        "name": name, "colormap": EVENTS, "contrast_limits": (-3, 3),
        "scale": (bin_s, 1, 1),
        "translate": ((stamps_us[0] - t0_us) / 1e6 + bin_s / 2, 0, 0)}


class TracePanel(QWidget):
    """One axis per trace, a cursor at the slider's time, click to seek."""

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.figure = Figure(figsize=(8, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(280)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.window = QDoubleSpinBox(suffix=" s", minimum=0, maximum=3600,
                                     value=10, singleStep=5, decimals=1)
        self.window.setToolTip("Visible span of the traces; 0 shows the "
                               "whole recording. The view pages to follow "
                               "the cursor.")
        self.window.valueChanged.connect(lambda _: self._page(force=True))
        bar = QHBoxLayout()
        bar.addWidget(self.toolbar)
        bar.addStretch()
        bar.addWidget(QLabel("Window"))
        bar.addWidget(self.window)
        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.canvas)
        self.series, self.cursors, self.readouts = [], [], []
        self.background, self.duration = None, 0.0
        self.canvas.mpl_connect("draw_event", self._grab_background)
        self.canvas.mpl_connect("button_press_event", self._seek)
        viewer.dims.events.point.connect(lambda _: self.update_cursor())

    def show(self, series):
        """Plot ``[(name, units, time_s, values), ...]``."""
        self.figure.clear()
        self.series, self.cursors, self.readouts = series, [], []
        if not series:
            self.figure.text(0.5, 0.5, "no traces", ha="center", va="center")
            self.canvas.draw_idle()
            return
        axes = self.figure.subplots(len(series), 1, sharex=True, squeeze=False)[:, 0]
        for ax, (name, units, time_s, values) in zip(axes, series):
            ax.plot(time_s, values, lw=0.8, color="C0")
            ax.set_ylabel(f"{name}\n[{units}]", rotation=0, ha="right", va="center")
            ax.margins(x=0)
            self.cursors.append(ax.axvline(time_s[0], color="C3", lw=1, animated=True))
            self.readouts.append(ax.text(
                0.995, 0.92, "", transform=ax.transAxes, ha="right", va="top",
                animated=True, fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.7, "lw": 0}))
        axes[-1].set_xlabel("time [s]")
        self.duration = max(float(t[-1]) for _, _, t, _ in series)
        self._page(force=True)
        self.canvas.draw_idle()

    def _now(self):
        return float(self.viewer.dims.point[0]) if self.viewer.dims.ndim else 0.0

    def _page(self, force=False):
        """Move the x-range a page when the cursor leaves it; True if moved.

        A zoom made with the toolbar is kept until the cursor leaves it.
        """
        if not self.series:
            return False
        span, now = self.window.value(), self._now()
        ax = self.figure.axes[0]
        low, high = ax.get_xlim()
        if not span or span >= self.duration:
            limits = (0.0, self.duration)
        elif not force and low <= now <= high:
            return False
        else:
            start = max(0.0, min(now - 0.1 * span, self.duration - span))
            limits = (start, start + span)
        if np.allclose((low, high), limits):
            return False
        ax.set_xlim(*limits)
        self.canvas.draw_idle()
        return True

    def _grab_background(self, _event):
        # A full draw runs inside Qt's paint event, so the cursor is drawn into
        # that frame here; blit() would repaint() recursively from inside it.
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        self._draw_cursor()

    def update_cursor(self):
        if not self.series or self._page():
            return                                   # the redraw draws it
        if self.background is None:
            return
        self.canvas.restore_region(self.background)
        self._draw_cursor()
        self.canvas.blit(self.figure.bbox)

    def _draw_cursor(self):
        if not self.series:
            return
        now = self._now()
        for ax, line, text, (_, units, time_s, values) in zip(
                self.figure.axes, self.cursors, self.readouts, self.series):
            line.set_xdata([now, now])
            i = min(np.searchsorted(time_s, now), len(values) - 1)
            text.set_text(f"{values[i]:.4g} {units}")
            ax.draw_artist(line)
            ax.draw_artist(text)

    def _seek(self, event):
        if event.inaxes is None or event.button != 1 or self.toolbar.mode:
            return
        self.viewer.dims.set_point(0, float(event.xdata))


class FitOnResize(QObject):
    """Qt event filter: ``reset_view`` once a resize of the canvas settles."""

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.timer = QTimer(singleShot=True, interval=100)
        self.timer.timeout.connect(viewer.reset_view)

    def eventFilter(self, obj, event):  # noqa: N802 (Qt's name)
        if event.type() == QEvent.Resize:
            self.timer.start()
        return False


class CachePanel(QWidget):
    """Folder, store list, perspective checklist, trace source, store attrs."""

    def __init__(self, viewer, traces):
        super().__init__()
        self.viewer, self.traces = viewer, traces
        self.cache, self.root, self.stores = None, None, []
        self.t0, self.trace_sources = 0.0, []
        self.unchecked = set()          # perspective names, kept across stores

        self.folder = QLabel("no cache folder")
        self.folder.setWordWrap(True)
        choose = QPushButton("Open cache folder…")
        choose.clicked.connect(self.choose_folder)
        self.search = QLineEdit(placeholderText="filter stores")
        self.search.textChanged.connect(self._filter)
        self.store_list = QListWidget()
        self.store_list.currentItemChanged.connect(lambda item, _: self._open(item))
        self.perspectives = QListWidget()
        self.perspectives.setMaximumHeight(110)
        self.perspectives.itemChanged.connect(self._perspective_toggled)
        self.source = QComboBox()
        self.source.currentIndexChanged.connect(lambda _: self._plot_traces())
        self.bin = QDoubleSpinBox(suffix=" ms", minimum=1, maximum=1000,
                                  value=33.3, decimals=1)
        self.bin.editingFinished.connect(self.show_store)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info.setAlignment(Qt.AlignTop)

        form = QFormLayout()
        form.addRow("Traces from", self.source)
        form.addRow("Event bin", self.bin)
        layout = QVBoxLayout(self)
        for widget in (choose, self.folder, self.search, self.store_list,
                       QLabel("Perspectives"), self.perspectives):
            layout.addWidget(widget)
        layout.addLayout(form)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(self.info)
        layout.addWidget(scroll)
        layout.setStretch(3, 2)
        layout.setStretch(layout.count() - 1, 1)

        viewer.bind_key("PageDown", lambda _: self.step(1), overwrite=True)
        viewer.bind_key("PageUp", lambda _: self.step(-1), overwrite=True)
        # napari fits the view to the canvas it has when a store opens, which
        # at start-up is its 800x600 default, not the maximised window; refit
        # whenever the canvas is resized.
        self._refit = FitOnResize(viewer)
        viewer.window._qt_viewer.canvas.native.installEventFilter(self._refit)

    # -- cache and store selection ------------------------------------------
    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Cache folder (the directory holding the *.zarr stores)",
            str(self.cache or Path.cwd()))
        if folder:
            self.load_cache(Path(folder))

    def load_cache(self, path):
        """Open a cache directory, or the directory of a single store.

        A directory with no stores of its own is searched one level down, so
        picking the folder that holds several caches (``D:/neckflix_zarr``
        with ``rgbid128/`` inside) lists them all, named ``<cache>/<store>``.
        """
        path = Path(path).resolve()
        store = path if path.suffix == ".zarr" else None
        self.cache = path.parent if store else path
        found = list(self.cache.glob("*.zarr")) or list(self.cache.glob("*/*.zarr"))
        names = {candidate: candidate.relative_to(self.cache).with_suffix("").as_posix()
                 for candidate in found}
        self.stores = sorted(found, key=lambda candidate: natural_key(names[candidate]))
        self.folder.setText(f"{self.cache}\n{len(self.stores)} stores")
        self.store_list.blockSignals(True)
        self.store_list.clear()
        for candidate in self.stores:
            item = QListWidgetItem(names[candidate])
            item.setData(Qt.UserRole, str(candidate))
            self.store_list.addItem(item)
        self.store_list.blockSignals(False)
        if self.stores:
            index = self.stores.index(store) if store in self.stores else 0
            self.store_list.setCurrentRow(index)
        self._filter(self.search.text())

    def _filter(self, text):
        for row in range(self.store_list.count()):
            item = self.store_list.item(row)
            item.setHidden(text.lower() not in item.text().lower())

    def step(self, direction):
        rows = [r for r in range(self.store_list.count())
                if not self.store_list.item(r).isHidden()]
        if not rows:
            return
        current = self.store_list.currentRow()
        later = [r for r in rows if (r - current) * direction > 0]
        self.store_list.setCurrentRow(
            (later[0] if direction > 0 else later[-1]) if later else current)

    def _open(self, item):
        if item is None:
            return
        path = Path(item.data(Qt.UserRole))
        try:
            self.root = zarr.open_group(str(path), mode="r")
        except Exception as error:                   # show it, keep browsing
            self.root = None
            self.info.setText(f"cannot open {path.name}: {error}")
            return
        self.perspectives.blockSignals(True)
        self.perspectives.clear()
        for name in sorted(self.root.group_keys(), key=natural_key):
            check = QListWidgetItem(name)
            check.setFlags(check.flags() | Qt.ItemIsUserCheckable)
            check.setCheckState(Qt.Unchecked if name in self.unchecked else Qt.Checked)
            self.perspectives.addItem(check)
        self.perspectives.blockSignals(False)
        self.viewer.title = f"cache viewer — {path.name}"
        try:
            self.show_store()
        except KeyError as error:                    # off-contract: say so, keep browsing
            self.root = None
            self.viewer.layers.clear()
            self.traces.show([])
            self.info.setText(
                f"<b>{path.name}</b> does not follow docs/cache-contract.md "
                f"(no {error} where the contract puts one); "
                f"<code>tools/validate_cache.py</code> lists what is wrong.")

    def _perspective_toggled(self, item):
        if item.checkState() == Qt.Checked:
            self.unchecked.discard(item.text())
        else:
            self.unchecked.add(item.text())
        self.show_store()

    def _checked(self):
        return [self.perspectives.item(r).text()
                for r in range(self.perspectives.count())
                if self.perspectives.item(r).checkState() == Qt.Checked]

    # -- display -------------------------------------------------------------
    def show_store(self):
        """Rebuild the layers and trace sources for the checked perspectives."""
        if self.root is None:
            return
        keep = self.viewer.dims.point[0] if self.viewer.dims.ndim else 0.0
        chosen = [(p, self.root[p]) for p in self._checked()]
        stamps = {(p, m): data(g[m], "timestamps_us")[:]
                  for p, g in chosen for m in sorted(g.group_keys(), key=modality_order)}
        stamps = {k: s for k, s in stamps.items() if s.size}
        t0 = min((float(s[0]) for s in stamps.values()), default=0.0)
        sources, layers = [], []
        for (p, m), clock in stamps.items():
            perspective, name = self.root[p], f"{p}/{m}"
            if "video" in perspective[m]:
                layers.append(frame_layer(
                    data(perspective[m], "video"), clock, t0, name, m))
            elif m == "ev":
                layers.append(event_layer(
                    perspective, clock, t0, self.bin.value() / 1e3, name))
            if traces_of(perspective[m]):
                sources.append((name, p, m, clock))
        # A grid tile is sized to the largest layer, so a 128 px frame beside
        # the 480 px event sensor would fill a corner of it: show every layer
        # at the tallest one's height.
        height = max((array.shape[1] for array, _ in layers), default=1)
        for array, kwargs in layers:
            zoom = height / array.shape[1]
            kwargs["scale"] = (kwargs["scale"][0], zoom, zoom)
        self._set_layers(layers)
        if self.viewer.dims.ndim:
            self.viewer.dims.axis_labels = ("t [s]",) + self.viewer.dims.axis_labels[1:]
            self.viewer.reset_view()
            self.viewer.dims.set_point(0, keep)
        self.t0, self.trace_sources = t0, sources
        previous = self.source.currentText()
        self.source.blockSignals(True)
        self.source.clear()
        self.source.addItems([s[0] for s in sources])
        # Keep the source picked last; else the first (RGB when there is one).
        names = [s[0] for s in sources]
        pick = previous if previous in names else (names or [""])[0]
        self.source.setCurrentText(pick)
        self.source.blockSignals(False)
        self._plot_traces()
        self._describe(chosen, stamps)

    def _set_layers(self, layers):
        """Show ``[(array, add_image kwargs), ...]``, reusing layers by name.

        Creating a napari layer costs most of a second (its vispy overlays),
        and consecutive stores usually hold the same perspectives and
        modalities, so a layer whose name carries over (``<perspective>/
        <modality>``, which fixes its kind) gets the new store's data in
        place and only a changed layout adds or removes any.
        """
        old = {layer.name: layer for layer in self.viewer.layers}
        reuse = {kwargs["name"] for _, kwargs in layers} & set(old)
        changed = set(old) != reuse or len(layers) != len(reuse)
        if changed:
            # In grid mode napari rebuilds every tile per added or removed layer.
            self.viewer.grid.enabled = False
            for name in set(old) - reuse:
                self.viewer.layers.remove(name)
        for array, kwargs in layers:
            if kwargs["name"] not in reuse:
                self.viewer.add_image(array, **kwargs)
                continue
            layer = old[kwargs["name"]]
            layer.data = array
            for key in ("scale", "translate", "contrast_limits"):
                setattr(layer, key, kwargs[key])
        if changed:
            # One row per perspective, its modalities across (layers are
            # added perspective by perspective, and napari fills row-major).
            per_row = collections.Counter(
                kwargs["name"].split("/")[0] for _, kwargs in layers)
            self.viewer.grid.shape = (len(per_row), max(per_row.values(), default=1))
            self.viewer.grid.spacing = 0.03
            self.viewer.grid.enabled = len(layers) > 1

    def _plot_traces(self):
        index = self.source.currentIndex()
        if self.root is None or index < 0 or index >= len(self.trace_sources):
            self.traces.show([])
            return
        _, p, m, clock = self.trace_sources[index]
        modality, series = self.root[p][m], []
        for trace in traces_of(modality):
            group = modality[trace]
            # A trace on its own clock (``ev``) carries its timestamps; a
            # frame modality's traces are index-aligned to its frames.
            own = "timestamps_us" in group
            time_us = data(group, "timestamps_us")[:] if own else clock
            values = np.asarray(data(modality, trace)[:], dtype=np.float64)
            series.append((trace, group.attrs.get("units", "?"),
                           (time_us - self.t0) / 1e6, values))
        self.traces.show(series)

    def _describe(self, chosen, stamps):
        scalars = {k: v for k, v in self.root.attrs.items()
                   if not isinstance(v, (dict, list))}
        lines = [f"<b>{k}</b>: {v}" for k, v in scalars.items()]
        for p, group in chosen:
            attrs = ", ".join(f"{k}={v}" for k, v in group.attrs.items())
            lines.append(f"<br><b>perspective {p}</b> ({attrs})")
            for m in sorted(group.group_keys(), key=modality_order):
                clock = stamps.get((p, m))
                if clock is None:
                    continue
                span = (clock[-1] - clock[0]) / 1e6
                if "video" in group[m]:
                    video = data(group[m], "video")
                    rate = (len(clock) - 1) / span if span else float("nan")
                    lines.append(f"&nbsp;&nbsp;{m}: {tuple(video.shape)} "
                                 f"{video.dtype}, {span:.1f} s @ {rate:.2f} Hz")
                else:
                    lines.append(f"&nbsp;&nbsp;{m}: {len(clock):,} events, "
                                 f"{span:.1f} s")
        self.info.setText("<br>".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?",
                        help="a cache directory or one store.zarr in it "
                             "(omit to pick the folder in a dialog)")
    args = parser.parse_args(argv)
    viewer = napari.Viewer(title="cache viewer")
    traces = TracePanel(viewer)
    panel = CachePanel(viewer, traces)
    viewer.window.add_dock_widget(panel, name="Cache", area="right")
    viewer.window.add_dock_widget(traces, name="Traces", area="bottom")
    if args.path:
        panel.load_cache(Path(args.path))
    else:
        panel.choose_folder()
    napari.run()


if __name__ == "__main__":
    main()
