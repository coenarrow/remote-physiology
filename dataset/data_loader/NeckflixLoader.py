"""The dataloader for the Neckflix dataset.

Details for the Neckflix Dataset see ####
If you use this dataset, please cite this paper:
C. Arrow, M. Ward, J. Eshraghian, G. Dwivedi.
"Neckflix:"
"""
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms.functional as F
import h5py
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

class NeckflixLoader(BaseLoader):
    """The data loader for the Neckflix dataset."""

    def __init__(self, name, data_path, config_data, device=None,test_participants:list = [],get_raw_resized=False, dict_output: bool = False):
        """Initializes an Neckflix dataloader.
            Args:
        """
        self.inputs = list()
        self.labels = list()
        self.name = name
        self.config_data = config_data
        self.cached_path = Path(self.config_data.CACHED_PATH)
        self.data_format = config_data.DATA_FORMAT
        self.test_participants = test_participants
        self.get_raw_resized = get_raw_resized
        self.dict_output = dict_output
        if dict_output:
            from neural_methods.signals import (resolve_channels, resolve_traces,
                                                signal_norm_overrides)
            self.slot_channels = resolve_channels(config_data)
            self.traces = resolve_traces(config_data)
            self.norm_overrides = signal_norm_overrides(config_data)
        # check if cuda is available
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        if self.cached_path.exists():
            self.load()
        else:
            raise ValueError(f"Neckflix Dataset must be preprocessed before loading\n \
                Could not find at {self.config_data.CACHED_PATH}")

    def __getitem__(self, index):
        if self.dict_output:
            filename = Path(self.inputs[index][0]).name
            chunk_id = self.inputs[index][1]
            np_input, labels, label_mask, ch_mask = self.load_recording_dict(index)
            frames = self.process_item_dict(np_input, ch_mask)
            return {'frames': frames, 'channel_mask': ch_mask,
                    'labels': labels, 'label_mask': label_mask,
                    'filename': filename, 'chunk_id': chunk_id}
        try:
            filename = Path(self.inputs[index][0]).name
            chunk_id = self.inputs[index][1]
            raw_input, raw_label = self.load_recording(index=index)
        except Exception as e:
            raise RuntimeError(f"Error loading data at index {index}: filename {filename}, chunk {chunk_id}") from e
        
        try:
            input, label, resized_input = self.process_item(raw_input, raw_label)
        except Exception as e:
            raise RuntimeError(f"Error processing data at index {index}: filename {filename}, chunk {chunk_id}") from e
        try: 
            # input shape is currently (D, C, H, W)
            if self.data_format == 'NDCHW':
                input = input
            elif self.data_format == 'NCDHW':
                input = input.transpose(1,0,2,3)
            elif self.data_format == 'NDHWC':
                input = input.transpose(0,2,3,1)
            else:
                raise ValueError('Unsupported Data Format!')
        except Exception as e:
            raise RuntimeError(f"Error formatting data at index {index}: filename {filename}, chunk {chunk_id}") from e
        
        label = label.squeeze()
        input = np.array(input, copy=True)
        label = np.array(label, copy=True)
        resized_input = np.array(resized_input, copy=True)
        # tqdm.write(f"Filename: {filename}, Chunk ID: {chunk_id}")
        # tqdm.write(f"Input shape: {input.shape}\nLabel shape: {label.shape}\nResized input shape: {resized_input.shape}\n\n")

        if self.get_raw_resized:
            return input, label, filename, chunk_id, resized_input
        else:
            return input, label, filename, chunk_id

    def get_cached_file_list(self):
        config_data = self.config_data
        selected_channels = config_data.PREPROCESS.NECKFLIX.CHANNELS
        selected_traces = config_data.PREPROCESS.NECKFLIX.TRACES
        selected_postures = config_data.PREPROCESS.NECKFLIX.POSTURES
        cached_dir = Path(config_data.CACHED_PATH)
        selected_files = []
        for file in cached_dir.glob("*.hdf5"):
            participant = file.name.split('_')[0]
            if self.name == 'train' and participant in self.test_participants:
                continue
            elif self.name == 'test' and participant not in self.test_participants:
                continue
            posture = file.name.split('_')[3]
            if posture not in selected_postures:
                continue
            if self.dict_output:
                with h5py.File(file, 'r') as f:
                    if not any(ch in f for ch in self.slot_channels):
                        continue
                selected_files.append(file)
                continue
            with h5py.File(file, 'r') as f:
                available_channels = set(f.keys())
                if not set(selected_channels).issubset(available_channels):
                    continue
                available_traces = []
                _=[available_traces.extend(list(f[channel].keys())) for channel in selected_channels]
                available_traces = set(available_traces)
                if not set(selected_traces).issubset(available_traces):
                    continue
            selected_files.append(file)
        return selected_files
    
    def load_recording(self, index) -> tuple[np.ndarray, np.ndarray]:
        h5_filepath, chunk_start_idx = self.inputs[index]
        selected_channels = self.config_data.PREPROCESS.NECKFLIX.CHANNELS
        selected_traces = self.config_data.PREPROCESS.NECKFLIX.TRACES
        chunk_size = self.config_data.PREPROCESS.CHUNK_LENGTH
        random_chunk = self.config_data.PREPROCESS.NECKFLIX.RANDOM_CHUNK

        with h5py.File(h5_filepath, 'r') as f:
            # Validate once
            if not all(ch in f for ch in selected_channels):
                raise ValueError("Not all selected channels are present in the HDF5 file.")
            for ch in selected_channels:
                if not all(tr in f[ch] for tr in selected_traces):
                    raise ValueError(
                        f"Expected {selected_traces} in {h5_filepath} but missing in channel {ch}."
                    )

            # Infer total length from first trace of first channel
            n_total = f[selected_channels[0]][selected_traces[0]].shape[0]
            n_tr = len(selected_traces)
            n_ch = len(selected_channels)
            inv_nch = 1.0 / n_ch

            # ----- 1) Build labels over full length -----
            labels_full = np.empty((n_tr, n_total), dtype=np.float32)
            for ti, tr in enumerate(selected_traces):
                acc = None
                for ch in selected_channels:
                    arr = f[ch][tr][...]  # read full trace
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32, copy=False)
                    acc = arr if acc is None else acc + arr
                labels_full[ti] = acc * inv_nch

            # ----- 2) Drop NaNs globally -----
            finite_mask = np.isfinite(labels_full).all(axis=0)
            valid_indices = np.flatnonzero(finite_mask)
            n_valid = valid_indices.shape[0]

            if n_valid == 0:
                raise ValueError(f"No finite labels in {h5_filepath}.")

            # Ensure we can take at least 1 frame
            if chunk_size > n_valid:
                # Either raise, or clamp chunk_size to n_valid; here we clamp
                chunk_len = n_valid
            else:
                chunk_len = chunk_size

            # ----- 3) Choose chunk in valid index space -----
            if random_chunk:
                # Start index in "valid_indices" space
                max_start = n_valid - chunk_len
                start_pos = 0 if max_start <= 0 else np.random.randint(0, max_start + 1)
            else:
                # Sequential chunks over valid frames
                start_pos = chunk_start_idx * chunk_size
                if start_pos >= n_valid:
                    raise IndexError(
                        f"chunk_start_idx {chunk_start_idx} out of range for {n_valid} valid frames."
                    )

            end_pos = min(start_pos + chunk_len, n_valid)
            sel_idx = valid_indices[start_pos:end_pos]  # absolute frame indices after drop

            # Slice labels to selected frames, shape (time, traces)
            np_label = labels_full[:, sel_idx].T

            # ----- 4) Read frames only for selected indices -----
            frames_list = []
            for ch in selected_channels:
                dset = f[ch]['frames']
                frames = dset[sel_idx, ...]  # sel_idx are absolute indices
                frames_list.append(frames)

        # Stack channels last and cast once
        np_input = np.stack(frames_list, axis=-1).astype(np.float32, copy=False)

        return np_input, np_label

    def load_recording_dict(self, index):
        """Dict-mode read: zero-filled channel slots + per-signal labels/masks."""
        from neural_methods.signals import normalize_signal
        h5_filepath, chunk_start_idx = self.inputs[index]
        chunk_size = self.config_data.PREPROCESS.CHUNK_LENGTH
        random_chunk = self.config_data.PREPROCESS.NECKFLIX.RANDOM_CHUNK
        slots = self.slot_channels

        with h5py.File(h5_filepath, 'r') as f:
            present = [ch in f for ch in slots]
            ch0 = slots[present.index(True)]
            keys = f[ch0]
            probe = keys['timestamps'] if 'timestamps' in keys else \
                keys[next(k for k in keys if k != 'frames')]
            n_total = probe.shape[0]

            # ----- labels over full length, averaged over channels that carry them -----
            labels_full = {}
            label_mask = {}
            for tr in self.traces:
                acc, n_carrier = None, 0
                for ch, ok in zip(slots, present):
                    if ok and tr in f[ch]:
                        arr = f[ch][tr][...].astype(np.float32)
                        acc = arr if acc is None else acc + arr
                        n_carrier += 1
                if n_carrier:
                    labels_full[tr] = acc / n_carrier
                    label_mask[tr] = 1.0
                else:
                    labels_full[tr] = np.zeros(n_total, dtype=np.float32)
                    label_mask[tr] = 0.0

            # ----- valid frames: finite across PRESENT traces only -----
            present_traces = [tr for tr in self.traces if label_mask[tr] > 0]
            if present_traces:
                finite = np.all([np.isfinite(labels_full[tr]) for tr in present_traces], axis=0)
            else:
                finite = np.ones(n_total, dtype=bool)
            valid_indices = np.flatnonzero(finite)
            n_valid = valid_indices.shape[0]
            if n_valid == 0:
                raise ValueError(f"No finite labels in {h5_filepath}.")
            chunk_len = min(chunk_size, n_valid)

            if random_chunk:
                max_start = n_valid - chunk_len
                start_pos = 0 if max_start <= 0 else np.random.randint(0, max_start + 1)
            else:
                start_pos = chunk_start_idx * chunk_size
                if start_pos >= n_valid:
                    raise IndexError(
                        f"chunk_start_idx {chunk_start_idx} out of range for {n_valid} valid frames.")
            end_pos = min(start_pos + chunk_len, n_valid)
            sel_idx = valid_indices[start_pos:end_pos]

            # ----- frames per slot (zeros where absent) -----
            frames_list = []
            for ch, ok in zip(slots, present):
                if ok:
                    frames_list.append(f[ch]['frames'][sel_idx, ...].astype(np.float32))
                else:
                    hh, ww = f[ch0]['frames'].shape[1:3]
                    frames_list.append(np.zeros((len(sel_idx), hh, ww), dtype=np.float32))

        np_input = np.stack(frames_list, axis=-1)                # (T, H, W, n_slots)
        labels = {tr: normalize_signal(labels_full[tr][sel_idx], tr, self.norm_overrides)
                       .astype(np.float32) if label_mask[tr] > 0
                  else np.zeros(len(sel_idx), dtype=np.float32)
                  for tr in self.traces}
        ch_mask = np.array(present, dtype=np.float32)
        return np_input, labels, {tr: np.float32(label_mask[tr]) for tr in self.traces}, ch_mask

    def process_item_dict(self, np_input, ch_mask):
        """Resize then CONCATENATE each DATA_TYPE transform along channels.

        np_input: (T, H, W, n_slots) -> frames: (n_types * n_slots, T, H, W)
        Zero-filled slots are re-zeroed after each transform (diffnorm/zstand
        of a constant-zero channel would otherwise produce 0/0 artifacts).
        """
        resized = self.resize_frames(np_input)                   # (T, C, H, W) torch
        mask = torch.from_numpy(ch_mask).view(1, -1, 1, 1)
        blocks = []
        for process in self.config_data.PREPROCESS.DATA_TYPE:
            if process == 'Standardized':
                block = self.zstand(resized.clone(), exclude_mask=True)
            elif process == 'DiffNormalized':
                block = self.diffnorm(resized.clone(), exclude_mask=True)
            elif process == '':
                continue
            else:
                raise ValueError(f"Unsupported preprocessing type {process}")
            blocks.append(torch.nan_to_num(block) * mask)
        frames = torch.cat(blocks, dim=1)                        # (T, C_total, H, W)
        return frames.permute(1, 0, 2, 3).contiguous().numpy()   # (C_total, T, H, W)

    def resize_frames(self, frames) -> torch.Tensor:
        """
        Resize video frames using torchvision, with automatic NumPy → Torch conversion and GPU support.

        Parameters:
        frames (np.ndarray | torch.Tensor): Array of shape (N, H, W, C) or (N, C, H, W)
        Returns:
        torch.Tensor: Resized frames of shape (N, C, new_height, new_width) on the specified device
        """
        # Convert numpy input to torch
        frames = torch.from_numpy(frames).permute(0,3,1,2)
        N, C, H, W = frames.shape
        new_height = self.config_data.PREPROCESS.RESIZE.H
        new_width = self.config_data.PREPROCESS.RESIZE.W
        resized_frames = torch.nn.functional.interpolate(frames, size=(new_height, new_width), mode="bilinear", align_corners=False)
        return resized_frames

    def normalise_trace(self, raw_trace:np.ndarray, trace_type:str) -> np.ndarray:
        if trace_type == 'CVP':
            trace_min, trace_max = self.config_data.PREPROCESS.NECKFLIX.CVP_NORM
        elif trace_type == 'ABP':
            trace_min, trace_max = self.config_data.PREPROCESS.NECKFLIX.ABP_NORM
        elif trace_type == 'ECG':
            trace_min, trace_max = self.config_data.PREPROCESS.NECKFLIX.ECG_NORM
        else:
            raise ValueError(f"Unsupported trace type {trace_type} for normalization")
        clipped_trace = np.clip(raw_trace, a_min=trace_min, a_max=trace_max)  # clip first
        normed_trace = (clipped_trace - trace_min) / (trace_max - trace_min) * 2 - 1
        return normed_trace

    def unnormalise_trace(self, normed_trace:np.ndarray, trace_type:str) -> np.ndarray:
        if trace_type == 'CVP':
            trace_min, trace_max = self.config_data.PREPROCESS.NECKFLIX.CVP_NORM
        elif trace_type == 'ABP':
            trace_min, trace_max = self.config_data.PREPROCESS.NECKFLIX.ABP_NORM
        elif trace_type == 'ECG':
            trace_min, trace_max = self.config_data.PREPROCESS.NECKFLIX.ECG_NORM
        else:
            raise ValueError(f"Unsupported trace type {trace_type} for unnormalization")
        raw_trace = (normed_trace + 1) / 2 * (trace_max - trace_min) + trace_min
        return raw_trace

    def process_item(self, input, label) -> tuple[np.ndarray, np.ndarray]:
        resized_input = self.resize_frames(input)
        processed_input = resized_input.clone()

        for process in self.config_data.PREPROCESS.DATA_TYPE:
            if process == 'Standardized':
                processed_input = self.zstand(processed_input, exclude_mask=True)
            elif process == 'DiffNormalized':
                processed_input = self.diffnorm(processed_input, exclude_mask=True)
            elif process == '':
                pass
            else:
                raise ValueError(f"Unsupported preprocessing type {process}")
        # convert back to numpy array
        resized_input = resized_input.to('cpu').numpy()
        processed_input = processed_input.to('cpu').numpy()

        processed_label = []
        for i, trace in enumerate(self.config_data.PREPROCESS.NECKFLIX.TRACES):
            normed_trace = self.normalise_trace(label[:, i], trace)
            processed_label.append(normed_trace)

        processed_label = np.stack(processed_label, axis=-1)
        return processed_input, processed_label, resized_input

    def load(self):
        self.config_data.PREPROCESS.NECKFLIX
        chunk_length = self.config_data.PREPROCESS.CHUNK_LENGTH
        inputs = []
        labels = []
        cached_file_list = sorted(self.get_cached_file_list())
        # iterate through files
        for file in cached_file_list:
            with h5py.File(file, 'r') as f:
                if self.dict_output:
                    ch0 = next(ch for ch in self.slot_channels if ch in f)
                    keys = f[ch0]
                    probe = keys['timestamps'] if 'timestamps' in keys else \
                        keys[next(k for k in keys if k != 'frames')]
                    ids = probe.shape[0]
                else:
                    first_trace = f[self.config_data.PREPROCESS.NECKFLIX.CHANNELS[0]][
                        self.config_data.PREPROCESS.NECKFLIX.TRACES[0]]
                    ids = len(first_trace[~np.isnan(first_trace[:])])
                if self.config_data.PREPROCESS.DO_CHUNK:
                    for i in range(ids//chunk_length):
                        inputs.append((file.as_posix(), i))
                        labels.append((file.as_posix(), i))
                else:
                    inputs.append((file.as_posix(), 0))
                    labels.append((file.as_posix(), 0))
        begin_idx = int(len(inputs)*self.config_data.BEGIN)
        end_idx = int(len(inputs)*self.config_data.END)
        inputs = inputs[begin_idx:end_idx]
        labels = labels[begin_idx:end_idx]
        self.inputs = inputs
        self.labels = labels

    @staticmethod
    def diffnorm(frames, exclude_mask: bool = True, precision=torch.float32, eps: float = 1e-7):
        """
        Input shape: (T, C, H, W) or (B, T, C, H, W), channel-first.
        Returns same shape and same type (torch or numpy). Values are float.
        """
        is_numpy = isinstance(frames, np.ndarray)
        x = torch.from_numpy(frames) if is_numpy else frames
        if x.ndim not in (4, 5):
            raise ValueError("frames must be 4D (T,C,H,W) or 5D (B,T,C,H,W)")
        x = x.to(dtype=precision)

        # Axes
        tdim = 1 if x.ndim == 5 else 0
        cdim = 2 if x.ndim == 5 else 1

        # Optional mask: treat zeros as missing
        if exclude_mask:
            x = torch.where(x == 0, torch.nan, x)

        # Temporal pairs
        x1 = x.narrow(tdim, 1, x.shape[tdim] - 1)      # ... T-1 ...
        x0 = x.narrow(tdim, 0, x.shape[tdim] - 1)
        diff = x1 - x0
        denom = x1 + x0 + eps
        dn = diff / denom

        # Per-channel std over all dims except channel
        reduce_dims = tuple(d for d in range(dn.ndim) if d != cdim)
        std = torch.std(torch.nan_to_num(dn, nan=0.0), dim=reduce_dims, keepdim=True)
        dn = dn / (std + 1e-12)

        # Pad one zero frame to restore original T
        pad_shape = list(dn.shape)
        pad_shape[tdim] = 1
        pad = torch.zeros(pad_shape, dtype=dn.dtype, device=dn.device)
        out = torch.cat([dn, pad], dim=tdim)

        # Replace NaNs/Infs
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        if is_numpy:
            return out.detach().cpu().numpy()
        return out
    
    @staticmethod
    def zstand(frames, exclude_mask: bool = True, precision=torch.float32):
        """
        Z-score standardization per channel for video tensors.
        Input shape: (T,C,H,W) or (B,T,C,H,W)
        Returns same shape and type (torch or numpy).
        """
        is_numpy = isinstance(frames, np.ndarray)
        x = torch.from_numpy(frames) if is_numpy else frames
        if x.ndim not in (4, 5):
            raise ValueError("frames must be 4D (T,C,H,W) or 5D (B,T,C,H,W)")
        x = x.to(dtype=precision)

        # Identify dimensions
        cdim = 2 if x.ndim == 5 else 1

        if exclude_mask:
            # Mask out zeros for all channels
            mask = x != 0
            masked = torch.nan_to_num(x * mask, nan=0.0)
            counts = mask.sum(dim=tuple(i for i in range(x.ndim) if i != cdim), keepdim=True).clamp_min(1)
            means = masked.sum(dim=tuple(i for i in range(x.ndim) if i != cdim), keepdim=True) / counts
            centered = torch.where(mask, x - means, torch.nan)
            stds = torch.std(torch.nan_to_num(centered, nan=0.0), dim=tuple(i for i in range(x.ndim) if i != cdim), keepdim=True)
        else:
            means = x.mean(dim=tuple(i for i in range(x.ndim) if i != cdim), keepdim=True)
            stds = x.std(dim=tuple(i for i in range(x.ndim) if i != cdim), keepdim=True)

        out = (x - means) / (stds + 1e-12)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return out.cpu().numpy() if is_numpy else out
