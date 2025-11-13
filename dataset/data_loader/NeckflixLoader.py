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

class NeckflixLoader(BaseLoader):
    """The data loader for the Neckflix dataset."""

    def __init__(self, name, data_path, config_data,device=None):
        """Initializes an Neckflix dataloader.
            Args:
        """
        self.inputs = list()
        self.labels = list()
        self.config_data = config_data
        self.cached_path = Path(self.config_data.CACHED_PATH)
        self.data_format = config_data.DATA_FORMAT
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
        raw_input, raw_label = self.load_recording(index=index)
        input, label = self.process_item(raw_input, raw_label)
        # input shape is currently (D, C, H, W)
        if self.data_format == 'NDCHW':
            input = input
        elif self.data_format == 'NCDHW':
            input = input.transpose(1,0,2,3)
        elif self.data_format == 'NDHWC':
            input = input.transpose(0,2,3,1)
        else:
            raise ValueError('Unsupported Data Format!')
        filename = Path(self.inputs[index][0]).name
        chunk_id = self.inputs[index][1]
        label = label.squeeze()
        return input, label, filename, chunk_id

    def get_cached_file_list(self):
        config_data = self.config_data
        selected_channels = config_data.PREPROCESS.NECKFLIX.CHANNELS
        selected_traces = config_data.PREPROCESS.NECKFLIX.TRACES
        selected_postures = config_data.PREPROCESS.NECKFLIX.POSTURES
        with open(config_data.FOLD.FOLD_PATH, 'r') as f:
            selected_participants = set(f.read().splitlines())
        cached_dir = Path(config_data.CACHED_PATH)
        selected_files = []
        for file in cached_dir.glob("*.hdf5"):
            participant = file.name.split('_')[0]
            if participant not in selected_participants:
                continue
            posture = file.name.split('_')[3]
            if posture not in selected_postures:
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
        # Single open, read-only
        with h5py.File(h5_filepath, 'r') as f:
            # Validate once
            if not all(ch in f for ch in selected_channels):
                raise ValueError("Not all selected channels are present in the HDF5 file.")
            for ch in selected_channels:
                if not all(tr in f[ch] for tr in selected_traces):
                    raise ValueError(f"Expected {selected_traces} in {h5_filepath}\
                    but missing in channel {ch}.")

            # Infer length from first trace of first channel
            n_total = f[selected_channels[0]][selected_traces[0]].shape[0]
            if not random_chunk:
                start_idx = chunk_start_idx*chunk_size
                end_idx = min((chunk_start_idx + 1) * chunk_size, n_total)
            else:
                start_idx = np.random.randint(0, max(1, n_total - 2*chunk_size))
                end_idx = start_idx + chunk_size
            # print(f"Loading chunk from {start_idx} to {end_idx} (chunk length {chunk_size})")
            n = end_idx - start_idx
            n_tr = len(selected_traces)
            n_ch = len(selected_channels)

            # Compute labels with streaming accumulation - read only the chunk we need
            labels = np.empty((n_tr, n), dtype=np.float32)
            inv_nch = 1.0 / n_ch
            for ti, tr in enumerate(selected_traces):
                acc = None
                for ch in selected_channels:
                    arr = f[ch][tr][start_idx:end_idx]  # read only the chunk
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32, copy=False)
                    acc = arr if acc is None else acc + arr
                labels[ti] = acc * inv_nch

            # Keep only rows without NaNs across all traces
            finite_mask = np.isfinite(labels).all(axis=0)
            np_label = labels[:, finite_mask].T  # shape (time, traces)

            # Get absolute indices for frames (relative to start of recording)
            absolute_indices = start_idx + np.flatnonzero(finite_mask)

            # Read only needed frames per channel
            frames_list = []
            for ch in selected_channels:
                dset = f[ch]['frames']
                frames = dset[absolute_indices, ...]
                frames_list.append(frames)

        # Stack channels last and cast once
        np_input = np.stack(frames_list, axis=-1).astype(np.float32, copy=False)
        return np_input, np_label

    def resize_frames(self, frames) -> torch.Tensor:
        """
        Resize video frames using torchvision, with automatic NumPy → Torch conversion and GPU support.

        Parameters:
        frames (np.ndarray | torch.Tensor): Array of shape (N, H, W, C) or (N, C, H, W)
        Returns:
        torch.Tensor: Resized frames of shape (N, C, new_height, new_width) on the specified device
        """
        # Convert numpy input to torch
        frames = torch.from_numpy(frames).permute(0,3,1,2).to(self.device)
        N, C, H, W = frames.shape
        new_height = self.config_data.PREPROCESS.RESIZE.H
        new_width = self.config_data.PREPROCESS.RESIZE.W
        resized_frames = torch.empty((N, C, new_height, new_width),
                                    dtype=frames.dtype, device=self.device)

        for i in range(N):
            resized_frames[i] = F.resize(frames[i], [new_height, new_width], antialias=True)
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
        processed_input = self.resize_frames(input)

        for process in self.config_data.PREPROCESS.DATA_TYPE:
            if process == 'Standardized':
                processed_input = self.zstand(processed_input, exclude_mask=True, device=self.device)
            elif process == 'DiffNormalized':
                processed_input = self.diffnorm(processed_input, exclude_mask=True, device=self.device)
            elif process == '':
                pass
            else:
                raise ValueError(f"Unsupported preprocessing type {process}")
        # convert back to numpy array
        processed_input = processed_input.to('cpu').numpy()

        processed_label = []
        for i, trace in enumerate(self.config_data.PREPROCESS.NECKFLIX.TRACES):
            normed_trace = self.normalise_trace(label[:, i], trace)
            processed_label.append(normed_trace)

        processed_label = np.stack(processed_label, axis=-1)
        return processed_input, processed_label

    def load(self):
        self.config_data.PREPROCESS.NECKFLIX
        chunk_length = self.config_data.PREPROCESS.CHUNK_LENGTH
        inputs = []
        labels = []
        cached_file_list = sorted(self.get_cached_file_list())
        # iterate through files
        for file in cached_file_list:
            with h5py.File(file, 'r') as f:
                first_trace = f[self.config_data.PREPROCESS.NECKFLIX.CHANNELS[0]][self.config_data.PREPROCESS.NECKFLIX.TRACES[0]]
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
    def diffnorm(frames, exclude_mask: bool = True, precision=torch.float32, device: str | torch.device = "cuda", eps: float = 1e-7):
        """
        Input shape: (T, C, H, W) or (B, T, C, H, W), channel-first.
        Returns same shape and same type (torch or numpy). Values are float.
        """
        is_numpy = isinstance(frames, np.ndarray)
        x = torch.from_numpy(frames) if is_numpy else frames
        if x.ndim not in (4, 5):
            raise ValueError("frames must be 4D (T,C,H,W) or 5D (B,T,C,H,W)")
        x = x.to(device=device, dtype=precision)

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
    def zstand(frames, exclude_mask: bool = True, precision=torch.float32, device: str | torch.device = "cuda"):
        """
        Z-score standardization per channel for video tensors.
        Input shape: (T,C,H,W) or (B,T,C,H,W)
        Returns same shape and type (torch or numpy).
        """
        is_numpy = isinstance(frames, np.ndarray)
        x = torch.from_numpy(frames) if is_numpy else frames
        if x.ndim not in (4, 5):
            raise ValueError("frames must be 4D (T,C,H,W) or 5D (B,T,C,H,W)")
        x = x.to(device=device, dtype=precision)

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
