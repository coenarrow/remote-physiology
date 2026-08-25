"""PhysHydra Trainer."""
import os
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import random
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysHydraLoss import PhysHydraLoss
from neural_methods.model.PhysHydra import PhysHydra
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import h5py
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler


NCOLS=120  # Number of columns for tqdm progress bars
DEBUG_STEP_SIZE = 10  # Frequency of GPU memory debug prints during training
TRAIN_STEP_SIZE = 10  # Frequency of training loss prints during training

class InterpretabilityHDF5Writer:
    def __init__(self, path, channel_names, compression='gzip', compression_opts=4, shuffle=True, fletcher32=False):
        """
        HDF5 writer for interpretability outputs with channel-specific storage.
        
        Args:
            path: Path to HDF5 file
            channel_names: List of channel names (e.g., ['R','G','B','I','D'])
            compression: e.g. "gzip", "lzf", or None
            compression_opts: gzip level (1–9) if compression == "gzip"
            shuffle: enable HDF5 shuffle filter (good with gzip)
            fletcher32: enable Fletcher32 checksum
        """
        self.path = path
        self.channel_names = channel_names
        
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.fletcher32 = fletcher32

        self.file = None
        self.subj_ds = None
        self.sort_ds = None
        self.channel_datasets = {}  # Dict mapping channel name to dataset
        self.attn_ds = None
        self.scores_ds = None
        self.n_samples = 0

    def _get_channel_dtype(self, channel_name):
        """Determine appropriate dtype for each channel type."""
        if channel_name in ['R', 'G', 'B']:
            return np.uint8
        elif channel_name in ['I', 'D']:
            return np.uint16
        else:
            # Unknown channel type, use float16
            return np.float16

    def _convert_channel_to_dtype(self, arr, channel_name):
        """Convert array to appropriate dtype for the channel."""
        target_dtype = self._get_channel_dtype(channel_name)
        
        if target_dtype == np.uint8:
            # RGB channels: expect 0-255 range
            if arr.max() <= 1.0:
                # Normalized to [0, 1], scale up
                arr = arr * 255.0
            return np.clip(arr, 0, 255).round().astype(np.uint8, copy=False)
        
        elif target_dtype == np.uint16:
            # I/D channels: expect 0-65535 range
            if arr.max() <= 1.0:
                # Normalized to [0, 1], scale up
                arr = arr * 65535.0
            elif arr.max() <= 255:
                # Looks like uint8 range, scale up
                arr = arr * 257.0  # 65535/255 ≈ 257
            return np.clip(arr, 0, 65535).round().astype(np.uint16, copy=False)
        
        else:
            # Float16 for unknown channels
            return arr.astype(np.float16, copy=False)

    def _open(self):
        if self.file is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.file = h5py.File(self.path, "w")

    def _create_dataset(self, name, shape, maxshape, dtype, chunks):
        """Helper to create a dataset with the configured compression options."""
        return self.file.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=chunks,
            compression=self.compression,
            compression_opts=self.compression_opts if self.compression == "gzip" else None,
            shuffle=self.shuffle,
            fletcher32=self.fletcher32,
        )

    def _init_dsets(self, channel_arrs, attn_arr, scores_arr):
        """
        channel_arrs: dict mapping channel_name -> np.ndarray for one sample
        attn_arr: np.ndarray for one sample (can be None)
        scores_arr: np.ndarray for one sample (can be None)
        """
        self._open()

        # Variable-length subject strings
        str_dt = h5py.string_dtype(encoding="utf-8")
        self.subj_ds = self._create_dataset(
            "subj_index",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dt,
            chunks=(1024,),
        )
        self.sort_ds = self._create_dataset(
            "sort_index",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(1024,),
        )

        # Create separate dataset for each channel
        for ch_name, ch_arr in channel_arrs.items():
            ch_dtype = self._get_channel_dtype(ch_name)
            self.channel_datasets[ch_name] = self._create_dataset(
                f"{ch_name}",
                shape=(0,) + ch_arr.shape,
                maxshape=(None,) + ch_arr.shape,
                chunks=(1,) + ch_arr.shape,
                dtype=ch_dtype,
            )

        # Attention maps (optional)
        if attn_arr is not None:
            self.attn_ds = self._create_dataset(
                "attention_maps",
                shape=(0,) + attn_arr.shape,
                maxshape=(None,) + attn_arr.shape,
                chunks=(1,) + attn_arr.shape,
                dtype=attn_arr.dtype,
            )

        # Channel scores (optional) - use float32
        if scores_arr is not None:
            self.scores_ds = self._create_dataset(
                "channel_scores",
                shape=(0,) + scores_arr.shape,
                maxshape=(None,) + scores_arr.shape,
                chunks=(1,) + scores_arr.shape,
                dtype=scores_arr.dtype,
            )

    def append(self, subj_index, sort_index, input_tensor, spatial_map, channel_score):
        """
        subj_index: anything convertible to string
        sort_index: int
        input_tensor: torch tensor with shape (C, H, W, ...) where C matches len(channel_names)
        spatial_map, channel_score: torch tensors on any device (can be None)
        """
        # Convert to CPU numpy (convert to float32 first to handle bfloat16/float16 from AMP)
        inp = input_tensor.detach().float().cpu().numpy()
        
        # Split input by channels
        num_channels = inp.shape[1]
        if num_channels != len(self.channel_names):
            raise ValueError(
                f"Input has {num_channels} channels but expected {len(self.channel_names)} "
                f"based on channel_names: {self.channel_names}"
            )
        
        # Create dict of channel arrays
        channel_arrs = {}
        for i, ch_name in enumerate(self.channel_names):
            ch_arr = inp[:, i]  # Extract single channel (H, W, ...)
            ch_arr = self._convert_channel_to_dtype(ch_arr, ch_name)
            channel_arrs[ch_name] = ch_arr
        
        # Convert attention maps and scores
        attn = None
        scores = None
        if spatial_map is not None:
            attn = spatial_map.detach().float().cpu().numpy()
        if channel_score is not None:
            scores = channel_score.detach().float().cpu().numpy()

        # Initialize datasets on first call
        if len(self.channel_datasets) == 0:
            self._init_dsets(channel_arrs, attn, scores)

        i = self.n_samples
        self.n_samples += 1

        # Resize metadata datasets
        self.subj_ds.resize((self.n_samples,))
        self.sort_ds.resize((self.n_samples,))
        
        # Write metadata
        self.subj_ds[i] = str(subj_index)
        self.sort_ds[i] = int(sort_index)
        
        # Write each channel separately
        for ch_name, ch_arr in channel_arrs.items():
            self.channel_datasets[ch_name].resize((self.n_samples,) + self.channel_datasets[ch_name].shape[1:])
            self.channel_datasets[ch_name][i] = ch_arr

        # Write attention maps and scores
        if self.attn_ds is not None and attn is not None:
            self.attn_ds.resize((self.n_samples,) + self.attn_ds.shape[1:])
            self.attn_ds[i] = attn

        if self.scores_ds is not None and scores is not None:
            self.scores_ds.resize((self.n_samples,) + self.scores_ds.shape[1:])
            self.scores_ds[i] = scores

    def close(self):
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None

class PhysHydraTrainer(BaseTrainer):
    def __init__(self, config, data_loader, rank=0, world_size=1,debug:bool=False,debug_gpu:bool=False):
        """Inits parameters from args and the writer for TensorboardX.
        rank: process rank for DDP (0 if single GPU)
        world_size: total number of processes (1 if single GPU)
        """
        super().__init__()

        # check if there's already a pickle file in the output directory
        pickle_files = Path(config.TEST.OUTPUT_SAVE_DIR).glob("*.pickle")
        if any(pickle_files):
            raise ValueError(f"Output directory {config.TEST.OUTPUT_SAVE_DIR} already contains pickle files.")

        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)  # Only main process saves/logs
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank}')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.debug = debug
        self.debug_gpu = debug_gpu
        
        if self.debug and self.is_main:
            tqdm.write(f"DEBUG MODE: Enabled for rank {rank}, device: {self.device}")

        if self.debug_gpu and torch.cuda.is_available():
            total_mem_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            tqdm.write(f"DEBUG: GPU total memory on {self.device}: {total_mem_gb:.2f} GiB")
        
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.num_channels = config.MODEL.PHYSHYDRA.NUM_CHANNELS
        self.num_labels = config.MODEL.PHYSHYDRA.NUM_LABELS
        self.frame_rate = config.TRAIN.DATA.FS
        self.num_frames = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH

        # Loss settings
        self.w_ccc = config.MODEL.PHYSHYDRA.W_CCC
        self.w_mean = config.MODEL.PHYSHYDRA.W_MEAN
        self.w_max = config.MODEL.PHYSHYDRA.W_MAX
        self.w_min = config.MODEL.PHYSHYDRA.W_MIN
        self.w_spec = config.MODEL.PHYSHYDRA.W_SPEC
        
        # Interpretability settings
        self.interpretable = config.MODEL.PHYSHYDRA.INTERPRETABLE
        self.preserve_channels = config.MODEL.PHYSHYDRA.PRESERVE_CHANNELS
        self.lambda_sparsity = config.MODEL.PHYSHYDRA.LAMBDA_SPARSITY
        self.lambda_smoothness = config.MODEL.PHYSHYDRA.LAMBDA_SMOOTHNESS
        self.save_attention_maps = config.MODEL.PHYSHYDRA.SAVE_ATTENTION_MAPS

        # Compression type for interpretability HDF5 writer
        self.compression_type = 'gzip'

        # Mixed precision training setup
        self.use_amp = config.TRAIN.get('USE_AMP', False)  # Default to False for backward compatibility
        self.amp_dtype = getattr(config.TRAIN, 'AMP_DTYPE', 'float32')  # 'float16' or 'bfloat16', defaults to 'float32'

        if self.use_amp and self.device.type != 'cuda':
            if self.is_main:
                tqdm.write(f"WARNING: AMP requested but device is {self.device.type}; disabling AMP")
            self.use_amp = False

        if self.use_amp:
            if self.amp_dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
                if self.is_main:
                    tqdm.write("WARNING: bfloat16 not supported on this GPU, falling back to float16")
                self.amp_dtype = 'float16'

            self.scaler = GradScaler() if self.amp_dtype == 'float16' else None
            if self.is_main:
                tqdm.write(f"Mixed precision training enabled with {self.amp_dtype}")
                if self.amp_dtype == 'bfloat16':
                    tqdm.write("Using bfloat16 (no gradient scaling needed)")
        else:
            self.scaler = None
            if self.is_main:
                tqdm.write("Using full precision (float32) training")

        # Tracking metrics
        self.loss_dict = OrderedDict()
        self.loss_dict['train_loss'] = []
        self.loss_dict['valid_loss'] = []
        self.loss_dict['learning_rate'] = []
        self.loss_dict['mean_loss'] = []
        self.loss_dict['max_loss'] = []
        self.loss_dict['min_loss'] = []
        self.loss_dict['ccc_loss'] = []
        self.loss_dict['spectral_loss'] = []
        self.loss_dict['sparsity_loss'] = []
        self.loss_dict['smoothness_loss'] = []

        self.model = PhysHydra(
            in_channels=self.num_channels, 
            out_signals=self.num_labels,
            frames=self.num_frames,
            interpretable=self.interpretable,
            preserve_channels=self.preserve_channels,
            debug=self.debug
        ).to(self.device)
        
        # Use DDP instead of DataParallel
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
            if self.debug and self.is_main:
                tqdm.write(f"Model wrapped with DDP on rank {rank}")

        # Define loss function
        self.loss_class = PhysHydraLoss(w_ccc=self.w_ccc, 
            w_mean=self.w_mean, 
            w_max=self.w_max, 
            w_min=self.w_min,
            w_spec=self.w_spec,
            ).to(self.device)


        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr=config.TRAIN.LR, 
                                        weight_decay=0.0005)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,                 # ↑ from 3e-3
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches,
                pct_start=0.15,              # ↓ warmup
                anneal_strategy='linear',    # ↓ slower late decay
                div_factor=10,               # controls initial LR = max_lr / div_factor
                final_div_factor=40          # increases how low LR goes at the end
            )
            
            # Update train dataloader with DistributedSampler
            if world_size > 1 and data_loader["train"] is not None:
                self.train_sampler = data_loader["train"].sampler
            else:
                self.train_sampler = None

    def _debug_gpu_memory(self, tag: str = ""):
        """Print current and peak GPU memory usage."""
        if not torch.cuda.is_available():
            return

        # Make sure all pending CUDA ops are accounted for
        torch.cuda.synchronize(self.device)

        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MiB
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)   # MiB
        peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)

        tqdm.write(
            f"DEBUG[{tag}] GPU mem on {self.device}: "
            f"allocated={allocated:.1f} MiB, reserved={reserved:.1f} MiB, "
            f"peak_alloc={peak_allocated:.1f} MiB, total={total:.1f} MiB"
        )

        # Optional: warn if very close to OOM
        usage_ratio = reserved * (1024 ** 2) / torch.cuda.get_device_properties(self.device).total_memory
        if usage_ratio > 0.9:
            tqdm.write(f"WARNING[{tag}]: Reserved memory > 90% of total; OOM likely soon.")

    def unnormalise_trace(self, normed_trace: torch.Tensor, trace_type: str) -> torch.Tensor:
        if trace_type == 'CVP':
            trace_min, trace_max = self.config.TRAIN.DATA.PREPROCESS.NECKFLIX.CVP_NORM
        elif trace_type == 'ABP':
            trace_min, trace_max = self.config.TRAIN.DATA.PREPROCESS.NECKFLIX.ABP_NORM
        elif trace_type == 'ECG':
            trace_min, trace_max = self.config.TRAIN.DATA.PREPROCESS.NECKFLIX.ECG_NORM
        else:
            raise ValueError(f"Unsupported trace type {trace_type} for unnormalization")

        # convert scalars to tensors on the same device as input
        device = normed_trace.device
        trace_min = torch.as_tensor(trace_min, dtype=normed_trace.dtype, device=device)
        trace_max = torch.as_tensor(trace_max, dtype=normed_trace.dtype, device=device)

        # inverse of (x - min) / (max - min) mapped to [-1, +1]
        raw_trace = (normed_trace + 1) / 2 * (trace_max - trace_min) + trace_min
        return raw_trace

    def train(self, data_loader):
        """Training routine for model with interpretability regularisation"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        if self.world_size > 1:
            dist.barrier()  # Synchronise all processes

        for epoch in range(self.max_epoch_num):
            if self.is_main:
                tqdm.write('')
                tqdm.write(f"====Training Epoch: {epoch}====")

            # Set epoch for distributed sampler
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            if self.debug_gpu and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
                self._debug_gpu_memory(tag=f"epoch {epoch} start")
            
            self.model.train()
            epoch_loss = 0.0
            epoch_mean_loss = 0.0
            epoch_max_loss = 0.0
            epoch_min_loss = 0.0
            epoch_ccc_loss = 0.0
            epoch_spec_loss = 0.0
            epoch_sparsity_loss = 0.0
            epoch_smoothness_loss = 0.0
            epoch_lr_sum = 0.0
            num_batches = 0
            
            # Only show progress bar on main process
            if self.is_main:
                tbar = tqdm(data_loader["train"], ncols=NCOLS)
            else:
                tbar = data_loader["train"]
            
            for idx, batch in enumerate(tbar):
                if self.is_main:
                    tbar.set_description("Train epoch %s" % epoch)
                
                try:
                    data, labels = batch[0].float(), batch[1].float()
                    N, D, C, H, W = data.shape
                    # Don't force dtype here - let autocast handle it
                    data = data.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    if self.debug:
                        # Debug checks
                        assert not torch.isnan(data).any(), f"NaN in input data at batch {idx}"
                        assert not torch.isinf(data).any(), f"Inf in input data at batch {idx}"
                        if self.is_main and idx == 0:
                            tqdm.write(f"DEBUG: Batch shape: {data.shape}, Device: {data.device}")

                    if self.debug_gpu and torch.cuda.is_available():
                        if self.is_main and idx == 0:
                            tqdm.write(f"DEBUG: Batch shape: {data.shape}, Device: {data.device}")
                            self._debug_gpu_memory(tag=f"epoch {epoch} batch {idx} (after move)")

                    self.optimizer.zero_grad()

                    # Use autocast for mixed precision
                    with autocast(enabled=self.use_amp, dtype=torch.float16 if self.amp_dtype == 'float16' else torch.bfloat16):
                        pred, (sparsity_loss, smoothness_loss) = self.model(data)

                        if pred.dim() == 2:
                            pred = pred.unsqueeze(1)
                        if labels.dim() == 2:
                            labels = labels.unsqueeze(1)

                        # Handle dimension ordering
                        if labels.shape[-1] < labels.shape[1]:
                            labels = labels.transpose(1, 2)
                        if pred.shape[-1] < pred.shape[1]:
                            pred = pred.transpose(1, 2)

                        signal_loss = self.loss_class(pred, labels)
                        epoch_mean_loss += self.loss_class.losses.get('mean_loss', 0.0)
                        epoch_max_loss += self.loss_class.losses.get('max_loss', 0.0)
                        epoch_min_loss += self.loss_class.losses.get('min_loss', 0.0)
                        epoch_ccc_loss += self.loss_class.losses.get('ccc_loss', 0.0)
                        epoch_spec_loss += self.loss_class.losses.get('spectral_loss', 0.0)

                        if self.debug:
                            assert not torch.isnan(signal_loss), f"NaN loss at batch {idx}"

                        # Interpretability regularisation
                        total_loss = signal_loss
                        if self.interpretable:
                            interp_loss = self.lambda_sparsity * sparsity_loss + self.lambda_smoothness * smoothness_loss
                            total_loss = total_loss + interp_loss
                            epoch_sparsity_loss += sparsity_loss.item()
                            epoch_smoothness_loss += smoothness_loss.item()

                    # Backward pass with gradient scaling for fp16
                    if self.use_amp and self.scaler is not None:
                        # Using float16 - need gradient scaling
                        self.scaler.scale(total_loss).backward()
                    else:
                        # Using float32 or bfloat16 - no scaling needed
                        total_loss.backward()

                    if idx % DEBUG_STEP_SIZE == 0:
                        # Check gradients
                        if self.debug:
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    grad_norm = param.grad.norm().item()
                                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                        tqdm.write(f"WARNING: Invalid gradient in {name}: norm={grad_norm}")
                        if self.debug_gpu and torch.cuda.is_available():
                            self._debug_gpu_memory(tag=f"epoch {epoch} batch {idx} (after backward)")

                    epoch_loss += signal_loss.item()
                    num_batches += 1

                    if idx % TRAIN_STEP_SIZE == TRAIN_STEP_SIZE - 1:
                        # Synchronize losses across GPUs for accurate reporting.
                        # IMPORTANT: all ranks must participate in the collective call
                        # to avoid deadlocks. Only `rank 0` (self.is_main) prints.
                        if self.world_size > 1:
                            # Build identical tensor on all ranks
                            print_losses = torch.tensor([
                                epoch_loss / num_batches,
                                epoch_mean_loss / num_batches,
                                epoch_max_loss / num_batches,
                                epoch_min_loss / num_batches,
                                epoch_ccc_loss / num_batches,
                                epoch_spec_loss / num_batches,
                                epoch_sparsity_loss / num_batches if self.interpretable else 0.0,
                                epoch_smoothness_loss / num_batches if self.interpretable else 0.0,
                            ], device=self.device)

                            # Use SUM across ranks (portable), then divide by world_size
                            dist.all_reduce(print_losses, op=dist.ReduceOp.SUM)
                            print_losses = print_losses / float(self.world_size)

                            # Unpack synchronized values (now averaged)
                            sync_loss, sync_mean, sync_max, sync_min, sync_ccc, sync_spec, sync_sparsity, sync_smoothness = print_losses.cpu().tolist()
                        else:
                            # Single GPU: use local values
                            sync_loss = epoch_loss / num_batches
                            sync_mean = epoch_mean_loss / num_batches
                            sync_max = epoch_max_loss / num_batches
                            sync_min = epoch_min_loss / num_batches
                            sync_ccc = epoch_ccc_loss / num_batches
                            sync_spec = epoch_spec_loss / num_batches
                            sync_sparsity = epoch_sparsity_loss / num_batches if self.interpretable else 0.0
                            sync_smoothness = epoch_smoothness_loss / num_batches if self.interpretable else 0.0

                        # Only main process prints human readable logs
                        if self.is_main:
                            print(f'[{epoch}, {idx + 1:5d}]')
                            print(f'Total Signal Loss: {sync_loss:.3f}')
                            print(f'{"Mean Loss:":<16} {sync_mean:.4f} weighted: {sync_mean * self.w_mean:.4f}')
                            print(f'{"Max Loss:":<16} {sync_max:.4f} weighted: {sync_max * self.w_max:.4f}')
                            print(f'{"Min Loss:":<16} {sync_min:.4f} weighted: {sync_min * self.w_min:.4f}')
                            print(f'{"CCC Loss:":<16} {sync_ccc:.4f} weighted: {sync_ccc * self.w_ccc:.4f}')
                            print(f'{"Spectral Loss:":<16} {sync_spec:.4f} weighted: {sync_spec * self.w_spec:.4f}')
                            if self.interpretable:
                                print(f'{"Sparsity Loss:":<16} {sync_sparsity:.4f} weighted: {sync_sparsity * self.lambda_sparsity:.4f}')
                                print(f'{"Smoothness Loss:":<16} {sync_smoothness:.4f} weighted: {sync_smoothness * self.lambda_smoothness:.4f}')
                                # print the min, max, mean and std of an attention map example
                                model_to_use = self.model.module if isinstance(self.model, DDP) else self.model
                                spatial_maps, channel_importance = model_to_use.get_interpretability_maps()
                                print("spatial_maps shape:", spatial_maps.shape)
                                print("spatial_maps dtype:", spatial_maps.dtype)
                                min_str = "Min: "
                                max_str = "Max: "
                                mean_str = "Mean: "
                                std_str = "Std: "
                                for i, channel in enumerate(self.config.TRAIN.DATA.PREPROCESS.NECKFLIX.CHANNELS):
                                    attn_map_example = spatial_maps[0, i, 0]  # First sample, first channel
                                    min_str += f"{channel} {attn_map_example.min():.4f}  "
                                    max_str += f"{channel} {attn_map_example.max():.4f}  "
                                    mean_str += f"{channel} {attn_map_example.mean():.4f}  "
                                    std_str += f"{channel} {attn_map_example.std():.4f}  "
                                print(f'Attention Map Examples')
                                print(min_str)
                                print(max_str)
                                print(mean_str)
                                print(std_str)

                    # Optimizer step with gradient scaling for fp16
                    if self.use_amp and self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    # Step scheduler after optimizer (correct order for OneCycleLR)
                    # Note: PyTorch may show a warning on first iteration, but this is a false positive
                    current_lr = self.scheduler.get_last_lr()[0]
                    epoch_lr_sum += current_lr
                    self.scheduler.step()

                    if idx % DEBUG_STEP_SIZE == 0:
                        if self.debug_gpu and torch.cuda.is_available():
                            self._debug_gpu_memory(tag=f"epoch {epoch} batch {idx} (after scheduler step)")
                    
                    if self.is_main:
                        postfix = {'loss': signal_loss.item(), 'lr': current_lr}
                        if self.interpretable:
                            postfix['interp'] = interp_loss.item() if 'interp_loss' in locals() else 0
                        tbar.set_postfix(postfix)
                        
                except Exception as e:
                    if self.debug:
                        tqdm.write(f"ERROR at epoch {epoch}, batch {idx}: {str(e)}")
                        tqdm.write(f"Data shape: {data.shape if 'data' in locals() else 'N/A'}")
                        tqdm.write(f"Labels shape: {labels.shape if 'labels' in locals() else 'N/A'}")
                        if 'pred' in locals():
                            tqdm.write(f"Pred shape: {pred.shape}")
                    raise e
            
            # Synchronize all ranks before aggregating losses
            if self.world_size > 1:
                dist.barrier()
                torch.cuda.synchronize()
            
            # Compute epoch averages for all ranks
            rank_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            rank_mean_loss = epoch_mean_loss / num_batches if num_batches > 0 else 0.0
            rank_max_loss = epoch_max_loss / num_batches if num_batches > 0 else 0.0
            rank_min_loss = epoch_min_loss / num_batches if num_batches > 0 else 0.0
            rank_ccc_loss = epoch_ccc_loss / num_batches if num_batches > 0 else 0.0
            rank_spec_loss = epoch_spec_loss / num_batches if num_batches > 0 else 0.0
            rank_sparsity_loss = epoch_sparsity_loss / num_batches if self.interpretable and num_batches > 0 else 0.0
            rank_smoothness_loss = epoch_smoothness_loss / num_batches if self.interpretable and num_batches > 0 else 0.0
            rank_lr = epoch_lr_sum / num_batches if num_batches > 0 else 0.0
            
            # Aggregate losses across all GPUs if using DDP
            if self.world_size > 1:
                # Convert to tensors for reduction
                loss_tensor = torch.tensor([
                    rank_train_loss, rank_mean_loss, rank_max_loss, rank_min_loss,
                    rank_ccc_loss, rank_spec_loss, rank_sparsity_loss, rank_smoothness_loss, rank_lr
                ], device=self.device)
                
                # Average across all ranks (implicit barrier)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                
                # Unpack averaged values
                (avg_train_loss, avg_mean_loss, avg_max_loss, avg_min_loss,
                 avg_ccc_loss, avg_spec_loss, avg_sparsity_loss, avg_smoothness_loss, avg_lr) = loss_tensor.cpu().tolist()
            else:
                # Single GPU: just use rank values
                avg_train_loss = rank_train_loss
                avg_mean_loss = rank_mean_loss
                avg_max_loss = rank_max_loss
                avg_min_loss = rank_min_loss
                avg_ccc_loss = rank_ccc_loss
                avg_spec_loss = rank_spec_loss
                avg_sparsity_loss = rank_sparsity_loss
                avg_smoothness_loss = rank_smoothness_loss
                avg_lr = rank_lr
            
            # Store epoch-level metrics (only on main process)
            if self.is_main:
                self.loss_dict['train_loss'].append(avg_train_loss)
                self.loss_dict['learning_rate'].append(avg_lr)
                self.loss_dict['mean_loss'].append(avg_mean_loss)
                self.loss_dict['max_loss'].append(avg_max_loss)
                self.loss_dict['min_loss'].append(avg_min_loss)
                self.loss_dict['ccc_loss'].append(avg_ccc_loss)
                self.loss_dict['spectral_loss'].append(avg_spec_loss)
                self.loss_dict['sparsity_loss'].append(avg_sparsity_loss)
                self.loss_dict['smoothness_loss'].append(avg_smoothness_loss)

                self.save_model(epoch)
                
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                if self.is_main:
                    self.loss_dict['valid_loss'].append(valid_loss)
                    tqdm.write('validation loss: ', valid_loss)
                    if self.min_valid_loss is None:
                        self.min_valid_loss = valid_loss
                        self.best_epoch = epoch
                        tqdm.write("Update best model! Best epoch: {}".format(self.best_epoch))
                    elif valid_loss < self.min_valid_loss:
                        self.min_valid_loss = valid_loss
                        self.best_epoch = epoch
                        tqdm.write("Update best model! Best epoch: {}".format(self.best_epoch))
            torch.cuda.empty_cache()
        
        # Plot losses and save loss dictionary (only on main process)
        if self.is_main:
            self.plot_losses_and_lrs(self.loss_dict, self.config)

    def valid(self, data_loader):
        """Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        if self.is_main:
            tqdm.write('')
            tqdm.write(" ====Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            if self.is_main:
                vbar = tqdm(data_loader["valid"], ncols=NCOLS)
            else:
                vbar = data_loader["valid"]
            for valid_idx, valid_batch in enumerate(vbar):
                if self.is_main:
                    vbar.set_description("Validation")
                data = valid_batch[0].to(device=self.device, non_blocking=True)
                label = valid_batch[1].to(device=self.device, non_blocking=True)

                # Use autocast for validation too
                with autocast(enabled=self.use_amp, dtype=torch.float16 if self.amp_dtype == 'float16' else torch.bfloat16):
                    pred, _ = self.model(data)

                    if pred.dim() == 2:
                        pred = pred.unsqueeze(1)
                    if label.dim() == 2:
                        label = label.unsqueeze(1)

                    signal_loss = self.loss_class(pred, label)

                valid_loss.append(signal_loss.item())
                valid_step += 1
                if self.is_main:
                    vbar.set_postfix(loss=signal_loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """Runs the model on test sets with optional attention map saving."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        # Only main process should test
        if not self.is_main:
            return
        
        tqdm.write('')
        tqdm.write("===Testing===")
        
        predictions = dict()
        labels = dict()
        interp_writer = None
        
        if self.save_attention_maps and self.interpretable and self.config.TEST.OUTPUT_SAVE_DIR:
            h5_path = os.path.join(self.config.TEST.OUTPUT_SAVE_DIR, f"interpretability_outputs.h5")
            # Get channel names from config
            channel_names = self.config.TEST.DATA.PREPROCESS.NECKFLIX.CHANNELS
            interp_writer = InterpretabilityHDF5Writer(h5_path, channel_names=channel_names, compression=self.compression_type)

        # Load model
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            model_to_load.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            tqdm.write("Testing uses pretrained model!")
            tqdm.write(self.config.INFERENCE.MODEL_PATH)
        else:
            epoch = self.max_epoch_num - 1 if self.config.TEST.USE_LAST_EPOCH else self.best_epoch
            model_path = os.path.join(self.model_dir, f'{self.model_file_name}_Epoch{epoch}.pth')
            model_to_load.load_state_dict(torch.load(model_path))
            msg = "last epoch" if self.config.TEST.USE_LAST_EPOCH else "best epoch selected using model selection"
            tqdm.write(f"Testing uses {msg} as non-pretrained model!")
            tqdm.write(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        tqdm.write("Running model evaluation on the testing dataset!")

        for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=NCOLS)):
            batch_size = test_batch[0].shape[0]
            data = test_batch[0].to(self.device, non_blocking=True)
            label = test_batch[1].to(self.device, non_blocking=True)

            with torch.no_grad():
                # Use autocast for inference too
                with autocast(enabled=self.use_amp, dtype=torch.float16 if self.amp_dtype == 'float16' else torch.bfloat16):
                    pred, _ = self.model(data)
                    spatial_maps, channel_importance = None, None
                    if self.interpretable and self.save_attention_maps:
                        model_to_use = self.model.module if isinstance(self.model, DDP) else self.model
                        spatial_maps, channel_importance = model_to_use.get_interpretability_maps()

                    if pred.dim() == 2 and self.num_labels == 1:
                        pred = pred.unsqueeze(1)
                    if label.dim() == 2 and self.num_labels == 1:
                        label = label.unsqueeze(1)

                    unnormed_labels = torch.empty_like(label)
                    unnormed_preds  = torch.empty_like(pred)

                    for i, trace in enumerate(self.config.TEST.DATA.PREPROCESS.NECKFLIX.TRACES):
                        unnormed_labels[:, i, :] = self.unnormalise_trace(label[:, i, :], trace)
                        unnormed_preds[:, i, :]  = self.unnormalise_trace(pred[:, i, :], trace)

                    pred_cpu = unnormed_preds.cpu() if self.config.TEST.OUTPUT_SAVE_DIR else pred
                    label_cpu = unnormed_labels.cpu() if self.config.TEST.OUTPUT_SAVE_DIR else label


            # Store predictions + labels
            for idx in range(batch_size):
                subj_index = test_batch[2][idx]
                sort_index = int(test_batch[3][idx])

                if subj_index not in predictions:
                    predictions[subj_index] = dict()
                    labels[subj_index] = dict()

                predictions[subj_index][sort_index] = pred_cpu[idx]
                labels[subj_index][sort_index] = label_cpu[idx]

                # Write interpretability data
                if self.save_attention_maps and self.interpretable and interp_writer is not None:
                    resized_input = test_batch[4][idx]
                    if isinstance(resized_input, torch.Tensor):
                        resized_input = resized_input.cpu()

                    spatial_map_i = spatial_maps[idx].cpu() if spatial_maps is not None and isinstance(spatial_maps[idx], torch.Tensor) else (spatial_maps[idx] if spatial_maps is not None else None)
                    channel_i = channel_importance[idx].cpu() if channel_importance is not None and isinstance(channel_importance[idx], torch.Tensor) else (channel_importance[idx] if channel_importance is not None else None)

                    interp_writer.append(
                        subj_index=subj_index,
                        sort_index=sort_index,
                        input_tensor=resized_input,
                        spatial_map=spatial_map_i,
                        channel_score=channel_i,
                    )

            del data, label, pred
            if spatial_maps is not None:
                del spatial_maps
            if channel_importance is not None:
                del channel_importance
        
        if interp_writer is not None:
            interp_writer.close()
        
        tqdm.write('')
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)
        calculate_metrics(predictions, labels, self.config)
    
    def plot_losses_and_lrs(self, loss_dict, config):
        if not self.is_main:
            return

        output_dir = os.path.join(config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, 'plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Filename ID to be used in plots that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        
        # Determine number of epochs from train_loss
        num_epochs = len(loss_dict['train_loss'])
        epochs = range(0, num_epochs)
        
        # ===== PLOT 1: Learning Rate =====
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss_dict['learning_rate'], label='Learning Rate', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'{filename_id} Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(epochs)
        
        # Set y-axis values in scientific notation
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
        lr_plot_filename = os.path.join(output_dir, filename_id + '_learning_rate.pdf')
        plt.savefig(lr_plot_filename, bbox_inches='tight', dpi=300)
        plt.close()
        tqdm.write(f'Saved learning rate plot: {lr_plot_filename}')
        
        # ===== PLOT 2: Unweighted Losses =====
        plt.figure(figsize=(12, 7))
        
        # Plot main losses
        plt.plot(epochs, loss_dict['train_loss'], label='Train Loss', linewidth=2)
        if len(loss_dict['valid_loss']) > 0:
            plt.plot(epochs, loss_dict['valid_loss'], label='Valid Loss', linewidth=2)
        
        # Plot component losses
        plt.plot(epochs, loss_dict['mean_loss'], label='Mean Loss', alpha=0.7, linestyle='--')
        plt.plot(epochs, loss_dict['max_loss'], label='Max Loss', alpha=0.7, linestyle='--')
        plt.plot(epochs, loss_dict['min_loss'], label='Min Loss', alpha=0.7, linestyle='--')
        plt.plot(epochs, loss_dict['ccc_loss'], label='CCC Loss', alpha=0.7, linestyle='--')
        plt.plot(epochs, loss_dict['spectral_loss'], label='Spectral Loss', alpha=0.7, linestyle='--')
        
        # Plot regularization losses if interpretable mode
        if self.interpretable:
            plt.plot(epochs, loss_dict['sparsity_loss'], label='Sparsity Loss', alpha=0.7, linestyle=':')
            plt.plot(epochs, loss_dict['smoothness_loss'], label='Smoothness Loss', alpha=0.7, linestyle=':')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Unweighted)')
        plt.title(f'{filename_id} Unweighted Losses')
        plt.legend(loc='best', fontsize='small', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.xticks(epochs)
        
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both'))
        
        unweighted_plot_filename = os.path.join(output_dir, filename_id + '_losses_unweighted.pdf')
        plt.savefig(unweighted_plot_filename, bbox_inches='tight', dpi=300)
        plt.close()
        tqdm.write(f'Saved unweighted losses plot: {unweighted_plot_filename}')
        
        # ===== PLOT 3: Weighted Losses =====
        plt.figure(figsize=(12, 7))
        
        # Calculate weighted losses
        weighted_mean = [val * self.w_mean for val in loss_dict['mean_loss']]
        weighted_max = [val * self.w_max for val in loss_dict['max_loss']]
        weighted_min = [val * self.w_min for val in loss_dict['min_loss']]
        weighted_ccc = [val * self.w_ccc for val in loss_dict['ccc_loss']]
        weighted_spec = [val * self.w_spec for val in loss_dict['spectral_loss']]
        
        # Plot main losses
        plt.plot(epochs, loss_dict['train_loss'], label='Train Loss (Total)', linewidth=2)
        if len(loss_dict['valid_loss']) > 0:
            plt.plot(epochs, loss_dict['valid_loss'], label='Valid Loss (Total)', linewidth=2)
        
        # Plot weighted component losses
        plt.plot(epochs, weighted_mean, label=f'Mean Loss (w={self.w_mean})', alpha=0.7, linestyle='--')
        plt.plot(epochs, weighted_max, label=f'Max Loss (w={self.w_max})', alpha=0.7, linestyle='--')
        plt.plot(epochs, weighted_min, label=f'Min Loss (w={self.w_min})', alpha=0.7, linestyle='--')
        plt.plot(epochs, weighted_ccc, label=f'CCC Loss (w={self.w_ccc})', alpha=0.7, linestyle='--')
        plt.plot(epochs, weighted_spec, label=f'Spectral Loss (w={self.w_spec})', alpha=0.7, linestyle='--')
        
        # Plot weighted regularization losses if interpretable mode
        if self.interpretable:
            weighted_sparsity = [val * self.lambda_sparsity for val in loss_dict['sparsity_loss']]
            weighted_smoothness = [val * self.lambda_smoothness for val in loss_dict['smoothness_loss']]
            plt.plot(epochs, weighted_sparsity, label=f'Sparsity Loss (λ={self.lambda_sparsity})', alpha=0.7, linestyle=':')
            plt.plot(epochs, weighted_smoothness, label=f'Smoothness Loss (λ={self.lambda_smoothness})', alpha=0.7, linestyle=':')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Weighted)')
        plt.title(f'{filename_id} Weighted Loss Components')
        plt.legend(loc='best', fontsize='small', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.xticks(epochs)
        
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both'))
        
        weighted_plot_filename = os.path.join(output_dir, filename_id + '_losses_weighted.pdf')
        plt.savefig(weighted_plot_filename, bbox_inches='tight', dpi=300)
        plt.close()
        tqdm.write(f'Saved weighted losses plot: {weighted_plot_filename}')
        
        # ===== Save Loss Dictionary as Pickle =====
        import pickle
        loss_dict_filename = os.path.join(output_dir, filename_id + '_loss_dict.pickle')
        with open(loss_dict_filename, 'wb') as f:
            pickle.dump(loss_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        tqdm.write(f'Saved loss dictionary: {loss_dict_filename}')
        
        tqdm.write(f'All plots and loss dictionary saved to: {output_dir}')

    def save_model(self, index):
        if not self.is_main:
            return
            
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        
        # Save the underlying model if using DDP
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        torch.save(model_to_save.state_dict(), model_path)
        tqdm.write('Saved Model Path: ' + model_path)