"""Trainer for the Neckflix batch-dict contract.

One trainer serves every dict-contract architecture, because the only thing
that differs between them is how a clip becomes a ``(B, S, T)`` tensor — and
that lives in the model (see :class:`DictModel`). Everything here is the part
that would otherwise be copy-pasted per model: DDP, AMP, the masked
multi-signal loss, checkpointing, and per-signal evaluation.

What makes it different from the legacy per-model trainers is that predictions
and labels stay *keyed by signal name* from the loader all the way to the
metric report, so a model predicting ABP and CVP together is scored on each
separately, and a window whose recording lacks one of them is simply not
counted for that signal (``label_mask``).
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dataset.data_loader.label_transforms import minmax_inverse, zscore_inverse
from dataset.data_loader.neckflix_config import frame_size
from evaluation.metrics_report import report_hr_metrics
from evaluation.post_process import calculate_metric_per_video
from neural_methods.batch import (
    LABEL_MASK, LABEL_STATS, LABELS, METADATA, PREDICTIONS,
    detach_to_cpu, iter_samples, move_to_device,
)
from neural_methods.frame_transforms import FrameTransform
from neural_methods.loss.MaskedMultiSignalLoss import MaskedMultiSignalLoss
from neural_methods.signals import resolve_channels, resolve_traces
from neural_methods.trainer.BaseTrainer import BaseTrainer

NCOLS = 80

#: Shortest window the HR post-processing can filter (filtfilt padlen).
MIN_HR_WINDOW = 9

#: Per-window label normalisations and the exact inverse of each.
_INVERSES = {'zscore': zscore_inverse, 'minmax': minmax_inverse}

#: Units the physical-scale report is in, per canonical signal.
SIGNAL_UNITS = {'ABP': 'mmHg', 'CVP': 'mmHg', 'ECG': 'uV', 'PPG': 'a.u.',
                'RESP': 'V', 'EDA': 'uS', 'SPO2': '%'}


def _build_physmamba(config, channels, traces, transform):
    from neural_methods.model.PhysMamba import PhysMamba
    return PhysMamba(channels=channels, traces=traces, frame_transform=transform)


def _build_deepphys(config, channels, traces, transform):
    from neural_methods.model.DeepPhys import DeepPhys
    from neural_methods.model.SignalDictWrapper import SignalDictWrapper
    size = transform.size or (config.TRAIN.DATA.PREPROCESS.RESIZE.H,
                              config.TRAIN.DATA.PREPROCESS.RESIZE.W)
    backbone = DeepPhys(in_channels=len(channels), out_signals=len(traces),
                        img_size=size[0])
    return SignalDictWrapper(backbone, channels=channels, traces=traces,
                             input_mode='frames2d', frame_transform=transform)


#: Architectures that speak the batch-dict contract. Add a builder here to make
#: a model available to ``TOOLBOX_MODE: train_and_test`` on Neckflix.
MODEL_REGISTRY = {
    'PhysMamba': _build_physmamba,
    'DeepPhys': _build_deepphys,
}


def build_model(config, data_config):
    """Construct the configured dict-contract model from a config pair."""
    name = config.MODEL.NAME
    builder = MODEL_REGISTRY.get(name)
    if builder is None:
        raise ValueError(
            f"Model {name!r} does not speak the Neckflix dict contract yet. "
            f"Available: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    channels = resolve_channels(data_config)
    traces = resolve_traces(data_config)
    data_types = [t for t in data_config.PREPROCESS.DATA_TYPE if t] or ['Standardized']
    transform = FrameTransform(data_types, size=frame_size(data_config))
    return builder(config, channels, traces, transform)


class MultiSignalTrainer(BaseTrainer):
    """Train/validate/test any :class:`DictModel` on the Neckflix zarr cache."""

    def __init__(self, config, data_loader, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.local_rank = int(os.environ.get('LOCAL_RANK', self.rank))
        self.device = self._select_device()
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.min_valid_loss = None
        self.best_epoch = 0

        data_config = (config.TRAIN.DATA if config.TOOLBOX_MODE == "train_and_test"
                       else config.TEST.DATA)
        self.traces = resolve_traces(data_config)
        self.channels = resolve_channels(data_config)
        self.frame_rate = config.TEST.DATA.FS or config.TRAIN.DATA.FS
        self.label_norm = config.TEST.DATA.PREPROCESS.NECKFLIX.LABEL_NORM
        if self.label_norm not in _INVERSES:
            raise ValueError(f"Unknown LABEL_NORM {self.label_norm!r}; "
                             f"known: {sorted(_INVERSES)}")

        model = build_model(config, data_config).to(self.device)
        if self.world_size > 1:
            device_ids = [self.local_rank] if self.device.type == 'cuda' else None
            model = DDP(model, device_ids=device_ids,
                        output_device=self.local_rank if device_ids else None)
        self.model = model

        self.criterion = MaskedMultiSignalLoss(
            self.traces, base=getattr(config.TRAIN, 'LOSS', 'negpearson'))

        self.use_amp = bool(getattr(config.TRAIN, 'USE_AMP', False)) and self.device.type == 'cuda'
        self.amp_dtype = torch.float16 if getattr(config.TRAIN, 'AMP_DTYPE', '') == 'float16' \
            else torch.bfloat16
        self.scaler = torch.amp.GradScaler('cuda') if (self.use_amp and self.amp_dtype == torch.float16) else None

        self.optimizer = None
        self.scheduler = None
        self.train_sampler = None
        if config.TOOLBOX_MODE == "train_and_test":
            if data_loader.get("train") is None:
                raise ValueError("train_and_test needs a train dataloader")
            self.num_train_batches = len(data_loader["train"])
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR,
                                        weight_decay=0.0005)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=max(self.num_train_batches, 1))
            if self.world_size > 1:
                self.train_sampler = data_loader["train"].sampler
        elif config.TOOLBOX_MODE != "only_test":
            raise ValueError("MultiSignalTrainer initialized in incorrect toolbox mode!")

    # --- setup helpers ---------------------------------------------------
    def _select_device(self):
        """Honour ``config.DEVICE`` when it asks for CPU; otherwise best available.

        ``DEVICE: cpu`` is how the smoke configs stay runnable on a machine that
        happens to have a GPU, so it has to actually mean CPU.
        """
        requested = str(getattr(self.config, 'DEVICE', '') or '').lower()
        if requested.startswith('cpu'):
            return torch.device('cpu')
        if torch.cuda.is_available():
            # LOCAL_RANK, not the global rank: on a multi-node job rank 5 is
            # local GPU 1 of node 1, not a sixth GPU on this node.
            return torch.device(f'cuda:{self.local_rank}')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _autocast(self):
        return torch.amp.autocast(self.device.type, dtype=self.amp_dtype,
                                  enabled=self.use_amp)

    # --- training --------------------------------------------------------
    def _loss_for(self, batch):
        """Forward one batch and reduce it to the masked multi-signal loss."""
        out = self.model(batch)
        return self.criterion(out[PREDICTIONS], batch[LABELS], batch[LABEL_MASK]), out

    def train(self, data_loader):
        if data_loader.get("train") is None:
            raise ValueError("No data for train")
        if self.world_size > 1:
            dist.barrier()

        mean_training_losses, mean_valid_losses, lrs = [], [], []
        for epoch in range(self.max_epoch_num):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            if self.is_main:
                print(f"\n====Training Epoch: {epoch}====")
            self.model.train()
            train_loss = []
            tbar = tqdm(data_loader["train"], ncols=NCOLS) if self.is_main else data_loader["train"]
            for batch in tbar:
                batch = move_to_device(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    loss, _ = self._loss_for(batch)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                lrs.append(self.scheduler.get_last_lr())
                self.scheduler.step()
                train_loss.append(loss.item())
                if self.is_main:
                    tbar.set_description(f"Train epoch {epoch}")
                    tbar.set_postfix(loss=loss.item())

            epoch_loss = self._reduce_mean(train_loss)
            mean_training_losses.append(epoch_loss)
            if self.is_main:
                # The progress bar only ever showed the last batch; the epoch
                # mean is what tells you whether training is going anywhere.
                print(f"mean training loss: {epoch_loss:.4f}")
            self.save_model(epoch)

            if not self.config.TEST.USE_LAST_EPOCH and data_loader.get("valid") is not None:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                if self.is_main:
                    print('validation loss: ', valid_loss)
                    if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                        self.min_valid_loss = valid_loss
                        self.best_epoch = epoch
                        print(f"Update best model! Best epoch: {self.best_epoch}")
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        if not self.config.TEST.USE_LAST_EPOCH and self.is_main:
            print(f"best trained epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")
        if self.config.TRAIN.PLOT_LOSSES_AND_LR and self.is_main:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        if data_loader.get("valid") is None:
            raise ValueError("No data for valid")
        if self.is_main:
            print("\n ====Validing===")
        self.model.eval()
        valid_loss = []
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=NCOLS) if self.is_main else data_loader["valid"]
            for batch in vbar:
                batch = move_to_device(batch, self.device)
                with self._autocast():
                    loss, _ = self._loss_for(batch)
                valid_loss.append(loss.item())
                if self.is_main:
                    vbar.set_description("Validation")
                    vbar.set_postfix(loss=loss.item())
        return self._reduce_mean(valid_loss)

    def _reduce_mean(self, values):
        """Mean of a per-rank list, averaged across ranks when distributed."""
        local = float(np.mean(values)) if values else float('nan')
        if self.world_size > 1:
            tensor = torch.tensor([local], device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            return tensor.cpu().item()
        return local

    # --- testing ---------------------------------------------------------
    def _load_weights_for_test(self):
        model = self._unwrap_model()
        if self.config.TOOLBOX_MODE == "only_test":
            path = self.config.INFERENCE.MODEL_PATH
            if not os.path.exists(path):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            print("Testing uses pretrained model!\n" + path)
        elif self.config.TEST.USE_LAST_EPOCH:
            path = os.path.join(self.model_dir,
                                f"{self.model_file_name}_Epoch{self.max_epoch_num - 1}.pth")
            print("Testing uses last epoch as non-pretrained model!\n" + path)
        else:
            path = os.path.join(self.model_dir,
                                f"{self.model_file_name}_Epoch{self.best_epoch}.pth")
            print("Testing uses best epoch selected using model selection as non-pretrained model!\n" + path)
        model.load_state_dict(torch.load(path, map_location=self.device))

    def test(self, data_loader):
        """Run inference and report metrics per predicted signal."""
        if not self.is_main:
            return None
        if data_loader.get("test") is None:
            raise ValueError("No data for test")
        print("\n===Testing===")
        self._load_weights_for_test()
        self.model = self.model.to(self.device)
        self.model.eval()

        windows = []          # per (recording, signal) window records, for saving
        stats = defaultdict(lambda: defaultdict(list))
        with torch.no_grad():
            for batch in tqdm(data_loader["test"], ncols=NCOLS):
                on_device = move_to_device(batch, self.device)
                with self._autocast():
                    out = self.model(on_device)
                out = detach_to_cpu(out)
                for sample in iter_samples(out):
                    windows.extend(self._score_sample(sample, stats))

        print('')
        report = {}
        hr_method = 'Peak' if self.config.INFERENCE.EVALUATION_METHOD == "peak detection" else 'FFT'
        for signal in self.traces:
            group = stats.get(signal)
            if not group:
                print(f"[{signal}] no windows carried this label — skipped")
                continue
            print(f"--- {signal}: {len(group['gt_hr'])} windows ---")
            unit = SIGNAL_UNITS.get(signal, 'physical units')
            print(f"[{signal}] waveform Pearson: {np.mean(group['pearson']):.4f}  "
                  f"MAE: {np.mean(group['mae']):.4f}  RMSE: {np.mean(group['rmse']):.4f} "
                  "(normalised units)")
            print(f"[{signal}] waveform MAE: {np.mean(group['mae_physical']):.4f}  "
                  f"RMSE: {np.mean(group['rmse_physical']):.4f} ({unit}, at the "
                  "window's own scale)")
            report[signal] = report_hr_metrics(
                group['gt_hr'], group['pred_hr'], group['snr'], group['macc'],
                metrics=self.config.TEST.METRICS, config=self.config,
                filename_id=self._filename_id(), hr_method=hr_method, scope=signal)
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_dict_outputs(windows)
        return report

    def _score_sample(self, sample, stats):
        """Accumulate one window's per-signal metrics; return its saveable records."""
        metadata = sample[METADATA]
        hr_method = 'Peak' if self.config.INFERENCE.EVALUATION_METHOD == "peak detection" else 'FFT'
        records = []
        for signal in self.traces:
            if not bool(sample[LABEL_MASK][signal]):
                continue
            prediction = sample[PREDICTIONS][signal].float().numpy()
            label = sample[LABELS][signal].float().numpy()
            group = stats[signal]
            group['mae'].append(float(np.mean(np.abs(prediction - label))))
            group['rmse'].append(float(np.sqrt(np.mean((prediction - label) ** 2))))
            group['pearson'].append(_safe_pearson(prediction, label))

            # Physical units, via the exact inverse of the normalisation the
            # loader applied. Each window carries its own stats, so this is the
            # error you would see given a perfect estimate of that window's
            # scale -- it measures shape, expressed in mmHg (or whatever the
            # signal's units are), not absolute-level accuracy. Absolute level
            # is a separate problem the per-window normalisation deliberately
            # removes.
            physical_pred, physical_label = self._to_physical(sample, signal)
            error = physical_pred - physical_label
            group['mae_physical'].append(float(np.mean(np.abs(error))))
            group['rmse_physical'].append(float(np.sqrt(np.mean(error ** 2))))
            if len(prediction) >= MIN_HR_WINDOW:
                gt_hr, pred_hr, snr, macc = calculate_metric_per_video(
                    prediction, label, diff_flag=False, fs=self.frame_rate,
                    hr_method=hr_method)
                group['gt_hr'].append(gt_hr)
                group['pred_hr'].append(pred_hr)
                group['snr'].append(snr)
                group['macc'].append(macc)
            records.append({
                'signal': signal,
                'recording_id': metadata['recording_id'],
                'camera_id': metadata['camera_id'],
                'start_frame': int(metadata['start_frame']),
                'prediction': prediction,
                'label': label,
                'label_stats': {k: float(v) for k, v in sample[LABEL_STATS][signal].items()},
            })
        return records

    def _to_physical(self, sample, signal):
        """Prediction and label for one signal back in physical units."""
        inverse = _INVERSES[self.label_norm]
        stats = sample[LABEL_STATS][signal]
        return (inverse(sample[PREDICTIONS][signal].float(), stats).numpy(),
                inverse(sample[LABELS][signal].float(), stats).numpy())

    def _filename_id(self):
        if self.config.TOOLBOX_MODE == 'train_and_test':
            return self.model_file_name
        root = os.path.basename(self.config.INFERENCE.MODEL_PATH).split(".pth")[0]
        return f"{root}_{self.config.TEST.DATA.DATASET}"

    def save_dict_outputs(self, windows):
        """Persist every scored window, keyed by signal and recording.

        Kept flat and self-describing (one record per window, carrying its own
        ``label_stats``) so downstream analysis can invert the normalisation
        without re-reading the cache.
        """
        output_dir = self.config.TEST.OUTPUT_SAVE_DIR
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, self._filename_id() + '_outputs.pickle')
        payload = {
            'windows': windows,
            'traces': list(self.traces),
            'channels': list(self.channels),
            'fs': self.frame_rate,
            'label_norm': self.config.TEST.DATA.PREPROCESS.NECKFLIX.LABEL_NORM,
        }
        with open(path, 'wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saving outputs to:', path)

    def save_model(self, index):
        if not self.is_main:
            return
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{index}.pth")
        torch.save(self._unwrap_model().state_dict(), path)
        print('Saved Model Path: ', path)


def _safe_pearson(prediction, label):
    """Correlation that returns 0 for a constant trace instead of NaN."""
    p = prediction - prediction.mean()
    l = label - label.mean()
    denominator = np.sqrt((p ** 2).sum() * (l ** 2).sum())
    return float((p * l).sum() / denominator) if denominator > 0 else 0.0
