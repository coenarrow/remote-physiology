import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SpectralLoss(nn.Module):
    """Frequency-domain shape loss using log-magnitude rFFT"""
    def __init__(self, fs=30, fmax=4, eps=1e-8):
        super().__init__()
        self.fs = fs
        self.fmax = fmax
        self.eps = eps

    def forward(self, preds, labels):
        """Compute spectral loss
        preds, labels: [B, C, T]
        Returns: scalar loss"""
        B, C, T = preds.shape

        # Cast to float32 for FFT to avoid ComplexHalf experimental warning with AMP
        preds_f32 = preds.float()
        labels_f32 = labels.float()

        Yp = torch.fft.rfft(preds_f32, dim=-1)
        Yl = torch.fft.rfft(labels_f32, dim=-1)

        mag_p = Yp.abs()
        mag_l = Yl.abs()

        if self.fs is not None and self.fmax is not None:
            freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(preds.device)
            mask = freqs <= self.fmax
            mag_p = mag_p[..., mask]
            mag_l = mag_l[..., mask]

        spec_p = torch.log(mag_p + self.eps)
        spec_l = torch.log(mag_l + self.eps)

        # focus on shape, not absolute scale
        spec_p = spec_p - spec_p.mean(dim=-1, keepdim=True)
        spec_l = spec_l - spec_l.mean(dim=-1, keepdim=True)

        L_spec = (spec_p - spec_l).abs().mean()
        return L_spec

class CCC_Loss(nn.Module):
    """Concordance Correlation Coefficient loss
    Handles multi-channel predictions and labels"""
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        """Compute CCC loss
        preds: torch.Tensor[batch, channels, frames]
        labels: torch.Tensor[batch, channels, frames]
        Returns: torch.Tensor scalar loss"""
        eps = 1e-8
        
        # Ensure 3D tensors
        if preds.dim() == 2:
            preds = preds.unsqueeze(1)
        if labels.dim() == 2:
            labels = labels.unsqueeze(1)
        
        # Compute statistics along time dimension
        mx = preds.mean(dim=-1, keepdim=True)
        my = labels.mean(dim=-1, keepdim=True)
        vx = ((preds - mx) ** 2).mean(dim=-1, keepdim=True)
        vy = ((labels - my) ** 2).mean(dim=-1, keepdim=True)
        cov = ((preds - mx) * (labels - my)).mean(dim=-1, keepdim=True)
        
        # CCC per channel
        ccc = (2 * cov) / (vx + vy + (mx - my) ** 2 + eps)
        
        # Average loss: 1 - CCC
        return (1 - ccc).mean()

class MeanLoss(nn.Module):
    """Mean Squared Error loss on mean values"""
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        """Compute mean loss
        preds, labels: [batch, channels, frames]
        Returns: scalar loss"""
        # Ensure 3D tensors
        if preds.dim() == 2:
            preds = preds.unsqueeze(1)
        if labels.dim() == 2:
            labels = labels.unsqueeze(1)
        
        pred_mean = preds.mean(dim=-1)
        label_mean = labels.mean(dim=-1)
        mean_loss = (pred_mean - label_mean).abs().mean()
        return mean_loss

class PeakLoss(nn.Module):
    """Mean Squared Error loss on peak statistics (max or min)"""
    def __init__(self, tau=0.2, temp=0.03, width=31, kind='max'):
        super().__init__()
        self.tau = tau
        self.temperature = temp
        self.width = width
        self.kind = kind
    
    @staticmethod
    def soft_peak_map(x, width=31, tau=0.1, kind='max'):
        """Differentiable peak indicator in [0,1]
        x: [batch, channels, frames]
        Returns: [batch, channels, frames] peak map
        """
        if kind == 'min':
            x = -x
        
        # Ensure proper shape for max_pool1d
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Local maxima via max-pooling
        m = F.max_pool1d(x, kernel_size=width, stride=1, padding=width//2)
        
        # Distance to local max
        d = (m - x).clamp_min(-1.0)
        
        # Soft indicator
        p = torch.exp(-d / (tau + 1e-12))
        return p, m
    
    @staticmethod
    def soft_peak_max(x, peak_map, temperature=0.05, eps=1e-8, kind='max'):
        """Weighted average of values at peaks
        x: [batch, channels, frames]
        peak_map: [batch, channels, frames]
        Returns: [batch, channels] peak values
        """
        if kind == 'min':
            x_work = -x
        else:
            x_work = x
        
        # Temperature-scaled logits
        logits = x_work / temperature
        
        # Mask to peak regions
        logits = logits + (peak_map + eps).log()
        
        # Attention weights
        a = torch.softmax(logits, dim=-1)
        
        # Weighted average
        return (a * x_work).sum(dim=-1)

    def _peak_stat(self, x, kind='max'):
        """
        Extract peak statistics
        x: [batch, channels, frames]
        Returns: [batch, channels] peak values
        """
        width = x.shape[-1] // 5
        if width % 2 == 0:
            width += 1
        width = max(3, width)  # Ensure minimum width
        
        peak_map, hard_peaks = self.soft_peak_map(x, width=width, tau=self.tau, kind=self.kind)
        soft_peaks = self.soft_peak_max(x, peak_map, temperature=self.temperature, kind=self.kind)
        return soft_peaks, hard_peaks

    def forward(self, preds, labels,kind='max'):
        """Compute peak loss
        preds, labels: [batch, channels, frames]
        Returns: scalar loss"""
        # Ensure 3D tensors
        if preds.dim() == 2:
            preds = preds.unsqueeze(1)
        if labels.dim() == 2:
            labels = labels.unsqueeze(1)
        
        pred_soft_peaks, pred_hard_peaks = self._peak_stat(preds, kind=self.kind)
        label_soft_peaks, label_hard_peaks = self._peak_stat(labels, kind=self.kind)
        peak_loss = (pred_soft_peaks - label_soft_peaks).abs().mean()
        return peak_loss

class PhysHydraLoss(nn.Module):
    """Multi-component loss:
    Mean Squared Error on mean, max, min + CCC + Spectral shape
    Total loss: sum_i w_i * L_i
    Weights w_i are fixed hyperparameters.
    """
    def __init__(self, w_mean, w_max, w_min, w_ccc, w_spec):
        super().__init__()
        self.w_mean = w_mean
        self.w_max = w_max
        self.w_min = w_min
        self.w_ccc = w_ccc
        self.w_spec = w_spec
        self.mean_loss = MeanLoss()
        self.ccc_loss = CCC_Loss()
        self.max_loss = PeakLoss(kind='max')
        self.min_loss = PeakLoss(kind='min')
        self.spectral_loss = SpectralLoss()
        self.losses = {}

    def forward(self, preds, labels):
        """Compute combined loss
        preds, labels: [batch, channels, frames] or [batch, frames]
        Returns: scalar loss
        """
        # Ensure 3D tensors
        if preds.dim() == 2:
            preds = preds.unsqueeze(1)
        if labels.dim() == 2:
            labels = labels.unsqueeze(1)
        
        # Mean Loss
        mean_loss = self.mean_loss(preds, labels)
        ccc = self.ccc_loss(preds, labels)
        max_loss = self.max_loss(preds, labels)
        min_loss = self.min_loss(preds, labels)
        spectral_loss = self.spectral_loss(preds, labels)

        # Store individual losses for logging
        self.losses = {
            'mean_loss': mean_loss.item(),
            'max_loss': max_loss.item(),
            'min_loss': min_loss.item(),
            'ccc_loss': ccc.item(),
            'spectral_loss': spectral_loss.item(),
        }
        
        # Combined loss
        total = (self.w_ccc * ccc +
                 self.w_mean * mean_loss +
                 self.w_max * max_loss +
                 self.w_min * min_loss +
                 self.w_spec * spectral_loss)
        
        return total
    




# class PhysHydraLoss(nn.Module):
#     """
#     Multi-component loss: CCC + mean + peak statistics + spectral shape
#     Metrics: [L_ccc, L_mean, L_max, L_min, L_spec]
#     Total loss: sum_i w_i * L_i, with adaptive weights w_i.

#     Clinical thresholds:
#       - tol_mean: acceptable mean BP error (mmHg)
#       - tol_max:  acceptable systolic (max peak) error (mmHg)
#       - tol_min:  acceptable diastolic (min peak) error (mmHg)

#     When |error| > tol_x, a strong extra penalty is added.
#     """

#     def __init__(
#         self,
#         tau=0.2,
#         temp=0.03,
#         # base (clinical) weights for adaptive weighting
#         w_ccc=0.35,
#         w_mean=0.15,
#         w_max=0.2,
#         w_min=0.2,
#         w_spec=0.1,
#         # adaptive weighting
#         use_adaptive_weights=True,
#         adapt_lambda=2.0,
#         ema_alpha=0.99,
#         # spectral config
#         fs=None,          # sampling rate (Hz), optional
#         fmax=None,        # max frequency (Hz) for spectral loss, optional
#         # clinical thresholds (mmHg errors)
#         tol_mean=5.0,
#         tol_max=8.0,
#         tol_min=8.0,
#         clinical_scale=10.0,      # scales the extra penalty outside thresholds
#         clinical_exponent=2.0,    # exponent for the excess error (>=2 for steep growth)
#         eps=1e-8,
#     ):
#         super().__init__()
#         self.tau = tau
#         self.temperature = temp
#         self.eps = eps

#         # Spectral
#         self.fs = fs
#         self.fmax = fmax

#         # Clinical thresholds
#         self.tol_mean = tol_mean
#         self.tol_max = tol_max
#         self.tol_min = tol_min
#         self.clinical_scale = clinical_scale
#         self.clinical_exponent = clinical_exponent

#         # Base weights
#         base = torch.tensor([w_ccc, w_mean, w_max, w_min, w_spec], dtype=torch.float32)
#         self.register_buffer("base_weights", base)

#         # EMA of each metric loss for normalisation
#         self.use_adaptive_weights = use_adaptive_weights
#         self.adapt_lambda = adapt_lambda
#         self.ema_alpha = ema_alpha
#         self.register_buffer("ema_metrics", torch.ones_like(base))
#         self.last_metrics = {}

#     # -------------------------------------------------------------------------
#     # Utils
#     # -------------------------------------------------------------------------
#     def _ensure_3d(self, x):
#         # [B, T] -> [B, 1, T]
#         if x.dim() == 2:
#             x = x.unsqueeze(1)
#         return x

#     def _clinical_penalty(self, err, tol):
#         """
#         err: |prediction - label| (any shape)
#         tol: scalar tolerance (mmHg)
#         Returns mean penalty: clinical_scale * max(err - tol, 0)^clinical_exponent
#         """
#         excess = (err - tol).clamp_min(0.0)
#         return (self.clinical_scale * excess.pow(self.clinical_exponent)).mean()

#     # -------------------------------------------------------------------------
#     # CCC
#     # -------------------------------------------------------------------------
#     def ccc_loss(self, preds, labels):
#         """CCC loss over [batch, channels, frames]."""
#         eps = self.eps

#         mx = preds.mean(dim=-1, keepdim=True)
#         my = labels.mean(dim=-1, keepdim=True)
#         vx = ((preds - mx) ** 2).mean(dim=-1, keepdim=True)
#         vy = ((labels - my) ** 2).mean(dim=-1, keepdim=True)
#         cov = ((preds - mx) * (labels - my)).mean(dim=-1, keepdim=True)

#         ccc = (2 * cov) / (vx + vy + (mx - my) ** 2 + eps)
#         return (1 - ccc).mean(), ccc.mean().detach()

#     # -------------------------------------------------------------------------
#     # Peak-related utilities
#     # -------------------------------------------------------------------------
#     def _peak_stat(self, x, kind="max"):
#         """
#         Extract soft and hard peak statistics.
#         x: [batch, channels, frames]
#         Returns:
#             soft_peaks: [batch, channels] (differentiable)
#             hard_peaks: [batch, channels] (true global max/min)
#         """
#         B, C, T = x.shape
#         width = T // 5
#         if width % 2 == 0:
#             width += 1
#         width = max(3, width)

#         peak_map, _ = self.soft_peak_map(x, width=width, tau=self.tau, kind=kind)
#         soft_peaks = self.soft_peak_max(x, peak_map, temperature=self.temperature, kind=kind)

#         if kind == "max":
#             hard_peaks = x.max(dim=-1).values
#         else:
#             hard_peaks = x.min(dim=-1).values

#         return soft_peaks, hard_peaks

#     @staticmethod
#     def soft_peak_map(x, width=31, tau=0.1, kind="max"):
#         """
#         Differentiable "peakness" map in [0,1]
#         x: [batch, channels, frames]
#         Returns:
#             peak_map: [batch, channels, frames]
#             m:        local max map (not used in loss)
#         """
#         if kind == "min":
#             x = -x  # minima as maxima of -x

#         if x.dim() == 2:
#             x = x.unsqueeze(1)

#         m = F.max_pool1d(x, kernel_size=width, stride=1, padding=width // 2)
#         d = (m - x).clamp_min(-1.0)
#         p = torch.exp(-d / (tau + 1e-12))
#         return p, m

#     @staticmethod
#     def soft_peak_max(x, peak_map, temperature=0.05, eps=1e-8, kind="max"):
#         """
#         Weighted average of values at (soft) peaks.
#         x: [batch, channels, frames]
#         peak_map: [batch, channels, frames]
#         Returns:
#             [batch, channels] peak values (differentiable)
#         """
#         if kind == "min":
#             x_work = -x
#         else:
#             x_work = x

#         logits = x_work / temperature
#         logits = logits + (peak_map + eps).log()
#         a = torch.softmax(logits, dim=-1)

#         if kind == "min":
#             return -(a * x_work).sum(dim=-1)
#         else:
#             return (a * x_work).sum(dim=-1)

#     # -------------------------------------------------------------------------
#     # Frequency-domain component
#     # -------------------------------------------------------------------------
#     def spectral_loss(self, preds, labels):
#         """
#         Frequency-domain shape loss.
#         Uses log-magnitude rFFT, optionally band-limited to [0, fmax].
#         preds, labels: [B, C, T]
#         Returns:
#             scalar spectral loss
#         """
#         eps = self.eps
#         B, C, T = preds.shape

#         Yp = torch.fft.rfft(preds, dim=-1)
#         Yl = torch.fft.rfft(labels, dim=-1)

#         mag_p = Yp.abs()
#         mag_l = Yl.abs()

#         if self.fs is not None and self.fmax is not None:
#             freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(preds.device)
#             mask = freqs <= self.fmax
#             mag_p = mag_p[..., mask]
#             mag_l = mag_l[..., mask]

#         spec_p = torch.log(mag_p + eps)
#         spec_l = torch.log(mag_l + eps)

#         # focus on shape, not absolute scale
#         spec_p = spec_p - spec_p.mean(dim=-1, keepdim=True)
#         spec_l = spec_l - spec_l.mean(dim=-1, keepdim=True)

#         L_spec = (spec_p - spec_l).abs().mean()
#         return L_spec

#     # -------------------------------------------------------------------------
#     # Adaptive weighting
#     # -------------------------------------------------------------------------
#     def _compute_weights(self, metric_losses):
#         """
#         metric_losses: tensor [5] with
#         [L_ccc, L_mean, L_max, L_min, L_spec]
#         Returns:
#             weights [5]
#         """
#         if not self.use_adaptive_weights:
#             w = self.base_weights / (self.base_weights.sum() + self.eps)
#             return w

#         with torch.no_grad():
#             self.ema_metrics.mul_(self.ema_alpha).add_(
#                 metric_losses.detach() * (1.0 - self.ema_alpha)
#             )

#             normed = metric_losses / (self.ema_metrics + self.eps)

#             logits = torch.log(self.base_weights + self.eps) + self.adapt_lambda * normed
#             w = torch.softmax(logits, dim=0)

#         return w

#     # -------------------------------------------------------------------------
#     # Forward: full composite loss
#     # -------------------------------------------------------------------------
#     def forward(self, preds, labels):
#         """
#         preds, labels: [batch, channels, frames] or [batch, frames]
#         Returns:
#             scalar loss
#         """
#         preds = self._ensure_3d(preds)
#         labels = self._ensure_3d(labels)

#         # CCC
#         L_ccc, ccc_val = self.ccc_loss(preds, labels)

#         # Mean loss + clinical threshold
#         pred_mean = preds.mean(dim=-1)     # [B, C]
#         label_mean = labels.mean(dim=-1)   # [B, C]
#         mean_err = (pred_mean - label_mean).abs()
#         base_mean = (mean_err ** 2).mean()
#         pen_mean = self._clinical_penalty(mean_err, self.tol_mean)
#         L_mean = base_mean + pen_mean

#         # Peak statistics
#         pred_soft_max, pred_hard_max = self._peak_stat(preds, kind="max")
#         label_soft_max, label_hard_max = self._peak_stat(labels, kind="max")
#         pred_soft_min, pred_hard_min = self._peak_stat(preds, kind="min")
#         label_soft_min, label_hard_min = self._peak_stat(labels, kind="min")

#         # Max (systolic) loss + clinical threshold
#         max_err = (pred_soft_max - label_soft_max).abs()
#         base_max = (max_err ** 2).mean()
#         pen_max = self._clinical_penalty(max_err, self.tol_max)
#         L_max = base_max + pen_max

#         # Min (diastolic) loss + clinical threshold
#         min_err = (pred_soft_min - label_soft_min).abs()
#         base_min = (min_err ** 2).mean()
#         pen_min = self._clinical_penalty(min_err, self.tol_min)
#         L_min = base_min + pen_min

#         # Spectral loss
#         L_spec = self.spectral_loss(preds, labels)

#         # Pack metric losses
#         metric_losses = torch.stack([L_ccc, L_mean, L_max, L_min, L_spec], dim=0)  # [5]

#         # Adaptive weights
#         weights = self._compute_weights(metric_losses)

#         # Weighted sum
#         total = (weights * metric_losses).sum()

#         # Store for external logging
#         self.last_metrics = {
#             "L_ccc": L_ccc.detach().item(),
#             "L_mean": L_mean.detach().item(),
#             "L_max": L_max.detach().item(),
#             "L_min": L_min.detach().item(),
#             "L_spec": L_spec.detach().item(),
#             "ccc": ccc_val.item(),
#             "w_ccc": weights[0].detach().item(),
#             "w_mean": weights[1].detach().item(),
#             "w_max": weights[2].detach().item(),
#             "w_min": weights[3].detach().item(),
#             "w_spec": weights[4].detach().item(),
#             "pred_mean_mean": pred_mean.mean().detach().item(),
#             "label_mean_mean": label_mean.mean().detach().item(),
#             "pred_soft_max_mean": pred_soft_max.mean().detach().item(),
#             "pred_hard_max_mean": pred_hard_max.mean().detach().item(),
#             "label_soft_max_mean": label_soft_max.mean().detach().item(),
#             "label_hard_max_mean": label_hard_max.mean().detach().item(),
#             "pred_soft_min_mean": pred_soft_min.mean().detach().item(),
#             "pred_hard_min_mean": pred_hard_min.mean().detach().item(),
#             "label_soft_min_mean": label_soft_min.mean().detach().item(),
#             "label_hard_min_mean": label_hard_min.mean().detach().item(),
#             "base_mean": base_mean.detach().item(),
#             "base_max": base_max.detach().item(),
#             "base_min": base_min.detach().item(),
#             "pen_mean": pen_mean.detach().item(),
#             "pen_max": pen_max.detach().item(),
#             "pen_min": pen_min.detach().item(),
#         }

#         return total

# class CCC_Loss(nn.Module):
#     """Concordance Correlation Coefficient loss
#     Handles multi-channel predictions and labels"""
#     def __init__(self):
#         super().__init__()

#     def forward(self, preds, labels):
#         """Compute CCC loss
#         preds: torch.Tensor[batch, channels, frames]
#         labels: torch.Tensor[batch, channels, frames]
#         Returns: torch.Tensor scalar loss"""
#         eps = 1e-8
        
#         # Ensure 3D tensors
#         if preds.dim() == 2:
#             preds = preds.unsqueeze(1)
#         if labels.dim() == 2:
#             labels = labels.unsqueeze(1)
        
#         # Compute statistics along time dimension
#         mx = preds.mean(dim=-1, keepdim=True)
#         my = labels.mean(dim=-1, keepdim=True)
#         vx = ((preds - mx) ** 2).mean(dim=-1, keepdim=True)
#         vy = ((labels - my) ** 2).mean(dim=-1, keepdim=True)
#         cov = ((preds - mx) * (labels - my)).mean(dim=-1, keepdim=True)
        
#         # CCC per channel
#         ccc = (2 * cov) / (vx + vy + (mx - my) ** 2 + eps)
        
#         # Average loss: 1 - CCC
#         return (1 - ccc).mean()