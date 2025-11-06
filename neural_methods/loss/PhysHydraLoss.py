import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysHydraLoss(nn.Module):
    """Multi-component loss: CCC + peak statistics
    L = w_ccc*(1-CCC) + w_mean*(mean_diff)^2 + w_max*(max_diff)^2 + w_min*(min_diff)^2
    """
    def __init__(self, tau=0.2, temp=0.03, w_ccc=0.4, w_mean=0.2, w_max=0.2, w_min=0.2):
        super().__init__()
        self.tau = tau
        self.temperature = temp
        self.w_ccc = w_ccc
        self.w_mean = w_mean
        self.w_max = w_max
        self.w_min = w_min

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
        
        peak_map = self.soft_peak_map(x, width=width, tau=self.tau, kind=kind)
        peak = self.soft_peak_max(x, peak_map, temperature=self.temperature, kind=kind)
        return peak
    
    def ccc_loss(self, preds, labels):
        """CCC loss
        preds, labels: [batch, channels, frames]
        Returns: scalar loss
        """
        eps = 1e-8
        mx = preds.mean(dim=-1, keepdim=True)
        my = labels.mean(dim=-1, keepdim=True)
        vx = ((preds - mx) ** 2).mean(dim=-1, keepdim=True)
        vy = ((labels - my) ** 2).mean(dim=-1, keepdim=True)
        cov = ((preds - mx) * (labels - my)).mean(dim=-1, keepdim=True)
        ccc = (2 * cov) / (vx + vy + (mx - my) ** 2 + eps)
        return (1 - ccc).mean()

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
        
        # CCC loss
        ccc = self.ccc_loss(preds, labels)
        
        # Peak statistics losses
        max_loss = ((self._peak_stat(preds, 'max') - 
                     self._peak_stat(labels, 'max')) ** 2).mean()
        min_loss = ((self._peak_stat(preds, 'min') - 
                     self._peak_stat(labels, 'min')) ** 2).mean()
        
        # Mean loss
        mean_loss = ((preds.mean(dim=-1) - 
                      labels.mean(dim=-1)) ** 2).mean()
        
        # Combined loss
        total = (self.w_ccc * ccc +
                 self.w_mean * mean_loss +
                 self.w_max * max_loss +
                 self.w_min * min_loss)
        
        return total
    
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
        return p
    
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
        if kind == 'min':
            return -(a * x_work).sum(dim=-1)
        else:
            return (a * x_work).sum(dim=-1)


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