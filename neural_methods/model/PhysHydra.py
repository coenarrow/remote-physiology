import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from mamba_ssm import Mamba
from torch.nn import functional as F

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction):
        """3D channel attention module using adaptive pooling
        in_channels: int, number of input channels
        reduction: int, channel reduction factor"""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.contiguous()
        avg_out = self.fc(self.avg_pool(x))
        
        # Manual max pooling to avoid CUDA alignment issues
        B, C, T, H, W = x.shape
        max_pooled = x.view(B, C, -1).max(dim=2, keepdim=True)[0]
        max_pooled = max_pooled.view(B, C, 1, 1, 1)
        max_out = self.fc(max_pooled)
        
        return x * self.sigmoid(avg_out + max_out)

class LateralConnection(nn.Module):
    def __init__(self, fast_channels=32, slow_channels=64):
        """Lateral connection from fast to slow stream
        fast_channels: int, channels in fast stream
        slow_channels: int, channels in slow stream"""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(fast_channels, slow_channels, [3,1,1], stride=[2,1,1], padding=[1,0,0]),
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(),
        )
        
    def forward(self, slow_path, fast_path):
        return self.conv(fast_path) + slow_path

class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.2):
        """Temporal difference convolution
        theta: float, difference weighting factor"""
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.theta = theta

    def forward(self, x):
        out = self.conv(x)
        if abs(self.theta) < 1e-8 or self.conv.weight.shape[2] <= 1:
            return out
        kernel_diff = self.conv.weight[:,:,0].sum((2,3)) + self.conv.weight[:,:,2].sum((2,3))
        kernel_diff = kernel_diff[:,:,None,None,None]
        out_diff = F.conv3d(x, kernel_diff, self.conv.bias, self.conv.stride, 0, self.conv.dilation, self.conv.groups)
        return out - self.theta * out_diff

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        """Mamba SSM layer with bidirectional processing
        dim: int, model dimension
        d_state: int, SSM state dimension
        d_conv: int, local convolution width
        expand: int, expansion factor"""
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = Mamba(dim, d_state, d_conv, expand, bimamba=True, use_fast_path=False)
        self.drop_path = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        B, C, T, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, T*H*W, C]
        seq_len = x_flat.shape[1]
        
        # Pad sequence to multiple of 8 for CUDA alignment
        pad_len = (8 - seq_len % 8) % 8
        if pad_len > 0:
            x_flat = F.pad(x_flat, (0, 0, 0, pad_len))
        
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        
        # Remove padding before residual connection
        if pad_len > 0:
            x_mamba = x_mamba[:, :seq_len, :]
            x_flat = x_flat[:, :seq_len, :]
        
        x_out = self.norm2(x_flat + self.drop_path(x_mamba))
        x_out = x_out.transpose(1, 2).contiguous().reshape(B, C, T, H, W)
        return x_out

def conv_block(in_channels, out_channels, kernel_size, stride, padding, bn=True, activation='relu'):
    """Create conv3d block with optional batch norm and activation
    activation: str, 'relu' or 'elu'"""
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if bn:
        layers.append(nn.BatchNorm3d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)

class SpatialAttentionBranch(nn.Module):
    def __init__(self, n_channels, base_filters=8, preserve_channels=True):
        """Maintains spatial resolution for interpretability
        n_channels: int, number of input channels (e.g. RGB=3, RGBI=4 etc.)
        base_filters: int, filters per channel branch
        preserve_channels: bool, if True, separate attention per input channel"""
        super().__init__()
        self.n_channels = n_channels
        self.preserve_channels = preserve_channels
        
        if preserve_channels:
            # Separate processing per input channel
            self.channel_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(1, base_filters, [1, 3, 3], padding=[0, 1, 1]),
                    nn.BatchNorm3d(base_filters),
                    nn.ReLU(),
                    nn.Conv3d(base_filters, base_filters//2, [1, 3, 3], padding=[0, 1, 1]),
                ) for _ in range(n_channels)
            ])
            self.attention_heads = nn.ModuleList([
                nn.Conv3d(base_filters//2, 1, [1, 1, 1]) 
                for _ in range(n_channels)
            ])
        else:
            # Process all channels together
            self.combined_branch = nn.Sequential(
                nn.Conv3d(n_channels, base_filters*2, [1, 3, 3], padding=[0, 1, 1]),
                nn.BatchNorm3d(base_filters*2),
                nn.ReLU(),
            )
            self.attention_head = nn.Conv3d(base_filters*2, 1, [1, 1, 1])
    
    def forward(self, x):
        """Returns attention maps and features for downstream fusion"""
        if self.preserve_channels:
            channels = x.split(1, dim=1)
            attention_maps = []
            features = []
            for i, ch in enumerate(channels):
                feat = self.channel_branches[i](ch)
                attn = torch.sigmoid(self.attention_heads[i](feat))
                attention_maps.append(attn)
                features.append(feat * attn)
            return torch.cat(attention_maps, dim=1), torch.cat(features, dim=1)
        else:
            feat = self.combined_branch(x)
            attn = torch.sigmoid(self.attention_head(feat))
            return attn, feat * attn

class ChannelImportanceScorer(nn.Module):
    def __init__(self, n_channels, hidden_dim=64):
        """Learns global importance weights for each input channel
        n_channels: int, number of input channels
        hidden_dim: int, hidden dimension for scoring network"""
        super().__init__()
        self.scorer = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global pooling
            nn.Conv3d(n_channels, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, n_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Returns channel weights [B, N, 1, 1, 1] and weighted features"""
        weights = self.scorer(x)
        return weights, x * weights

class InterpretabilityHead(nn.Module):
    def __init__(self, feature_dim, n_channels):
        """Aggregates all interpretability outputs
        feature_dim: int, dimension of features
        n_channels: int, number of input channels"""
        super().__init__()
        self.spatial_maps = None
        self.channel_scores = None
        self.temporal_attention = None
        
    def forward(self, spatial_attn, channel_importance, features):
        """Store interpretability tensors for analysis"""
        self.spatial_maps = spatial_attn.detach()
        self.channel_scores = channel_importance.detach()

        # Better sparsity: encourage binary attention (close to 0 or 1)
        # This penalizes values around 0.5 (uniform attention)
        # L = sum(4 * a * (1-a)) which is minimized when a is 0 or 1
        sparsity = (4 * spatial_attn * (1 - spatial_attn)).mean()

        # Concentration loss: encourage top regions to contain most attention mass
        # Flatten spatial dimensions and compute Gini-like concentration
        B, C, T, H, W = spatial_attn.shape
        attn_flat = spatial_attn.reshape(B, C, T, -1)  # [B, C, T, H*W]
        attn_sorted, _ = torch.sort(attn_flat, dim=-1, descending=True)

        # Top 20% of pixels should contain >80% of mass
        top_k = max(1, int(0.2 * (H * W)))
        top_mass = attn_sorted[..., :top_k].sum(dim=-1)  # [B, C, T]
        total_mass = attn_sorted.sum(dim=-1).clamp(min=1e-8)  # [B, C, T]
        concentration = 1.0 - (top_mass / total_mass).mean()  # Penalize if top 20% has <100% mass

        # Spatial smoothness (unchanged)
        smoothness = ((spatial_attn[:,:,:,1:] - spatial_attn[:,:,:,:-1])**2).mean()

        # Combine: sparsity encourages 0/1, concentration encourages clustering
        combined_sparsity = sparsity + 0.5 * concentration

        return features, (combined_sparsity, smoothness)
    
class PhysHydra(nn.Module):
    def __init__(self, in_channels=3, out_signals=1, theta=0.5, drop_rate1=0.25, 
                 drop_rate2=0.5, frames=128, interpretable=True, preserve_channels=True, debug=False):
        """Multi-channel interpretable rPPG extraction network
        in_channels: int, input video channels (RGB=3, multispectral=N)
        out_signals: int, output signals (1=PPG, multi for ABP/ECG/etc)
        theta: float, CDC_T parameter
        drop_rate1/2: float, dropout rates
        frames: int, output temporal length
        interpretable: bool, enable interpretability modules
        preserve_channels: bool, separate attention per input channel
        debug: bool, enable debug mode with extra checks"""
        super().__init__()
        
        self.in_channels = in_channels
        self.out_signals = out_signals
        self.interpretable = interpretable
        self.frames = frames
        self.debug = debug
        
        # Main processing path
        self.ConvBlock1 = conv_block(in_channels, 16, [1,5,5], 1, [0,2,2])
        self.ConvBlock2 = conv_block(16, 32, [3,3,3], 1, 1)
        self.ConvBlock3 = conv_block(32, 64, [3,3,3], 1, 1)
        self.ConvBlock4 = conv_block(64, 64, [4,1,1], [4,1,1], 0)
        self.ConvBlock5 = conv_block(64, 32, [2,1,1], [2,1,1], 0)
        self.ConvBlock6 = conv_block(32, 32, [3,1,1], 1, [1,0,0], activation='elu')
        
        # Interpretability modules
        if interpretable:
            self.spatial_attention = SpatialAttentionBranch(
                n_channels=in_channels, 
                base_filters=8, 
                preserve_channels=preserve_channels
            )
            self.channel_importance = ChannelImportanceScorer(
                n_channels=in_channels, 
                hidden_dim=32
            )
            self.interpretability = InterpretabilityHead(
                feature_dim=48, 
                n_channels=in_channels
            )
            # Adjust fusion dimensions for spatial features
            spatial_feat_dim = (4 * in_channels) if preserve_channels else 16
            self.spatial_fusion = nn.Conv3d(spatial_feat_dim, 8, [1,1,1])
            # Modify final conv to accept additional features
            self.ConvBlockLast = nn.Conv3d(48 + 8, out_signals, [1,1,1], 1, 0)
        else:
            self.ConvBlockLast = nn.Conv3d(48, out_signals, [1,1,1], 1, 0)
        
        # Dual-stream blocks
        self.Block1 = self._build_block(64, theta)
        self.Block2 = self._build_block(64, theta)
        self.Block3 = self._build_block(64, theta)
        self.Block4 = self._build_block(32, theta)
        self.Block5 = self._build_block(32, theta)
        self.Block6 = self._build_block(32, theta)
        
        # Upsampling
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(64, 64, [3,1,1], 1, (1,0,0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(96, 48, [3,1,1], 1, (1,0,0)),
            nn.BatchNorm3d(48),
            nn.ELU(),
        )
        
        # Pooling and fusion
        self.MaxpoolSpa = nn.MaxPool3d((1,2,2), stride=(1,2,2), padding=(0,1,1))
        self.fuse_1 = LateralConnection(32, 64)
        self.fuse_2 = LateralConnection(32, 64)
        
        # Dropout
        self.drop_1 = nn.Dropout(drop_rate1)
        self.drop_2 = nn.Dropout(drop_rate1)
        self.drop_3 = nn.Dropout(drop_rate2)
        self.drop_4 = nn.Dropout(drop_rate2)
        self.drop_5 = nn.Dropout(drop_rate2)
        self.drop_6 = nn.Dropout(drop_rate2)
        
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def _build_block(self, channels, theta):
        """Build processing block with CDC_T, Mamba, and attention"""
        return nn.Sequential(
            CDC_T(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            MambaLayer(dim=channels),
            ChannelAttention3D(in_channels=channels, reduction=2),
        )
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        if self.debug:
            assert not torch.isnan(x).any(), f"NaN in model input"
            assert not torch.isinf(x).any(), f"Inf in model input"
            assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        
        # Interpretability branch (high resolution)
        if self.interpretable:
            # Extract spatial attention early (before heavy downsampling)
            x_early = x  # Store original input
            channel_weights, x_weighted = self.channel_importance(x)
            
            # After initial feature extraction for spatial attention
            x_feat = self.ConvBlock1(x_weighted)
            x_feat = self.MaxpoolSpa(x_feat)  # Now H/2, W/2
            spatial_attn, spatial_features = self.spatial_attention(x_early[:,:,:,:H,:W])
            

        # Main processing path
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpa(x)
        
        if self.debug:
            assert not torch.isnan(x).any(), "NaN after initial conv blocks"
        
        # Dual-stream processing
        s_x = self.ConvBlock4(x)  # Slow stream
        f_x = self.ConvBlock5(x)  # Fast stream
        
        s_x1 = self.drop_1(self.MaxpoolSpa(self.Block1(s_x)))
        f_x1 = self.drop_2(self.MaxpoolSpa(self.Block4(f_x)))
        s_x1 = self.fuse_1(s_x1, f_x1)
        
        s_x2 = self.drop_3(self.MaxpoolSpa(self.Block2(s_x1)))
        f_x2 = self.drop_4(self.MaxpoolSpa(self.Block5(f_x1)))
        s_x2 = self.fuse_2(s_x2, f_x2)
        
        s_x3 = self.drop_5(self.upsample1(self.Block3(s_x2)))
        f_x3 = self.drop_6(self.ConvBlock6(self.Block6(f_x2)))
        
        # Align temporal dimensions
        if s_x3.shape[2] != f_x3.shape[2]:
            f_x3 = F.interpolate(f_x3, size=(s_x3.shape[2], f_x3.shape[3], f_x3.shape[4]), mode='nearest')
        
        x_fusion = torch.cat((f_x3, s_x3), dim=1)
        x_final = self.upsample2(x_fusion)  # [B, 48, 128, 8, 8]
        
        if self.debug:
            assert not torch.isnan(x_final).any(), "NaN before final processing"
        
        # Incorporate spatial attention features
        if self.interpretable:
            # Adaptive pool to match x_final's dimensions exactly
            spatial_features_matched = F.interpolate(
                spatial_features,
                size=(x_final.shape[2], x_final.shape[3], x_final.shape[4]),
                mode='trilinear',
                align_corners=False
            )
            
            # Compress channels if needed
            spatial_features_compressed = self.spatial_fusion(spatial_features_matched)
            
            # Store interpretability outputs
            x_final, (reg_sparsity, reg_smoothness) = self.interpretability(spatial_attn, channel_weights, x_final)
            
            # Concatenate with main features
            x_final = torch.cat([x_final, spatial_features_compressed], dim=1)
        
        x_final = self.poolspa(x_final)  # This pools to [B, 48+8, frames, 1, 1]
        x_final = self.ConvBlockLast(x_final)
        out = x_final.squeeze(-1).squeeze(-1)
        
        if self.debug:
            assert not torch.isnan(out).any(), "NaN in model output"
            assert out.shape == (B, self.out_signals, self.frames), f"Unexpected output shape: {out.shape}"
        
        if self.interpretable:
            return out, (reg_sparsity, reg_smoothness)
        else:
            return out, (None, None)
    
    def get_interpretability_maps(self):
        """Access stored interpretability outputs
        Returns: (spatial_maps, channel_scores)"""
        if not self.interpretable:
            return None, None
        return self.interpretability.spatial_maps, self.interpretability.channel_scores
    
    def get_regularisation_loss(self):
        """Get sparsity and smoothness losses for training"""
        if not self.interpretable:
            return 0.0, 0.0
        return self.interpretability.get_regularisation_loss()