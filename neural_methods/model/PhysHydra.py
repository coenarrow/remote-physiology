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
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ret = x * self.sigmoid(avg_out + max_out)
        return ret

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
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, 
                            padding, dilation, groups, bias)
        self.theta = theta

    def forward(self, x):
        out = self.conv(x)
        if abs(self.theta) < 1e-8 or self.conv.weight.shape[2] <= 1:
            return out
        kernel_diff = self.conv.weight[:,:,0].sum((2,3)) + self.conv.weight[:,:,2].sum((2,3))
        kernel_diff = kernel_diff[:,:,None,None,None]
        out_diff = F.conv3d(x, kernel_diff, self.conv.bias, self.conv.stride,
                          0, self.conv.dilation, self.conv.groups)
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
        self.mamba = Mamba(dim, d_state, d_conv, expand, bimamba=True)
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
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = self.norm2(x_flat + self.drop_path(x_mamba))
        return x_out.transpose(1, 2).reshape(B, C, T, H, W)

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

class PhysHydra(nn.Module):
    def __init__(self, in_channels=3, out_signals=1, theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=128):
        """Multi-channel rPPG extraction network
        in_channels: int, input video channels (1=grayscale, 3=RGB, etc)
        out_signals: int, output signals (1=PPG, more for ECG/resp/etc)
        theta: float, CDC_T parameter
        drop_rate1/2: float, dropout rates
        frames: int, output temporal length"""
        super().__init__()
        
        self.ConvBlock1 = conv_block(in_channels, 16, [1,5,5], 1, [0,2,2])
        self.ConvBlock2 = conv_block(16, 32, [3,3,3], 1, 1)
        self.ConvBlock3 = conv_block(32, 64, [3,3,3], 1, 1)
        self.ConvBlock4 = conv_block(64, 64, [4,1,1], [4,1,1], 0)
        self.ConvBlock5 = conv_block(64, 32, [2,1,1], [2,1,1], 0)
        self.ConvBlock6 = conv_block(32, 32, [3,1,1], 1, [1,0,0], activation='elu')
        
        self.Block1 = self._build_block(64, theta)
        self.Block2 = self._build_block(64, theta)
        self.Block3 = self._build_block(64, theta)
        self.Block4 = self._build_block(32, theta)
        self.Block5 = self._build_block(32, theta)
        self.Block6 = self._build_block(32, theta)
        
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
        
        self.ConvBlockLast = nn.Conv3d(48, out_signals, [1,1,1], 1, 0)
        self.MaxpoolSpa = nn.MaxPool3d((1,2,2), stride=(1,2,2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        
        self.fuse_1 = LateralConnection(32, 64)
        self.fuse_2 = LateralConnection(32, 64)
        
        self.drop_1 = nn.Dropout(drop_rate1)
        self.drop_2 = nn.Dropout(drop_rate1)
        self.drop_3 = nn.Dropout(drop_rate2)
        self.drop_4 = nn.Dropout(drop_rate2)
        self.drop_5 = nn.Dropout(drop_rate2)
        self.drop_6 = nn.Dropout(drop_rate2)
        
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def _build_block(self, channels, theta):
        """Build processing block with CDC_T, Mamba, and attention
        channels: int, number of channels
        theta: float, CDC_T parameter"""
        return nn.Sequential(
            CDC_T(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            MambaLayer(dim=channels),
            ChannelAttention3D(in_channels=channels, reduction=2),
        )
    
    def forward(self, x):
        B, C, T, W, H = x.shape
        
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpa(x)
        
        s_x = self.ConvBlock4(x)
        f_x = self.ConvBlock5(x)
        
        s_x1 = self.drop_1(self.MaxpoolSpa(self.Block1(s_x)))
        f_x1 = self.drop_2(self.MaxpoolSpa(self.Block4(f_x)))
        s_x1 = self.fuse_1(s_x1, f_x1)
        
        s_x2 = self.drop_3(self.MaxpoolSpa(self.Block2(s_x1)))
        f_x2 = self.drop_4(self.MaxpoolSpa(self.Block5(f_x1)))
        s_x2 = self.fuse_2(s_x2, f_x2)
        
        s_x3 = self.drop_5(self.upsample1(self.Block3(s_x2)))
        f_x3 = self.drop_6(self.ConvBlock6(self.Block6(f_x2)))
        
        if s_x3.shape[2] != f_x3.shape[2]:
            f_x3 = F.interpolate(f_x3, size=(s_x3.shape[2], f_x3.shape[3], f_x3.shape[4]), mode='nearest')
        
        x_fusion = torch.cat((f_x3, s_x3), dim=1)
        x_final = self.upsample2(x_fusion)
        x_final = self.poolspa(x_final)
        x_final = self.ConvBlockLast(x_final)
        
        return x_final.squeeze(-1).squeeze(-1)