"""The routing-attention block of RhythmFormer."""

from timm.layers import DropPath
from torch import nn

from neural_methods.model.rhythmformer.video_bra import VideoBRA


class VideoBiFormerBlock(nn.Module):
    """One routing-attention block and its convolutional MLP."""

    def __init__(self, dim, region_split, drop_path=0., num_heads=4, t_patch=1,
                 qk_scale=None, topk=4, mlp_ratio=2, side_dwconv=5):
        super().__init__()
        self.t_patch = t_patch
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = VideoBRA(dim=dim, region_split=region_split, num_heads=num_heads,
                             t_patch=t_patch, qk_scale=qk_scale, topk=topk,
                             side_dwconv=side_dwconv)
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = nn.Sequential(nn.Conv3d(dim, int(mlp_ratio * dim), kernel_size=1),
                                 nn.BatchNorm3d(int(mlp_ratio * dim)),
                                 nn.GELU(),
                                 nn.Conv3d(int(mlp_ratio * dim), int(mlp_ratio * dim),
                                           3, stride=1, padding=1),
                                 nn.BatchNorm3d(int(mlp_ratio * dim)),
                                 nn.GELU(),
                                 nn.Conv3d(int(mlp_ratio * dim), dim, kernel_size=1),
                                 nn.BatchNorm3d(dim),
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
