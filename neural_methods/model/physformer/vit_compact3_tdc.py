"""The published PhysFormer network: 3-D stem, tube tokens, three TDC stages, head."""

import torch
from einops import rearrange, reduce
from torch import nn
from torch.nn import functional as F

from neural_methods.model._shared_modules.utils import nearest_multiple, require_min_frame
from neural_methods.model.physformer.transformer_tdc import TransformerTDC


class ViTCompact3TDC(nn.Module):
    """stem_3DCNN + ST-ViT with local depth-wise spatio-temporal MLP.

    ``in_channels`` is the only width that differs from the published network;
    at ``3`` this is exactly the original. Any frame size from 8x8 and any
    window length are accepted (see the docstring of
    :mod:`neural_methods.model.physformer.physformer` for how, and why the
    paper's shape is untouched).
    """

    def __init__(
        self,
        min_frame: int,
        head_temporal_upsample: int,
        patches: int = 4,
        dim: int = 96,
        ff_dim: int = 144,
        num_heads: int = 4,
        num_layers: int = 12,
        dropout_rate: float = 0.1,
        in_channels: int = 3,
        theta: float = 0.7,
    ):
        """``min_frame`` is the smallest frame the stem leaves anything of and
        ``head_temporal_upsample`` the head's temporal factor; both are
        PhysFormer's, passed in."""
        super().__init__()

        self.min_frame = min_frame
        self.dim = dim
        self.in_channels = in_channels
        self.patch = patches                       # (4, 4, 4) tubes in the paper

        if patches != head_temporal_upsample:
            raise ValueError(
                f"PhysFormer's head upsamples time by {head_temporal_upsample}x "
                f"(two Upsample(scale_factor=(2,1,1)) stages), so the temporal "
                f"patch size must be {head_temporal_upsample} for the prediction "
                f"to come back at the window length; got {patches}."
            )

        # Patch embedding: one [ft x fh x fw] tube -> one token.
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(patches,) * 3,
                                         stride=(patches,) * 3)

        stage_layers = num_layers // 3
        self.transformer1 = TransformerTDC(
            num_layers=stage_layers, dim=dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout_rate, theta=theta)
        self.transformer2 = TransformerTDC(
            num_layers=stage_layers, dim=dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout_rate, theta=theta)
        self.transformer3 = TransformerTDC(
            num_layers=stage_layers, dim=dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout_rate, theta=theta)

        # --- The one deviation: the first layer takes ``in_channels``, not 3. ---
        self.stem_0 = nn.Sequential(
            nn.Conv3d(in_channels, dim // 4, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.stem_1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.stem_2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim // 2, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim // 2),
            nn.ELU(),
        )

        # The readout: one plane, bias kept (the trainer seeds it with the
        # trace's physiological prior) and deliberately no activation, so an
        # absolute-class signal is expressible in its own units.
        self.conv_block_last = nn.Conv1d(dim // 2, 1, 1, stride=1, padding=0)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)

    def forward(self, x, gra_sharp):
        """``(B, C_in, T, H, W)`` -> ``((B, 1, T), score_1, score_2, score_3)``."""
        frames, height, width = x.shape[2:]
        require_min_frame("PhysFormer", self.min_frame, height, width)

        x = self.stem_0(x)
        x = self.stem_1(x)
        x = self.stem_2(x)                                 # [B, dim, T, H/8, W/8]

        # Any size: bring the stem's output to whole tubes. At the paper's
        # 128x128 frames and 160-frame windows the target equals the input
        # and this is skipped, so that path stays the published network.
        target = tuple(nearest_multiple(n, self.patch) for n in x.shape[2:])
        if tuple(x.shape[2:]) != target:
            x = F.adaptive_avg_pool3d(x, target)

        x = self.patch_embedding(x)                        # [B, dim, T/4, gh, gw]
        grid = tuple(x.shape[3:])
        x = rearrange(x, "b c gt gh gw -> b (gt gh gw) c")

        trans_features_1, score_1 = self.transformer1(x, gra_sharp, grid)
        trans_features_2, score_2 = self.transformer2(trans_features_1, gra_sharp, grid)
        trans_features_3, score_3 = self.transformer3(trans_features_2, gra_sharp, grid)

        gh, gw = grid
        features_last = rearrange(trans_features_3, "b (gt gh gw) c -> b c gt gh gw",
                                  gh=gh, gw=gw)           # [B, dim, T/4, gh, gw]

        features_last = self.upsample(features_last)       # [B, dim,   T/2, gh, gw]
        features_last = self.upsample2(features_last)      # [B, dim/2, T,   gh, gw]

        # Pool the token grid away; time survives. Two reductions rather than
        # one over both axes: in float32 the accumulation order is part of the
        # answer, and this is the order the original averaged in.
        features_last = reduce(reduce(features_last, "b c t gh gw -> b c t gw", "mean"),
                               "b c t gw -> b c t", "mean")
        rppg = self.conv_block_last(features_last)         # [B, 1, T']

        # Back to the window length if the tubes did not divide it.
        if rppg.shape[-1] != frames:
            rppg = F.interpolate(rppg, size=frames, mode="linear", align_corners=False)

        return rppg, score_1, score_2, score_3
