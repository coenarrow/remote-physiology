"""The head of FactorizePhys: voxel embeddings to one trace, through FSAM."""

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from neural_methods.model.factorizephys.features_factorization_module import FeaturesFactorizationModule
from neural_methods.model._shared_modules.conv_block_3d import ConvBlock3D


class BVPHead(nn.Module):
    """Voxel embeddings in, one trace out, factorized attention in between."""

    def __init__(self, filters, head_spatial, dropout_rate=0.1, use_fsam=True, residual=True):
        """``filters`` are the three stages' filter counts and ``head_spatial`` the
        feature-map side the valid convolutions take down to a point; both are
        FactorizePhys's, passed in."""
        super().__init__()
        self.head_spatial = head_spatial
        self.use_fsam = use_fsam
        self.residual = residual

        self.conv_block = nn.Sequential(
            ConvBlock3D(filters[2], filters[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], bias=False), #B, filters[2], 160, 11, 11
            ConvBlock3D(filters[2], filters[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], bias=False), #B, filters[2], 160, 9, 9
            ConvBlock3D(filters[2], filters[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], bias=False), #B, filters[2], 160, 7, 7
            nn.Dropout3d(p=dropout_rate),
        )

        if self.use_fsam:
            self.fsam = FeaturesFactorizationModule(filters[2], align_channels=filters[2] // 2)
            self.fsam_norm = nn.InstanceNorm3d(filters[2])
            self.bias1 = nn.Parameter(torch.tensor(1.0))

        self.final_layer = nn.Sequential(
            ConvBlock3D(filters[2], filters[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], bias=False),            #B, filters[1], 160, 5, 5
            ConvBlock3D(filters[1], filters[0], [3, 3, 3], [1, 1, 1], [1, 0, 0], bias=False),            #B, filters[0], 160, 3, 3
            nn.Conv3d(filters[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=True),    #B, 1, 160, 1, 1
        )

    def output_layers(self):
        """The activation-free readout: the final layer's 1-plane conv."""
        return (self.final_layer[-1],)

    def forward(self, voxel_embeddings):
        """``(B, filters[2], T, H', W')`` -> ``(B, 1, T)``."""
        # Any frame size: the convolutions below are all valid and take
        # head_spatial down to a point, so whatever the extractor hands over
        # is resampled to that. At the paper's 72x72 frames it is already 13x13
        # and this is skipped.
        frames = voxel_embeddings.shape[2]
        if tuple(voxel_embeddings.shape[3:]) != (self.head_spatial, self.head_spatial):
            voxel_embeddings = F.adaptive_avg_pool3d(
                voxel_embeddings, (frames, self.head_spatial, self.head_spatial))

        voxel_embeddings = self.conv_block(voxel_embeddings)

        if self.use_fsam:
            # NMF factorises a non-negative matrix, so the embeddings are
            # shifted before the mask is taken, and again before it is applied.
            att_mask = self.fsam(voxel_embeddings - voxel_embeddings.min())
            x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1,
                          att_mask - att_mask.min() + self.bias1)
            factorized_embeddings = self.fsam_norm(x)
            if self.residual:
                factorized_embeddings = voxel_embeddings + factorized_embeddings
            x = self.final_layer(factorized_embeddings)
        else:
            x = self.final_layer(voxel_embeddings)

        return rearrange(x, "b 1 t 1 1 -> b 1 t")
