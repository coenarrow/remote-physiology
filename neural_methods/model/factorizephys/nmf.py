"""The matrix decomposition inside FSAM: the ``MD_TYPE: NMF`` path, with bases
initialised afresh on every forward (``RAND_INIT``). See
:mod:`neural_methods.model.factorizephys.features_factorization_module`.

All reshaping is einops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class NMF(nn.Module):
    """Non-negative matrix factorisation of a voxel embedding, used as attention.

    ``(B, C, T, H, W)`` in, the same shape out. The clip is flattened and cut
    into ``splits`` matrices of ``depth = T // splits`` rows, factorised into
    ``rank`` bases and coefficients by multiplicative updates, and multiplied
    back. The cut is the published one — the contiguous ``(C, T, H, W)``
    buffer read as ``(splits, depth, N)``, which mixes the channel and time
    axes rather than slicing time cleanly. It is what the paper's code does
    and what its checkpoints were factorised with, so it is kept exactly,
    written as the flatten-and-regroup it is.

    The module holds no parameters: the bases start from a normalised
    constant on every forward.
    """

    def __init__(self, rank: int = 1, splits: int = 1, steps: int = 3):
        super().__init__()
        self.rank = rank
        self.splits = splits
        self.steps = steps

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        """One multiplicative update of both factors of ``x ~ bases @ coef^T``."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(rearrange(x, "b d n -> b n d"), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(torch.bmm(rearrange(bases, "b d r -> b r d"), bases))
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(torch.bmm(rearrange(coef, "b n r -> b r n"), coef))
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    @torch.no_grad()
    def local_inference(self, x, bases):
        """``steps`` updates from a softmax initialisation of the coefficients."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(rearrange(x, "b d n -> b n d"), bases)
        coef = F.softmax(coef, dim=-1)
        for _ in range(self.steps):
            bases, coef = self.local_step(x, bases, coef)
        return bases, coef

    def compute_coef(self, x, bases, coef):
        """The one update that carries a gradient back into the embedding."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(rearrange(x, "b d n -> b n d"), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(torch.bmm(rearrange(bases, "b d r -> b r d"), bases))
        return coef * numerator / (denominator + 1e-6)

    def forward(self, x):
        """``(B, C, T, H, W)`` -> ``(B, C, T, H, W)``, any T, H and W."""
        channels, frames, height = x.shape[1], x.shape[2], x.shape[3]
        depth = frames // self.splits

        # (B, C, T, H, W) -> (B * S, D, N): the published regrouping.
        flat = rearrange(x, "b c t h w -> b (c t h w)")
        matrix = rearrange(flat, "b (s d n) -> (b s) d n", s=self.splits, d=depth)

        bases = torch.ones(matrix.shape[0], depth, self.rank,
                           device=matrix.device, dtype=matrix.dtype)
        bases = F.normalize(bases, dim=1)

        bases, coef = self.local_inference(matrix, bases)
        coef = self.compute_coef(matrix, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        matrix = torch.bmm(bases, rearrange(coef, "b n r -> b r n"))

        flat = rearrange(matrix, "(b s) d n -> b (s d n)", s=self.splits)
        return rearrange(flat, "b (c t h w) -> b c t h w",
                         c=channels, t=frames, h=height)
