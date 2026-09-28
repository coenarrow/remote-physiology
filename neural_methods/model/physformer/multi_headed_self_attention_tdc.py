"""PhysFormer's attention: temporal-difference convolutions as Q/K projections."""

from einops import einsum, rearrange
from torch import nn
from torch.nn import functional as F

from neural_methods.model._shared_modules.cdc_t import CDCT


class MultiHeadedSelfAttentionTDC(nn.Module):
    """Multi-headed dot-product attention whose Q/K projections are 3-D CDCs.

    The token sequence is folded back into its ``(gt, gh, gw)`` tube grid so the
    projections can be depth-wise 3-D convolutions over space *and* time —
    which is what makes the attention temporal-difference aware. The grid is
    a forward-time argument: the projections are convolutions, so nothing in
    the parameters depends on it.
    """

    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
        self.proj_q = nn.Sequential(
            CDCT(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDCT(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )

        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None  # for visualization

    def forward(self, x, gra_sharp, grid):
        """``(B, gt*gh*gw, dim)`` in, the same shape out (plus the score map)."""
        gh, gw = grid
        x = rearrange(x, "b (gt gh gw) c -> b c gt gh gw", gh=gh, gw=gw)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (
            rearrange(t, "b (nh dh) gt gh gw -> b nh (gt gh gw) dh", nh=self.n_heads)
            for t in (q, k, v)
        )

        # ``gra_sharp`` replaces the usual sqrt(d_k): a tunable softmax
        # temperature, which is the "gradient sharpness" of the paper's title.
        scores = einsum(q, k, "b nh s dh, b nh u dh -> b nh s u") / gra_sharp
        scores = self.drop(F.softmax(scores, dim=-1))

        h = einsum(scores, v, "b nh s u, b nh u dh -> b nh s dh")
        h = rearrange(h, "b nh s dh -> b s (nh dh)")
        self.scores = scores
        return h, scores
