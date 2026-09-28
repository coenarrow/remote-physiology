"""One pre-norm transformer block of PhysFormer."""

from torch import nn

from neural_methods.model.physformer.multi_headed_self_attention_tdc import (
    MultiHeadedSelfAttentionTDC,
)
from neural_methods.model.physformer.position_wise_feed_forward_st import (
    PositionWiseFeedForwardST,
)


class TransformerBlockTDC(nn.Module):
    """One pre-norm transformer block."""

    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedSelfAttentionTDC(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForwardST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp, grid):
        atten, score = self.attn(self.norm1(x), gra_sharp, grid)
        h = self.drop(self.proj(atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x), grid))
        x = x + h
        return x, score
