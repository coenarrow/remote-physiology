"""One of PhysFormer's three transformer stages: a stack of blocks."""

from torch import nn

from neural_methods.model.physformer.transformer_block_tdc import TransformerBlockTDC


class TransformerTDC(nn.Module):
    """One of the three transformer stages."""

    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlockTDC(dim, num_heads, ff_dim, dropout, theta)
            for _ in range(num_layers)
        ])

    def forward(self, x, gra_sharp, grid):
        for block in self.blocks:
            x, score = block(x, gra_sharp, grid)
        return x, score
