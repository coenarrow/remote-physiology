"""Bi-level routing attention helpers of RhythmFormer.

Adapted from https://github.com/rayleizhu/BiFormer.
"""

import torch
from einops import rearrange, repeat
from torch import Tensor


def _grid2seq(x: Tensor, region_size: tuple, num_heads: int):
    """``(B, C, T, H, W)`` -> ``(B, nhead, nregion, region_volume, head_dim)``.

    Also returns the number of regions per t / row / column.
    """
    region_t, region_h, region_w = (n // r for n, r in zip(x.shape[2:], region_size))
    x = rearrange(x, "b (m d) (rt o) (rh p) (rw q) -> b m (rt rh rw) (o p q) d",
                  m=num_heads, o=region_size[0], p=region_size[1], q=region_size[2])
    return x, region_t, region_h, region_w


def _seq2grid(x: Tensor, region_t: int, region_h: int, region_w: int,
              region_size: tuple) -> Tensor:
    """``(B, nhead, nregion, region_volume, head_dim)`` -> ``(B, C, T, H, W)``."""
    return rearrange(x, "b m (rt rh rw) (o p q) d -> b (m d) (rt o) (rh p) (rw q)",
                     rt=region_t, rh=region_h, rw=region_w,
                     o=region_size[0], p=region_size[1], q=region_size[2])


def video_regional_routing_attention_torch(
        query: Tensor, key: Tensor, value: Tensor, scale: float,
        region_graph: Tensor, region_size: tuple,
        kv_region_size: tuple = None):
    """Token-to-token attention inside the routed regions.

    Args:
      query, key, value: ``(B, C, T, H, W)`` tensors.
      scale: the scale/temperature of the dot product.
      region_graph: ``(B, nhead, q_nregion, topk)``, the routed key regions.
      region_size: ``(rt, rh, rw)``, the region size for the queries.
      kv_region_size: the same for keys and values; defaults to ``region_size``.

    Returns the attended ``(B, C, T, H, W)`` tensor and the attention matrix.
    """
    kv_region_size = kv_region_size or region_size
    num_heads, q_nregion = region_graph.shape[1:3]

    # To sequence format, i.e. (bs, nhead, nregion, region_volume, head_dim).
    query, region_t, region_h, region_w = _grid2seq(query, region_size, num_heads)
    key, _, _, _ = _grid2seq(key, kv_region_size, num_heads)
    value, _, _, _ = _grid2seq(value, kv_region_size, num_heads)

    # Gather the routed keys and values. torch.gather does not broadcast, so
    # the graph is expanded to the shape it indexes into.
    kv_volume, head_dim = key.shape[3:]
    graph = repeat(region_graph, "b m s k -> b m s k r d", r=kv_volume, d=head_dim)
    key_g = torch.gather(repeat(key, "b m u r d -> b m s u r d", s=q_nregion),
                         dim=3, index=graph)
    value_g = torch.gather(repeat(value, "b m u r d -> b m s u r d", s=q_nregion),
                           dim=3, index=graph)

    # (bs, nhead, q_nregion, reg_volume, head_dim) against the topk*kv_volume
    # tokens each query region routed to.
    attn = (query * scale) @ rearrange(key_g, "b m s k r d -> b m s d (k r)")
    attn = torch.softmax(attn, dim=-1)
    output = attn @ rearrange(value_g, "b m s k r d -> b m s (k r) d")

    return _seq2grid(output, region_t, region_h, region_w, region_size), attn
