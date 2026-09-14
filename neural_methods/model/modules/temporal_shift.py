import torch
from torch import nn

class TSM(nn.Module):
    """Temporal shift over segments of ``frame_depth`` consecutive frames of
    the same clip.

    Operates on ``(B, T, C, H, W)``, one clip per row of ``B``. Each clip's
    ``T`` frames are cut into chunks of ``frame_depth``, and every chunk is
    shifted independently, so the shift never crosses a clip boundary. A
    trailing partial chunk (``T`` not a multiple of ``frame_depth``) is
    shifted as its own shorter segment — the published shift already
    zero-pads at segment ends, so a short segment is well defined. At
    ``T % frame_depth == 0`` this computes exactly the published,
    single-chunk-size shift.

    ``wrap`` says what happens at a segment's ends. ``False`` (TS-CAN's
    shift) zero-pads them; ``True`` wraps the frame shifted out of one end
    round to the other end of the same segment, which is BigSmall's
    published WTSM (a shorter trailing segment wraps within itself).
    """

    def __init__(self, frame_depth: int = 20, fold_div: int = 3,
                 wrap: bool = False):
        super().__init__()
        self.frame_depth = frame_depth
        self.fold_div = fold_div
        self.wrap = wrap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, c, _, _ = x.shape
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        for start in range(0, t, self.frame_depth):
            end = min(start + self.frame_depth, t)
            segment = x[:, start:end]
            shifted = torch.zeros_like(segment)
            shifted[:, :-1, :fold] = segment[:, 1:, :fold]                  # shift left
            shifted[:, 1:, fold:2 * fold] = segment[:, :-1, fold:2 * fold]  # shift right
            shifted[:, :, 2 * fold:] = segment[:, :, 2 * fold:]             # not shifted
            if self.wrap:
                shifted[:, -1, :fold] = segment[:, 0, :fold]                # wrap left
                shifted[:, 0, fold:2 * fold] = segment[:, -1, fold:2 * fold]  # wrap right
            out[:, start:end] = shifted
        return out
