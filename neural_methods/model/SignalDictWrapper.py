"""Shared dict boundary: raw backbone tensors in, {signal: (B, T)} out."""
import torch.nn as nn

from neural_methods.signals import validate_traces

INPUT_MODES = ('frames2d', 'video3d')


class SignalDictWrapper(nn.Module):
    def __init__(self, backbone, traces, input_mode):
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown input_mode {input_mode!r}; known: {INPUT_MODES}")
        self.backbone = backbone
        self.traces = validate_traces(traces)
        self.input_mode = input_mode

    def forward(self, frames):
        B, C, T, H, W = frames.shape
        if self.input_mode == 'frames2d':
            x = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            out = self.backbone(x)                       # (B*T, S)
            out = out.view(B, T, -1).permute(0, 2, 1)    # (B, S, T)
        else:  # video3d
            out = self.backbone(frames)                  # (B, S, T)
        return {sig: out[:, i, :] for i, sig in enumerate(self.traces)}
