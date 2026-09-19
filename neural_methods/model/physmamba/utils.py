"""Layer factories of PhysMamba that are not modules of their own."""

import torch.nn as nn


def conv_block(in_channels, out_channels, kernel_size, stride, padding, bn=True, activation='relu'):
    """A ``Conv3d``, optionally followed by a batch norm and a ReLU or an ELU."""
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if bn:
        layers.append(nn.BatchNorm3d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)
