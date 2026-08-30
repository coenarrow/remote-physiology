import torch
import torch.nn as nn

# batches=2
# in_channels = 4
# out_channels = 16
# frames=128
# height=100
# width=100
# kernel_size=[1, 5, 5]
# stride=1
# padding=[0, 2, 2]



# x = torch.rand((batches,in_channels,frames,height,width))

# def conv_block(in_channels, out_channels, kernel_size, stride, padding): 
#     layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)] 
#     layers.append(nn.BatchNorm3d(out_channels)) 
#     layers.append(nn.ReLU(inplace=True))
#     return nn.Sequential(*layers)

# ConvBlock1 = conv_block(4, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])
# MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
# ConvBlock2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
# ConvBlock3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
# ConvBlock4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
# ConvBlock5 = conv_block(64, 32, [2, 1, 1], stride=[2, 1, 1], padding=0)

# print(f"{x.shape}: Input")
# x = ConvBlock1(x)
# print(f"{x.shape}: ConvBlock1")
# x = MaxpoolSpa(x)
# print(f"{x.shape}: MaxPoolSpa")
# x = ConvBlock2(x)
# print(f"{x.shape}: ConvBlock2")
# x = ConvBlock3(x)
# print(f"{x.shape}: ConvBlock3")
# x = MaxpoolSpa(x)
# print(f"{x.shape}: MaxPoolSpa")

# s_x = ConvBlock4(x) # Slow stream 
# print(f"{s_x.shape}: Slow Stream")


# f_x = ConvBlock5(x) # Fast stream 
# print(f"{f_x.shape}: Fast Stream")



import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

N, C, T, H, W = 1, 1, 3, 64, 64
x = torch.zeros((N, C, T, H, W))

# Region A: static bright block
x[:, :, 0, 10:20, 10:20] = 1.0
x[:, :, 1, 10:20, 10:20] = 1.0
x[:, :, 2, 10:20, 10:20] = 1.0

# Region B: blinking block
x[:, :, 0, 10:20, 40:50] = 0.0
x[:, :, 1, 10:20, 40:50] = 0.2
x[:, :, 2, 10:20, 40:50] = 0.0


class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.2):
        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kh, kw] = self.conv.weight.shape

            if t > 1:
                kernel_diff = (
                    self.conv.weight[:, :, 0, :, :].sum(2).sum(2) +
                    self.conv.weight[:, :, 2, :, :].sum(2).sum(2)
                )
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(
                    input=x,
                    weight=kernel_diff,
                    bias=self.conv.bias,
                    stride=self.conv.stride,
                    padding=0,
                    dilation=self.conv.dilation,
                    groups=self.conv.groups
                )
                return out_normal - self.theta * out_diff
            else:
                return out_normal

model = CDC_T(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False, theta=0.5)

with torch.no_grad():
    model.conv.weight.zero_()

with torch.no_grad():
    w = model.conv.weight  # shape: (1, 1, 3, 3, 3)
    # temporal index 0, center (1,1)
    w[0, 0, 0, 1, 1] = 1.0
    # temporal index 1, center (1,1)
    w[0, 0, 1, 1, 1] = 2.0
    # temporal index 2, center (1,1)
    w[0, 0, 2, 1, 1] = 3.0

with torch.no_grad():
   out_conv3d = model.conv(x)   # shape: (1, 1, 3, 64, 64)
   out_cdc    = model(x)        # same shape

inp = x[0, 0].cpu().numpy()           # (T, H, W)
conv = out_conv3d[0, 0].cpu().numpy() # (T, H, W)
cdc  = out_cdc[0, 0].cpu().numpy()    # (T, H, W)


titles_row1 = ['Input t=0', 'Input t=1', 'Input t=2']
titles_row2 = ['Conv3d t=0', 'Conv3d t=1', 'Conv3d t=2']
titles_row3 = ['CDC_T t=0', 'CDC_T t=1', 'CDC_T t=2']

fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for t in range(3):
    # Row 1: inputs
    ax = axes[0, t]
    im = ax.imshow(inp[t], origin='lower')
    ax.set_title(titles_row1[t])
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # Row 2: standard conv
    ax = axes[1, t]
    im = ax.imshow(conv[t], origin='lower')
    ax.set_title(titles_row2[t])
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # Row 3: CDC_T
    ax = axes[2, t]
    im = ax.imshow(cdc[t], origin='lower')
    ax.set_title(titles_row3[t])
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
plt.tight_layout()
plt.show()

rA, cA = 15, 15
rB, cB = 15, 45

# shape (T,)
conv_A = conv[:, rA, cA]
conv_B = conv[:, rB, cB]
cdc_A  = cdc[:, rA, cA]
cdc_B  = cdc[:, rB, cB]

t_axis = [0, 1, 2]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Static region A
axes[0].plot(t_axis, conv_A, marker='o', label='Conv3d')
axes[0].plot(t_axis, cdc_A, marker='x', label='CDC_T')
axes[0].set_title('Static region A')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Activation')
axes[0].legend()

# Blinking region B
axes[1].plot(t_axis, conv_B, marker='o', label='Conv3d')
axes[1].plot(t_axis, cdc_B, marker='x', label='CDC_T')
axes[1].set_title('Blinking region B')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Activation')
axes[1].legend()

plt.tight_layout()
plt.show()
