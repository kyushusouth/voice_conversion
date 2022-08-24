from turtle import forward
import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        out1,  out2 = torch.split(out, out.shape[1] // 2, dim=1)
        out = out1 * torch.sigmoid(out2)
        return out


class GLU2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        out1, out2 = torch.split(out, out.shape[1] // 2, dim=1)
        out = out1 * torch.sigmoid(out2)
        return out