"""
cyclegan-vc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from glu import GLU


class FirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm1d(out_channels),
            GLU(out_channels, out_channels, kernel_size),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class SecondBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, inner_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm1d(inner_channels),
            GLU(inner_channels, inner_channels, kernel_size),
            nn.Conv1d(inner_channels, in_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm1d(in_channels),
        )

    def forward(self, x):
        res = x
        out = self.layers(x)
        out += res
        return out


class ThirdBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2, stride=2),
            # nn.PixelShuffle(2),   # (B, C, H, W)じゃないと使えないのでやめた
            nn.InstanceNorm1d(out_channels),
            GLU(out_channels, out_channels, kernel_size),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        cs = [128, 256, 512, 1024]

        self.first_layers = nn.Sequential(
            nn.Conv1d(in_channels, cs[0], kernel_size=15, padding=7),
            GLU(cs[0], cs[0], kernel_size=15),
        )

        self.first_blocks = nn.Sequential(
            FirstBlock(cs[0], cs[1], kernel_size=5, stride=2),
            FirstBlock(cs[1], cs[2], kernel_size=5, stride=2),
        )

        self.second_blocks = nn.Sequential(
            SecondBlock(cs[2], cs[3], kernel_size=3),
            SecondBlock(cs[2], cs[3], kernel_size=3),
            SecondBlock(cs[2], cs[3], kernel_size=3),
            SecondBlock(cs[2], cs[3], kernel_size=3),
            SecondBlock(cs[2], cs[3], kernel_size=3),
            SecondBlock(cs[2], cs[3], kernel_size=3),
        )

        self.third_blocks = nn.Sequential(
            ThirdBlock(cs[2], cs[3], kernel_size=5),
            ThirdBlock(cs[3], cs[2], kernel_size=5),
        )

        self.last_conv = nn.Conv1d(cs[2], in_channels, kernel_size=15, padding=7)

    def forward(self, x):
        """
        x : (B, C, T)
        """
        out = self.first_layers(x)
        # print(f"out = {out.shape}")
        out = self.first_blocks(out)
        # print(f"out = {out.shape}")
        out = self.second_blocks(out)
        # print(f"out = {out.shape}")
        out = self.third_blocks(out)
        # print(f"out = {out.shape}")
        out = self.last_conv(out)
        # print(f"out = {out.shape}")
        return out


if __name__ == "__main__":
    x = torch.rand(1, 80, 300)
    net = Generator(in_channels=80)
    out = net(x)
    breakpoint()