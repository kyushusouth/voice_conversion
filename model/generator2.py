"""
cyclegan-vc2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from glu import GLU, GLU2D


class FirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            GLU2D(out_channels, out_channels, kernel_size),
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
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size, padding=padding),
            # nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(out_channels),
            GLU2D(out_channels, out_channels, kernel_size),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class Generator(nn.Module):
    def __init__(self, in_channels, feat_channels):
        super().__init__()
        cs = [128, 256, 512, 1024]

        self.first_layers = nn.Sequential(
            nn.Conv1d(in_channels, cs[0], kernel_size=(5, 15), padding=(2, 7)),
            GLU2D(cs[0], cs[0], kernel_size=(5, 15)),
        )

        self.first_blocks = nn.Sequential(
            FirstBlock(cs[0], cs[1], kernel_size=(5, 5), stride=(2, 2)),
            FirstBlock(cs[1], cs[2], kernel_size=(5, 5), stride=(2, 2)),
        )

        self.dim_change_layer = nn.Sequential(
            nn.Conv1d(cs[2] * feat_channels // 4, cs[1], kernel_size=1),
            nn.InstanceNorm1d(cs[1]),
        )

        self.second_blocks = nn.Sequential(
            SecondBlock(cs[1], cs[2], kernel_size=3),
            SecondBlock(cs[1], cs[2], kernel_size=3),
            SecondBlock(cs[1], cs[2], kernel_size=3),
            SecondBlock(cs[1], cs[2], kernel_size=3),
            SecondBlock(cs[1], cs[2], kernel_size=3),
            SecondBlock(cs[1], cs[2], kernel_size=3),
        )

        self.dim_change_layer2 = nn.Sequential(
            nn.Conv1d(cs[1], cs[2] * feat_channels // 4, kernel_size=1),
            nn.InstanceNorm1d(cs[2] * feat_channels // 4),
        )

        self.third_blocks = nn.Sequential(
            ThirdBlock(cs[2], cs[1], kernel_size=(5, 5)),
            ThirdBlock(cs[1], cs[0], kernel_size=(5, 5)),
        )

        self.last_conv = nn.Conv2d(cs[0], in_channels, kernel_size=(5, 15), padding=(2, 7))

    def forward(self, x):
        """
        x : (B, C, T)
        """
        B, C, T = x.shape
        x = x.unsqueeze(1)  # (B, 1, C, T)
        out = self.first_layers(x)
        # print(f"out = {out.shape}")
        out = self.first_blocks(out)
        # print(f"out = {out.shape}")

        out = out.reshape(B, -1, T // 4)    # (B, C, T)
        # print(f"out = {out.shape}")
        out = self.dim_change_layer(out)
        # print(f"out = {out.shape}")

        out = self.second_blocks(out)
        # print(f"out = {out.shape}")

        out = self.dim_change_layer2(out)
        # print(f"out = {out.shape}")
        out = out.reshape(B, -1, C // 4, T // 4)
        # print(f"out = {out.shape}")

        out = self.third_blocks(out)
        # print(f"out = {out.shape}")
        out = self.last_conv(out)
        # print(f"out = {out.shape}")
        out = out.squeeze(1)    # (B, C, T)
        return out


if __name__ == "__main__":
    x = torch.rand(1, 80, 300)
    net = Generator(in_channels=1, feat_channels=80)
    out = net(x)
    breakpoint()