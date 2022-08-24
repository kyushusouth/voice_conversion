import torch
import torch.nn as nn
import torch.nn.functional as F
from glu import GLU2D


class DownSampleBlock(nn.Module):
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

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        cs = [128, 256, 512, 1024]
        self.first_layers = nn.Sequential(
            nn.Conv2d(in_channels, cs[0], kernel_size=(3, 3), padding=(1, 1)),
            GLU2D(cs[0], cs[0], kernel_size=(3, 3)),
        )

        self.down_sample_layers = nn.Sequential(
            DownSampleBlock(cs[0], cs[1], kernel_size=(3, 3), stride=(2, 2)),
            DownSampleBlock(cs[1], cs[2], kernel_size=(3, 3), stride=(2, 2)),
            DownSampleBlock(cs[2], cs[3], kernel_size=(3, 3), stride=(2, 2)),
            DownSampleBlock(cs[3], cs[3], kernel_size=(1, 5), stride=(1, 1)),
        )

        self.last_layer = nn.Conv2d(cs[3], 1, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        """
        x : (B, C, T)
        """
        x = x.unsqueeze(1)  # (B, 1, C, T)
        out = self.first_layers(x)
        out = self.down_sample_layers(out)
        out = self.last_layer(out)
        return out


if __name__ == "__main__":
    x = torch.rand(1, 80, 300)
    net = Discriminator(in_channels=1)
    out = net(x)
    breakpoint()