import torch
import torch.nn as nn
from torch import cat

from .DSConv import DCN_Conv

class Discriminator(torch.nn.Module):
    def __init__(self, device, channels=1, dim=128):
        super().__init__()
        self.pre_module = nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.pool = nn.MaxPool3d(2, stride=2)

        self.dsc0 = DCN_Conv(channels, 16, 3, 1.0, 0, True, device)
        self.dsc1 = DCN_Conv(channels, 16, 3, 1.0, 1, True, device)
        self.dsc2 = DCN_Conv(channels, 16, 3, 1.0, 2, True, device)

        self.main_module = nn.Sequential(
            nn.Conv3d(in_channels=112, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),        

            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.output = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Tanh())


    def forward(self, x):
        xx = self.pre_module(x)
        x = self.pool(x)
        x0 = self.dsc0(x)
        x1 = self.dsc1(x)
        x2 = self.dsc2(x)
        x = cat([xx, x0, x1, x2], dim=1)

        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)