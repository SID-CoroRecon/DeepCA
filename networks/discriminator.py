import torch
import torch.nn as nn
from torch import cat

try:
    from .DSConv import DCN_Conv
except:
    from DSConv import DCN_Conv

class Discriminator(torch.nn.Module):
    def __init__(self, device, channels=1):
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

    def print_parameter_counts(self):
        """Print the number of parameters in the model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print("\nDiscriminator Parameter Counts:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"Memory savings: {(non_trainable_params/total_params)*100:.2f}%")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = Discriminator(device)
    discriminator.print_parameter_counts()