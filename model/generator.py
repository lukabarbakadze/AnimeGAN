import torch
from torch import nn
import torch.nn.functional as F
from .blocks import InvResBlock, ConvLNormLRelu

class AnimeGANv2Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample = nn.Sequential(
            ConvLNormLRelu(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
            ConvLNormLRelu(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            ConvLNormLRelu(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            ConvLNormLRelu(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            InvResBlock(in_channels=256, exp_ratio=2, res_connection=False),
        )
        self.res_connection = nn.Sequential(
            InvResBlock(in_channels=256, exp_ratio=2, res_connection=False),
            *[InvResBlock(in_channels=256, exp_ratio=2, res_connection=True) for i in range(8)],
            ConvLNormLRelu(in_channels=256, out_channels=128, kernel_size=3, stride=1),
        )
        self.upsample = nn.Sequential(
            nn.Upsample((128,128), mode="bilinear", align_corners = True),
            ConvLNormLRelu(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            ConvLNormLRelu(in_channels=128, out_channels=128, kernel_size=3, stride=1),

            nn.Upsample((256,256), mode="bilinear", align_corners = True),
            ConvLNormLRelu(in_channels=128, out_channels=64, kernel_size=3, stride=1),
            ConvLNormLRelu(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            ConvLNormLRelu(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            ConvLNormLRelu(in_channels=32, out_channels=3, kernel_size=3, stride=1, last=True),
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.res_connection(x)
        x = self.upsample(x)
        return x

if __name__=="__main__":
    x = torch.randn(4, 3, 256, 256)
    model = AnimeGANv2Generator()
    out = model(x)
    print(out.shape)