import torch
from torch import nn
import torch.nn.functional as F
from .blocks import ConvLNormLRelu, ConvINormLRelu

class AnimeGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            ConvLNormLRelu(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            ConvLNormLRelu(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            ConvINormLRelu(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            ConvLNormLRelu(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            ConvLNormLRelu(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvLNormLRelu(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x):
        x = self.seq(x)
        return torch.sigmoid(x)

if __name__=="__main__":
    x = torch.randn((4, 3, 256, 256))
    model = AnimeGANDiscriminator()
    out = model(x)
    print(out.shape)