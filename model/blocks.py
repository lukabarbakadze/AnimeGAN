import torch
from torch import nn
import torch.nn.functional as F

#-----------------------------------------------------------------------------------
class ConvLNormLRelu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pad_mode="reflect",
                 last=False,
                 generator=False):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=pad_mode),
            nn.GroupNorm(num_groups=1, num_channels=out_channels) if generator else nn.Identity(),
            nn.LeakyReLU(inplace=True) if last else nn.Tanh()
        )

    def forward(self, x):
        return self.block(x)

#-----------------------------------------------------------------------------------
class InvResBlock(nn.Module):
    def __init__(self, in_channels, exp_ratio=2, res_connection=False):
        super().__init__()

        self.res_connection = res_connection
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * exp_ratio, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels * exp_ratio, in_channels * exp_ratio, 
                      groups=in_channels * exp_ratio, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.Conv2d(in_channels * exp_ratio, in_channels, kernel_size=1)
        )
    
    def forward(self, x):
        res = x
        x = self.block(x)
        if self.res_connection:
            x += res
        return x

#-----------------------------------------------------------------------------------
class ConvINormLRelu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 pad_mode="reflect"):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=pad_mode),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)