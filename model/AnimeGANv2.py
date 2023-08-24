from datetime import datetime
from typing import Any
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision.utils import make_grid
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

class AnimeGANv2_LightningSystem(pl.LightningModule):
    def __init__(
            self,
            gen,
            disc,
            loader,
            w_adv=300, 
            w_gra=3, 
            w_con=1.5, 
            w_col=10
        ):
        super().__init__()
        self.gen = gen
        self.disc = disc
        self.loader = loader
        self.w_adv = w_adv
        self.w_gra = w_gra
        self.w_con = w_con
        self.w_col = w_col

        vgg19 = torchvision.models.vgg19(pretrained=True).features[:26]
        for param in vgg19.parameters():
            param.requires_grad = False 
        self.vgg = vgg19
        self.vgg.eval()  # Set to evaluation mode

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = nn.HuberLoss()

        self.automatic_optimization = False
    
    def configure_optimizers(self) -> Any:
        self.gen_opt = optim.Adam(
            self.gen.parameters(),
            lr=3e-4,
        )
        self.disc_opt = optim.Adam(
            self.disc.parameters(),
            lr=6e-4,
        )
        return self.gen_opt, self.disc_opt

    def gram_matrix(self, features):
        batch_size, num_channels, height, width = features.size()
        features = features.view(batch_size, num_channels, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram / (num_channels * height * width)
        return gram

    def rgb_to_yuv(self, images):
        r_coeff = torch.tensor(0.299)
        g_coeff = torch.tensor(0.587)
        b_coeff = torch.tensor(0.114)

        y_channel = r_coeff * images[:, 0] + g_coeff * images[:, 1] + b_coeff * images[:, 2]
        u_channel = 0.5 - (0.168736 * images[:, 0]) - (0.331264 * images[:, 1]) + (0.5 * images[:, 2])
        v_channel = 0.5 + (0.5 * images[:, 0]) - (0.418688 * images[:, 1]) - (0.081312 * images[:, 2])

        yuv_images = torch.stack((y_channel, u_channel, v_channel), dim=1)
        return yuv_images

    def training_step(self, batch, batch_idx):
        gen_opt, disc_opt = self.optimizers()

        I, T, G = batch
        
        ### Optimize Discriminator
        self.disc_opt.zero_grad()

        fake_T = self.gen(I)

        D_real = self.disc(T)
        D_fake = self.disc(fake_T.detach())

        D_real_loss = self.mse(D_real, torch.ones_like(D_real))
        D_fake_loss = self.mse(D_fake, torch.zeros_like(D_fake))

        D_adv_loss = D_real_loss + D_fake_loss

        disc_opt.zero_grad()
        self.manual_backward(D_adv_loss)
        disc_opt.step()

        ### Optimize Generator

        # Adversarial Loss
        D_fake = self.disc(fake_T)
        G_adv_loss = self.mse(D_fake, torch.ones_like(D_fake))

        # Greyscale Loss
        vgg_input_img_features = self.vgg(I)
        vgg_grey_img_features = self.vgg(G).detach()

        gram_input = self.gram_matrix(vgg_input_img_features)
        gram_target = self.gram_matrix(vgg_grey_img_features)

        G_greyscale_loss = self.mae(gram_input, gram_target)

        # Content Loss
        vgg_target_img_features = self.vgg(T).detach()
        G_content_loss = self.mae(vgg_input_img_features, vgg_target_img_features)

        # Color Reconstruction Loss
        gen_yuv = self.rgb_to_yuv(fake_T)
        target_yuv = self.rgb_to_yuv(T)

        y_loss = self.mae(gen_yuv[:, 0], target_yuv[:, 0])
        u_loss = self.huber(gen_yuv[:, 1], target_yuv[:, 1])
        v_loss = self.huber(gen_yuv[:, 2], target_yuv[:, 2])

        G_color_loss = y_loss + u_loss + v_loss

        # Full Generator Loss
        G_loss = (
            self.w_adv * G_adv_loss + 
            self.w_gra * G_greyscale_loss + 
            self.w_con * G_content_loss + 
            self.w_col * G_color_loss
        )

        gen_opt.zero_grad()
        self.manual_backward(G_loss)
        gen_opt.step()
        losses = {
            "d_loss": D_adv_loss,
            "g_loss": G_loss
        }

        self.log_dict(losses, prog_bar=True)
    
    def forward(self, x):
        return self.gen(x)