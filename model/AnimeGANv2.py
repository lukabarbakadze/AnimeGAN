from datetime import datetime
from typing import Any
import torch
from torch import nn
import torch.optim as optim
from torchvision.utils import make_grid
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

class AnimeGANv2_LightningSystem(pl.LightningModule):
    def __init__(
            self,
            gen,
            disc,
            loader
        ):
        super().__init__()
        self.gen = gen
        self.disc = disc
        self.loader = loader
        
        self.mae = nn.MSELoss()

        self.automatic_optimization = False
    
    def configure_optimizers(self) -> Any:
        self.gen_opt = optim.Adam(
            self.gen.parameters(),
            lr=3e-4,
        )
        self.disc_opt = optim.Adam(
            self.disc.parameters(),
            lr=3e-4,
        )
        return self.gen_opt, self.disc_opt
    
    def training_step(self, batch, batch_idx):
        gen_opt, disc_opt = self.optimizers()

        I, T = batch
        
        ### Optimize Discriminator
        self.disc_opt.zero_grad()

        fake_T = self.gen(I)

        D_real = self.disc(T)
        D_fake = self.disc(fake_T.detach())

        D_real_loss = self.mse(D_real, torch.ones_like(D_real))
        D_fake_loss = self.mse(D_fake, torch.ones_like(D_fake))

        D_adv_loss = D_real_loss + D_fake_loss

        




        # Generate fake images using the generator
        fake_T = self.gen(I)

        # Compute discriminator outputs for real and fake images
        real_pred = self.disc(T)
        fake_pred = self.disc(fake_T.detach())  # Detach fake_T to avoid gradient flow to generator

        # Compute the adversarial loss for discriminator
        adv_loss_disc = torch.mean((real_pred - 1)**2) + torch.mean(fake_pred**2)  # Least Squares GAN loss

        # Additional grayscale adversarial loss
        gray_fake_pred = self.disc(torch.mean(fake_T, dim=1, keepdim=True))
        gray_adv_loss_disc = torch.mean(gray_fake_pred**2)  # Least Squares GAN loss

        # Calculate total discriminator loss
        disc_loss = adv_loss_disc + 0.1 * gray_adv_loss_disc  # Adjust the scaling factor as needed

        # Backpropagate and update discriminator's weights
        self.manual_backward(disc_loss)
        disc_opt.step()

        ### Optimize Generator
        self.gen_opt.zero_grad()

        # Generate fake images using the generator again
        fake_T = self.gen(I)

        # Compute discriminator output for the generated fake images
        fake_pred = self.disc(fake_T)

        # Calculate content loss Lcon (G, D)
        content_loss = torch.mean(torch.abs(self.vgg_features(T) - self.vgg_features(fake_T)))

        # Calculate grayscale style loss Lgra (G, D)
        gram_matrix_T = self.gram_matrix(self.vgg_features(T))
        gram_matrix_fake_T = self.gram_matrix(self.vgg_features(fake_T))
        grayscale_style_loss = torch.mean(torch.abs(gram_matrix_T - gram_matrix_fake_T))

        # Calculate color reconstruction loss Lcol (G, D)
        color_loss_Y = torch.mean(torch.abs(fake_T[:, 0, :, :] - T[:, 0, :, :]))
        color_loss_U = torch.mean(F.smooth_l1_loss(fake_T[:, 1, :, :], T[:, 1, :, :]))
        color_loss_V = torch.mean(F.smooth_l1_loss(fake_T[:, 2, :, :], T[:, 2, :, :]))
        color_reconstruction_loss = color_loss_Y + color_loss_U + color_loss_V

        # Calculate total generator loss
        gen_loss = (
            300 * torch.mean((fake_pred - 1)**2) +
            1.5 * content_loss +
            3 * grayscale_style_loss +
            10 * color_reconstruction_loss
        )

        # Backpropagate and update generator's weights
        self.manual_backward(gen_loss)
        gen_opt.step()

        # Logging and returning
        logs = {'gen_loss': gen_loss, 'disc_loss': disc_loss}
        self.log_dict(logs, prog_bar=True)
        return logs
    
    def forward(self, x):
        return self.gen(x)
    
    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['gen_M_state_dict'] = self.gen_M.state_dict()

    def on_load_checkpoint(self, checkpoint) -> None:
        if 'gen_M_state_dict' in checkpoint:
            self.gen_M.load_state_dict(checkpoint['gen_M_state_dict'])
    
    def forward(self, x):
        return self.gen_M(x)