import argparse
from tqdm import tqdm
import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from model.discriminator import AnimeGANDiscriminator
from model.generator import AnimeGANv2Generator
from model.AnimeGANv2 import AnimeGANv2_LightningSystem
from dataset.dataset import AnimeGANDataset, TransformsModule
from config import config

def build_model():
    gen = AnimeGANv2Generator()
    disc = AnimeGANDiscriminator()
    transforms = TransformsModule()

    loader = AnimeGANDataset(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        transform=transforms
    )
    AnimeGAN = AnimeGANv2_LightningSystem(
        gen=gen,
        disc=disc,
        loader=loader
    )

    return AnimeGAN, loader

def train():
    model, loader = build_model()
    logger = TensorBoardLogger(
        "tb_logs",
    )
    trainer = pl.Trainer(
        min_epochs=1, 
        max_epochs=config.NUM_EPOCHS,
        logger=logger,
        gpus=[0, 1],
        distributed_backend='dp'
    )
    trainer.fit(model, loader)

if __name__=="__main__":
    train()