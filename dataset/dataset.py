import os
from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pytorch_lightning as pl


class MainDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

        self.input_imgs  = glob(input_dir + "*.jpg")  + glob(input_dir + "*/*.jpg")
        self.target_imgs = glob(target_dir + "*.jpg") + glob(target_dir + "*/*.jpg")

        self.input_len  = len(self.input_imgs)
        self.target_len = len(self.target_imgs)

        self.length_dataset = max(self.input_len, self.target_len)


    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        input_img  = self.input_imgs[idx % self.input_len]
        target_img = self.target_imgs[idx % self.target_len]
        blurred_img = target_img.replace("Target", "GrayScale")

        input_img  = Image.open(input_img).convert("RGB")
        target_img = Image.open(target_img).convert("RGB")
        blurred_img = Image.open(blurred_img).convert("RGB")
        
        input_img  = self.transform(input_img)
        target_img = self.transform(target_img)
        blurred_img = self.transform(blurred_img)

        return input_img, target_img, blurred_img

class AnimeGANDataset(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, transform):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage):
        # done on multiple GPU
        entire_dataset = MainDataset(
            input_dir  = "images/Input/",
            target_dir = "images/Target/",
            transform = self.transform
        )
        self.train_ds = entire_dataset
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )