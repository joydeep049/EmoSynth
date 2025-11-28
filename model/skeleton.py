# Copyright (C) 2025 Joydeep Tripathy
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image
import os

class VariationalAutoEncoder(nn.Module):
  def __init__(self, latent_dim: int):
    super().__init__()
    #encoder
    self.fc_mu = nn.Linear(128, latent_dim)
    self.fc_logvar = nn.Linear(128, latent_dim)

    self.decoder = nn.Sequential(
      nn.Linear(latent_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 500),
      nn.ReLU(),
      nn.Linear(500, 64*32*32), # (500,) -> (64*32*32,)
      nn.Unflatten(1, (64,32,32)), # (64*32*32,) -> (64, 32, 32)
      nn.ReLU(),

      nn.ConvTranspose2d(64,32, stride = 2, padding = 1, kernel_size = 4), #(64, 32, 32) -> (32, 64, 64)
      nn.ConvTranspose2d(32, 3, stride = 2, padding = 1, kernel_size = 4), #(32, 64, 64) -> (3, 128, 128)
      nn.Sigmoid(),
    )
    # Encoder convolutional layer
    self.convolutions = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),

      nn.MaxPool2d(kernel_size=2, stride=2), #(32, 128 , 128) -> (32, 64, 64)

      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), #(32, 64, 64) -> (64, 64, 64)
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2), #(64, 64, 64) -> (64, 32, 32)
      nn.Flatten(), # (64*32*32,)
      nn.Linear(64*32*32, 500), # (64*32*32,) -> (500,)
      nn.ReLU(),
      nn.Linear(500, 128), # (500,) -> (128,)
      nn.ReLU(),
    )

  def encode(self,x):
    x = self.convolutions(x)

    mu = self.fc_mu(x)
    logvar = self.fc_logvar(x)

    return mu, logvar

  def reparametrize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    return (mu + epsilon * std)

  # def decoder(self,z):
  #   return self.decode(z)

  def forward(self,x):
    mu,logvar = self.encode(x)
    z = self.reparametrize(mu,logvar)
    x_hat = self.decoder(z)

    return x_hat, mu, logvar


class EmojiDataset(Dataset):
    def __init__(self, root, small_sample = False):
        self.root = root
        self.small_sample = small_sample
        if not self.small_sample:
            self.files = [f for f in os.listdir(root) if f.endswith(".png")]
        else:
            i = 0
            self.files = []
            for f in os.listdir(root):
                if(f.endswith(".png") and i < 100):
                    self.files.append(f)
                    i += 1
                if i >= 100:
                    break

        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),               #converts to [0,1]
            # T.Normalize([0.5], [0.5])   
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])
        img = Image.open(image_path).convert("RGBA")   
        img = img.convert("RGB")
        img = self.transform(img)
        return img