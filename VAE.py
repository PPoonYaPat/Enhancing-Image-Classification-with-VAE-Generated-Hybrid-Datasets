# This file is for VAE model class

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam

class VAE(nn.Module):

    def __init__(self, image_width, image_height, hidden_dim, latent_dim, device, optimizer):
        super(VAE, self).__init__()

        self.device = device
        self.image_width = image_width
        self.image_height = image_height
        self.input_dim = image_width * image_height
        self.optimizer = optimizer

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReu(0.2)
        )

        # Latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar
    
    def reparameterization(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(device=self.device)
        z = mean + torch.exp(logvar/2)*epsilon
        return z
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def hybrid_image(self, batch_size):
        random_points = torch.rand_like(batch_size, 2).to(self.device)
        self.eval()
        with torch.no_grad():
            results = self.decoder(random_points)
        self.train()
        return results.view(batch_size, self.image_height, self.image_width)

    def loss(x, x_hat, mean, logvar):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ logvar - mean.pow(2) - logvar.exp())
        return reproduction_loss + KLD

    # Encoding Dataset into latent space has to be implemented
    # input: Dataset
    # output: List of point/(mean,variance) which one is easier to calculate the likelihood distribution for random point

    # Training function has to be implemented
    # input: Dataset, epochs
    # output: void (maybe loading bar in terminal)