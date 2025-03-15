# This file is for VAE model class

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
import copy
import time

class VAE(nn.Module):

    def __init__(self, image_width, image_height, hidden_dim, latent_dim, device, optimizer):
        super(VAE, self).__init__()

        self.device = device
        self.image_width = image_width
        self.image_height = image_height
        self.input_dim = image_width * image_height * 3 # RGB-channel
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

        # the return value should be reults and probability of the each of the result
        # don't forget the rgb-channel

    def loss(x, x_hat, mean, logvar):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ logvar - mean.pow(2) - logvar.exp())
        return reproduction_loss + KLD
    
    # Function to pre-calculating the output into latent space in order to calculate the likelihood distribution for random point (the below function)
    # input: Dataset

    # Encoding Dataset into latent space has to be implemented
    # input: Dataset
    # output: List of point/(mean,variance) which one is easier to calculate the likelihood distribution for random point

    # Training function has to be implemented
    # input: Dataset, epochs
    # output: void (maybe loading bar in terminal)
    def train_model(self, dataloaders, epochs):
        since = time.time()

        test_loss_history = []
        train_loss_history = []

        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = float('inf')

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))

            for phase in ['train', 'test']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0

                for inputs, _ in dataloaders[phase]:
                    inputs = inputs.view(-1, self.input_dim).to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, mean, logvar = self.forward(inputs)
                        loss = self.loss(inputs, outputs, mean, logvar)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                print('{} Average Loss: {:.4f}'.format(phase, epoch_loss))

                if phase == 'test':
                    test_loss_history.append(epoch_loss)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(self.state_dict())
                if phase == 'train':
                    train_loss_history.append(epoch_loss)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test loss: {:4f}'.format(best_loss))

        history_dict = {'test_loss': test_loss_history, 'train_loss': train_loss_history}
        return best_model_wts, history_dict