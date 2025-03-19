# This file is for VAE model class

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset
import copy
import time

class VAE(nn.Module):

    def __init__(self, image_width, image_height, hidden_dim, latent_dim, device, optimizer, class_num):
        super(VAE, self).__init__()

        self.class_num = class_num
        self.device = device
        self.image_width = image_width
        self.image_height = image_height
        self.input_dim = image_width * image_height * 3 # RGB-channel
        self.optimizer = optimizer
        self.gamma = 1e-5

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
    
    def loss(x, x_hat, mean, logvar):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ logvar - mean.pow(2) - logvar.exp())
        return reproduction_loss + KLD

    # image: [number_of_images, 3, image_width, image_height], probability: [number_of_images, class_num]
    def hybrid_image(self, number_of_images):
        random_points = torch.rand_like(number_of_images, 2).to(self.device)
        probability = self.probability(random_points)

        self.eval()
        with torch.no_grad():
            images = self.decoder(random_points)
        self.train()

        images = images.view(-1, 3, self.image_width, self.image_height)
        dataloaders = DataLoader(TensorDataset(images, probability), shuffle=True)
        return dataloaders
    
    # Function to pre-calculating the output into latent space in order to calculate the likelihood distribution for random point (the below function)
    # input: Dataset
    def calculate_class_mu_var(self, dataset):
        self.eval()
        self.class_lists_mu = [[] for _ in range(self.class_num)]
        self.class_lists_var = [[] for _ in range(self.class_num)]
        self.class_mu = [None] * self.class_num
        self.class_var = [None] * self.class_num

        for _, (images, labels) in enumerate(dataset):
            for (image, label) in zip(images, labels):
                with torch.no_grad():
                    mu, logvar = self.encode(image.view(-1, self.input_dim).to(self.device)).squeeze(0)
                var = torch.exp(logvar)
                self.class_lists_mu[label.item()].append(mu)
                self.class_lists_var[label.item()].append(var)
        
        for idx in range(self.class_num):

            mu = self.class_lists_mu[idx].sum() / len(self.class_lists_mu[idx])

            var = 0.0
            for i in range(len(self.class_lists_var[idx])):
                var += self.class_lists_var[idx][i] + (self.class_lists_mu[idx][i] - mu).pow(2)
            var = var / len(self.class_lists_var[idx])

            self.class_mu[idx] = mu
            self.class_var[idx] = var


    # P(Normal(mu,var) = x)
    def probs(self, x, mu, var):
        normalized_x = (x - mu) / var
        return torch.special.ndtr(normalized_x + self.gamma) - torch.special.ndtr(normalized_x - self.gamma)


    # input: Random points in latent space -> [batch_size, 2, 2]
    # output: probability of the random points -> [batch_size, class_num(=100)]
    def probability(self, random_points):
        probs = []
        for _, (x, y) in random_points:
            sum = 0.0
            class_prob = []
            for c in range(self.class_num):
                mu_x = self.class_mu[c][0]
                mu_y = self.class_mu[c][1]
                var_x = self.class_var[c][0]
                var_y = self.class_var[c][1]

                # calculate p(z|c) = p((x,y)|c) = p(x|c) p(y|c)
                prob_x = self.probs(x, mu_x, var_x)
                prob_y = self.probs(y, mu_y, var_y)
                class_prob.append(prob_x * prob_y)
                sum += prob_x * prob_y

            class_prob_normalized = [p / sum for p in class_prob]
            probs.append(class_prob_normalized)

        return torch.tensor(probs)
        
    

    def train_model(self, dataloaders, epochs):
        self.train()
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