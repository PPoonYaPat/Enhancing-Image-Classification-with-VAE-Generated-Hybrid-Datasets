# This file is for VAE model class

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset
import copy
import time

class VAE(nn.Module):

    def __init__(self, image_width, image_height, in_channels, hidden_dims, latent_dim, device, learning_rate, class_num, batch_size):
        super(VAE, self).__init__()

        self.class_num = class_num
        self.device = device
        self.image_width = image_width
        self.image_height = image_height
        self.input_dim = image_width * image_height * 3 # RGB-channel -> wronggggggggg
        self.gamma = 1e-3
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.last_dim = hidden_dims[-1]

        modules_encoder = []

        for hidden_dim in hidden_dims:
            modules_encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(*modules_encoder)

        # Latent mean and variance
        self.mean_layer = nn.Linear(self.last_dim * 4, latent_dim)
        self.logvar_layer = nn.Linear(self.last_dim * 4, latent_dim)

        modules_decoder = []

        self.decoder_input = nn.Linear(latent_dim, self.last_dim * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules_decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
            modules_decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i + 1],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 1,
                                       padding=1,
                                       output_padding=0),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        # Decoder
        self.decoder = nn.Sequential(*modules_decoder)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(device=self.device)
        z = mean + torch.exp(logvar/2)*epsilon
        return z

    def decode(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, self.last_dim, 2, 2)
        x = self.decoder(x)
        return self.final_layer(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def loss(self, x, x_hat, mean, logvar):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ logvar - mean.pow(2) - logvar.exp())
        return reproduction_loss + KLD
    
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
                    inputs = inputs.to(self.device)

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
    

    # Function to generate hybrid images

    # image: [number_of_images, 3, image_width, image_height], probability: [number_of_images, class_num]
    def hybrid_image(self, number_of_images):
        random_points = torch.randn(number_of_images, self.latent_dim, device=self.device)
        probability = self.probability(random_points)

        self.eval()
        with torch.no_grad():
            images = self.decoder(random_points)
        self.train()

        images = images.view(-1, 3, self.image_width, self.image_height)
        dataloaders = DataLoader(TensorDataset(images, probability), batch_size=self.batch_size, shuffle=True)
        return dataloaders

    # Function to pre-calculating the output into latent space in order to calculate the likelihood distribution for random point (the below function)
    # input: Dataset with extended labels -> [images, labels]
    def calculate_class_mu_var(self, dataset):
        self.eval()
        self.class_lists_mu = [[] for _ in range(self.class_num)]
        self.class_lists_var = [[] for _ in range(self.class_num)]
        self.class_mu = [None] * self.class_num
        self.class_var = [None] * self.class_num

        for _, (images, labels) in enumerate(dataset):
            for (image, label) in zip(images, labels):
                class_index = torch.argmax(label).item()
                with torch.no_grad():
                    mu, logvar = self.encode(image.to(self.device))
                mu = mu.squeeze(0)
                var = torch.exp(logvar).squeeze(0)
                var = torch.clamp(var, min=1e-8)
                self.class_lists_mu[class_index].append(mu)
                self.class_lists_var[class_index].append(var)


        for idx in range(self.class_num):
            mu = 0.0
            for i in range(len(self.class_lists_mu[idx])):
                mu += self.class_lists_mu[idx][i]
            mu = mu / len(self.class_lists_mu[idx])

            var = 0.0
            for i in range(len(self.class_lists_var[idx])):
                var += self.class_lists_var[idx][i] + (self.class_lists_mu[idx][i] - mu).pow(2)
            var = var / len(self.class_lists_var[idx])

            self.class_mu[idx] = mu
            self.class_var[idx] = var


    # P(Normal(mu,var) = x)
    def probs(self, x, mu, var):
        var_temp = var.clone()
        var = torch.clamp(var, min=1e-8)
        if var_temp != var:
            print('variance from probs function is clamped')
        normalized_x = (x - mu) / var
        return torch.special.ndtr(normalized_x + self.gamma) - torch.special.ndtr(normalized_x - self.gamma)


    # input: Random points in latent space -> [batch_size, 2] -> FIX
    # output: probability of the random points -> [batch_size, class_num(=100)]
    def probability(self, random_points):
        probs = []
        for point in random_points:
            x, y = point[0], point[1]
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

            if sum == 0:
                class_prob_normalized = [0.0 for _ in class_prob]
                print('sum is zero')
                print(x, y)
                for c in range(self.class_num):
                    mu_x = self.class_mu[c][0]
                    mu_y = self.class_mu[c][1]
                    var_x = self.class_var[c][0]
                    var_y = self.class_var[c][1]
                    prob_x = self.probs(x, mu_x, var_x)
                    prob_y = self.probs(y, mu_y, var_y)
                    print(prob_x, prob_y)
            else:
                class_prob_normalized = [p / sum for p in class_prob]

            probs.append(class_prob_normalized)

        return torch.tensor(probs).to(self.device)