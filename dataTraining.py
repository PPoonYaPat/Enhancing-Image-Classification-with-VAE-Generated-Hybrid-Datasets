# This file is for dataTraining class
# which is the class that will create the model and train it using various numbers of hybrid datasets and original datasets
# and return the best model and its accuracy

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from CNN import CNN
from VAE import VAE

class DataTraining:
    def __init__(self, batch_size, path, learning_rate, device, class_num):
        self.class_num = class_num
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.path = path
        self.load_CNN_data()
        self.load_VAE_data()
        self.image_width = 32
        self.image_height = 32

        self.CNN_class_indices = [[] for _ in range(self.class_num)]
        self.VAE_class_indices = [[] for _ in range(self.class_num)]

        for i in range(len(self.CNN_train)):
            self.CNN_class_indices[self.CNN_train[i][1]].append(i)

        for i in range(len(self.VAE_train)):
            self.VAE_class_indices[self.VAE_train[i][1]].append(i)


    def load_CNN_data(self):
        self.transforms_CNN = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter()
        ])

        self.CNN_train = CIFAR100(self.path, train=True, download=True, transform=self.transforms_CNN)
        self.CNN_test = CIFAR100(self.path, train=False, download=True, transform=self.transforms_CNN)
        self.CNN_test_dl = DataLoader(self.CNN_test, batch_size=self.batch_size, shuffle=False)

    def load_VAE_data(self):
        self.transforms_VAE = transforms.Compose([transforms.ToTensor()])
        self.VAE_train = CIFAR100(self.path, train=True, download=True, transform=self.transforms_VAE)
        self.VAE_test = CIFAR100(self.path, train=False, download=True, transform=self.transforms_VAE)
        extended_label_VAE_train = []
        for i in range(len(self.VAE_train)):
            extended_label_VAE_train.append((self.VAE_train[i][0].clone(), self.extent_label(self.VAE_train[i][1]).clone()))
        self.VAE_test_dl = DataLoader(extended_label_VAE_train, batch_size=self.batch_size, shuffle=False)


    # Randomly select num_each_class images from different classes from the dataset
    # return the random images and their labels (as dataLoader)
    def random_data_normal(self, num_each_class):
        sampled_indices = []
        for i in range(self.class_num):
            sampled_indices.extend(np.random.choice(self.CNN_class_indices[i], num_each_class, replace=False))
        sampled_CNN_train = DataLoader(Subset(self.CNN_train, sampled_indices), batch_size=self.batch_size, shuffle=True)
        return {'train': sampled_CNN_train, 'test': self.CNN_test_dl}


    # Randomly select num_each_class images from different classes from the dataset
    # return the random images and their labels (as dataLoader)
    def random_data_new(self, num_each_class):
        sampled_indices = []
        for i in range(self.class_num):
            sampled_indices.extend(np.random.choice(self.VAE_class_indices[i], num_each_class, replace=False))

        # Extend labels to one-hot encoded and store in the dataset
        extended_label_sampled_dataset = []
        for i in sampled_indices:
            extended_label_sampled_dataset.append((self.VAE_train[i][0].clone(), self.extent_label(self.VAE_train[i][1]).clone()))  # One-hot encode

        # Use DataLoader
        sampled_VAE_train = DataLoader(extended_label_sampled_dataset, batch_size=self.batch_size, shuffle=True)

        return {'train': sampled_VAE_train, 'test': self.VAE_test_dl}


    # extend the provided label to the array with lenght of class_num
    def extent_label(self, label):
        extended_label = [0 for _ in range(self.class_num)]
        extended_label[label] = 1
        return torch.tensor(extended_label, dtype=torch.float32)


    def train_normal_CNN(self, num_each_class, epochs):
        dataLoader = self.random_data_normal(num_each_class)
        model = CNN(self.image_width, self.image_height, self.device, self.learning_rate, self.class_num).to(self.device)
        return model.train_normal_model(dataLoader, epochs)


    def train_new_CNN(self, num_each_class, num_hybrid_times, epochs):
        dataLoader = self.random_data_new(num_each_class) # has train and test

        # train the VAE model
        model_VAE = VAE(self.image_width, self.image_height, 3, [64, 128, 256, 512], 2, self.device, self.learning_rate, self.class_num, self.batch_size).to(self.device)
        model_VAE.train_model(dataLoader, epochs)
        model_VAE.calculate_class_mu_var(dataLoader['train'])
        sampled_hybrid_images = model_VAE.hybrid_image(num_hybrid_times * num_each_class)
        hybrid_dataLoader = {'train': sampled_hybrid_images, 'test': dataLoader['test']}

        # train the CNN model
        model_CNN = CNN(self.image_width, self.image_height, self.device, self.learning_rate, self.class_num).to(self.device)
        model_CNN.train_hybrid_model(dataLoader, epochs)
        return model_CNN.train_hybrid_model(hybrid_dataLoader, epochs)