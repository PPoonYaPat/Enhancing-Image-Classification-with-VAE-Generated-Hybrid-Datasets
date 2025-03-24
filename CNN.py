# This file is for CNN model class

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
import copy
import time

import math

class CNN(nn.Module):
    def __init__(self, image_width, image_height, device, learning_rate, class_num):
        super(CNN, self).__init__()

        self.device = device
        self.image_width = image_width
        self.image_height = image_height
        self.class_num = class_num

        self.layer1 = self.ConvModule(3, 64)
        self.layer2 = self.ConvModule(64, 128)
        self.layer3 = self.ConvModule(128, 256)
        self.layer4 = self.ConvModule(256, 512)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear((int)(image_width/16)*(int)(image_height/16)*512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.class_num)
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(x)
        return x

    def ConvModule(self, in_features, out_features):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

    def train_normal_model(self, dataloaders, epochs):
        since = time.time()
        ce_criterion = nn.CrossEntropyLoss()

        test_acc_history = []
        test_loss_history = []
        train_acc_history = []
        train_loss_history = []

        best_model_wts = copy.deepcopy(self.state_dict())
        best_acc = 0.0

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))

            for phase in ['train', 'test']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.forward(inputs)
                        loss = ce_criterion(outputs, labels) # here is the average of each batch
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'test':
                    test_acc_history.append(epoch_acc.cpu().numpy())
                    test_loss_history.append(epoch_loss)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.state_dict())
                if phase == 'train':
                    train_acc_history.append(epoch_acc.cpu().numpy())
                    train_loss_history.append(epoch_loss)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))

        history_dict = {'train_loss':train_loss_history, 'train_accuracy':train_acc_history, 'test_loss':test_loss_history, 'test_accuracy':test_acc_history}
        return best_model_wts, history_dict



    def train_hybrid_model(self, dataloaders, epochs):
        since = time.time()

        test_acc_history = []
        test_loss_history = []
        train_acc_history = []
        train_loss_history = []

        best_model_wts = copy.deepcopy(self.state_dict())
        best_acc = 0.0

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))

            for phase in ['train', 'test']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, expected_probabilities in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    expected_probabilities = expected_probabilities.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.forward(inputs)
                        outputs = outputs.softmax(dim=1)

                        # Compute Cross-Entropy Loss manually
                        loss = -(expected_probabilities * torch.log(outputs + 1e-8)).sum(dim=1).mean()

                        if math.isnan(loss.item()):
                            print("Loss is NaN")
                            print("Outputs - Min:", outputs.min().item())
                            print("Outputs - Max:", outputs.max().item())
                            print("Outputs - Row sums:", outputs.sum(dim=1))
                            print("Outputs contain NaN:", torch.isnan(outputs).any())

                            print("Expected probabilities - Min:", expected_probabilities.min().item())
                            print("Expected probabilities - Max:", expected_probabilities.max().item())
                            print("Expected probabilities - Row sums:", expected_probabilities.sum(dim=1))
                            print("Expected probabilities contain NaN:", torch.isnan(expected_probabilities).any())
                            #for  (i, j) in zip(outputs, expected_probabilities):
                            #    print(i, j)
                            break

                        preds = torch.argmax(outputs, dim=1)
                        expected_preds = torch.argmax(expected_probabilities, dim=1)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == expected_preds)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'test':
                    test_acc_history.append(epoch_acc.cpu().numpy())
                    test_loss_history.append(epoch_loss)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.state_dict())
                if phase == 'train':
                    train_acc_history.append(epoch_acc.cpu().numpy())
                    train_loss_history.append(epoch_loss)

            print()


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))
        print("-------------------")
        print()

        history_dict = {'train_loss':train_loss_history, 'train_accuracy':train_acc_history, 'test_loss':test_loss_history, 'test_accuracy':test_acc_history}
        return best_model_wts, history_dict