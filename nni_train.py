import nni
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt
from typing import Callable, Any
import torch.optim as optim
import torch.nn.functional as F

import argparse

from Dataset import ToTensor, IEGM_DataSET
from Module import EEGNet, SampleNet, SampleNetSmall
from Trainer import IEGMTrainer


# Get optimized hyperparameters
params = {
          'bs': 128, 
          'lr': 0.001,
        }
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

epochs = 200
data_path = "../tinyml_contest_data_training/"
data_indices = "./data_indices"
SIZE = 1250

num_classes = 8

# Load dataset
trainset = IEGM_DataSET(root_dir=data_path,
                        indice_dir=data_indices,
                        mode='train',
                        size=SIZE,
                        transform=transforms.Compose([ToTensor()]),
                        num_classes=8)

trainloader = DataLoader(trainset, batch_size=params['bs'], shuffle=True, num_workers=0)

testset = IEGM_DataSET(root_dir=data_path,
                       indice_dir=data_indices,
                       mode='test',
                       size=SIZE,
                       transform=transforms.Compose([ToTensor()]),
                       num_classes=8)

testloader = DataLoader(testset, batch_size=params['bs'], shuffle=False, num_workers=0)

# Build model    # 'ch1': {'_type': 'randint', '_value ': [2, 10]},
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

net = SampleNetSmall()
net = net.to(device)

# Training functions
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=params['lr'])

def train(dataloader, net, loss_fn, optimizer):
    net.train()
    for data in dataloader:
        X, y = data['IEGM_seg'], data['label']
        X, y = X.to(device, dtype=torch.float), y.to(device)
        pred = net(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, net, loss_fn):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            X, y = data['IEGM_seg'], data['label']
            X, y = X.to(device, dtype=torch.float), y.to(device)
            pred = net(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return correct / len(dataloader.dataset)

# Train the model
epochs = 10
for t in range(epochs):
    train(trainloader, net, loss_fn, optimizer)
    accuracy = test(testloader, net, loss_fn)
    nni.report_intermediate_result(accuracy)
    # print("accuracy: {}".format(accuracy))
nni.report_final_result(accuracy)
