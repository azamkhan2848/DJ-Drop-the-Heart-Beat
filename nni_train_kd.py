import nni
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt
from typing import Callable, Any
import torch.optim as optim
import torch.nn.functional as F

import argparse

from Dataset import ToTensor, IEGM_DataSET
from Module import IEGMNet, SampleNet, EEGNet, SampleNetSmall, SampleNetVerySmall
from Trainer import IEGMTrainer


# Get optimized hyperparameters
params = {
          'bs': 128, 
          'lr': 0.001,
          'alpha': 0.5,
          'T': 20
        }
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

epochs = 200
data_path = "../tinyml_contest_data_training/"
data_indices = "./data_indices"
SIZE = 1250

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

# Build model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

student = SampleNetVerySmall()
student = student.to(device)

teacher = torch.load('./saved_models/SampleNet8.pkl')
teacher = teacher.to(device)

# Training functions
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(student.parameters(), lr=params['lr'])

def loss_fn_kd(student_out, labels, teacher_out):
        alpha = params['alpha']
        T = params['T']
        kd_loss = nn.KLDivLoss()(F.log_softmax(student_out/T, dim=1), 
                                 F.softmax(teacher_out/T, dim=1))*(alpha*T*T)+ \
                                F.cross_entropy(student_out, labels)*(1.-alpha)
        return kd_loss

def train(dataloader, student, teacher, optimizer):
    student.train()
    teacher.eval()
    for data in dataloader:
        X, y = data['IEGM_seg'], data['label']
        X, y = X.to(device, dtype=torch.float), y.to(device)
        student_out = student(X)
        teacher_out = teacher(X)
        loss = loss_fn_kd(student_out, y, teacher_out)
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
    train(trainloader, student, teacher, optimizer)
    accuracy = test(testloader, student, loss_fn)
    nni.report_intermediate_result(accuracy)
    # print("accuracy: {}".format(accuracy))
nni.report_final_result(accuracy)
