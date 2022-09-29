import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Dataset import ToTensor, IEGM_DataSET
from Module import IEGMNet

from Trainer import IEGMTrainer
import torchvision.models as models
import torch.nn as nn

import os

from Module import IEGMNet, SampleNet, EEGNet, SampleNetSmall
from Trainer import IEGM_kd_Trainer

def main() -> None:
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    num_classes = 8

    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]),
                            num_classes=num_classes)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]),
                           num_classes=num_classes)

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)

    print("Training Dataset loading finish.")
    
    teacher_name = 'SampleNet8'
    teacher_net = torch.load('./saved_models/EEGNetLarge8.pkl')
    student_name = 'SampleNetVerySmall8'
    student_net = SampleNetSmall()
    
    criterion = nn.CrossEntropyLoss()
    optimizer_teacher = optim.Adam(teacher_net.parameters(), lr=LR)
    optimiizer_student = optim.Adam(student_net.parameters(), lr=LR)
    epoch_num = EPOCH

    print("Preparing teacher")
    trainer_teach = IEGMTrainer(teacher_net, optimizer_teacher, criterion, teacher_name)    
    if not os.path.exists(f"./saved_models/{teacher_name}.pkl"):
        trainer_teach.train(epoch_num, trainloader, testloader)
    print("Preparing student")  
    trainer_from_scratch = IEGMTrainer(student_net, optimiizer_student, criterion, student_name)
    if not os.path.exists(f"./saved_models/{student_name}.pkl"):
        trainer_from_scratch.train(epoch_num, trainloader, testloader)
    print("Preparing kd")
    trainer_kd = IEGM_kd_Trainer(teacher_net, student_net, optimiizer_student, f"{student_name}_kd", teacher_name)
    if not os.path.exists(f"./saved_models/{student_name}_kd.pkl"):
        trainer_kd.train(epoch_num, trainloader, testloader)


    print('Finish training')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=200)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0002973442497261265)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='../tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    print("device is --------------", device)

    main()