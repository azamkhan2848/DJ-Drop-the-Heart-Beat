import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# from help_code_demo import ToTensor, IEGM_DataSET

from Dataset import ToTensor, IEGM_DataSET
from Module import IEGMNet, EEGNet, SampleNet, EEGNetLarge, SampleNetSmall, DemoNet, SampleNetVerySmall
from Trainer import IEGMTrainer

from torchinfo import summary
from torchvision import models

def main() -> None:
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    num_classes = 2

    # from scratch
    # net = IEGMNet()
    # model_name = 'IEGMNet8'
    # net = SampleNet()
    # model_name = 'SampleNet8'
    # net = models.resnet18(pretrained=True)
    # model_name = 'resnet18_8'
    # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    # net.fc = nn.Linear(net.fc.in_features, num_classes)    
    # net = EEGNet()
    # model_name = 'EEGNet8'
    # net = SampleNetLarge()
    # model_name = 'SampleNetLarge8'
    # summary(net, (BATCH_SIZE, 1, 1250, 1))
    # model_name = 'SampleNetSmall8'
    # net = SampleNetSmall()
    # model_name = 'DemoNet'
    # net = DemoNet()
    # model_name = 'SampleNetVerySmall'
    # net = SampleNetVerySmall()

    # fine tuning
    # model_name = 'IEGMNet'
    # model_name = 'SampleNet'
    # model_name = 'EEGNet'
    # model_name = 'EEGNetLarge'
    # model_name = 'EEGNet_kd'
    # model_name = 'SampleNet'
    # model_name = 'SampleNetSmall_kd'    
    # model_name = 'SampleNetVerySmall'
    model_name = 'SampleNetVerySmall_kd'

    # net = torch.load(f"./saved_models/{model_name}8.pkl")
    net = torch.load(f"./saved_models/SampleNetVerySmall8_kd.pkl")
    net.classifier = nn.Linear(net.classifier.in_features, 2)


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

    # data = next(iter(trainloader))
    
    # print(data['label'])

    # print("Training Dataset loading finish.")
    
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    epoch_num = EPOCH

    trainer = IEGMTrainer(net, optimizer, criterion, model_name)

    trainer.train(epoch_num, trainloader, testloader)

    print('Finish training')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=200)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='../tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    print("device is --------------", device)

    main()