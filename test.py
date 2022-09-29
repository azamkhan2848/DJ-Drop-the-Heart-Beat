import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# from help_code_demo import ToTensor, IEGM_DataSET

from Dataset import ToTensor, IEGM_DataSET
from Module import IEGMNet, EEGNet, SampleNet, SampleNetSmall, DemoNet
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
    num_classes = 8

    # from scratch
    # model_name = 'IEGMNet'
    # net = torch.load(f"./saved_models/{model_name}.pkl")
    # summary(net, (BATCH_SIZE, 1, 1250, 1))
    # model_name = 'SampleNet'
    # net = torch.load(f"./saved_models/{model_name}.pkl")
    # summary(net, (BATCH_SIZE, 1, 1250, 1))
    # model_name = 'EEGNet'
    # net = torch.load(f"./saved_models/{model_name}.pkl")
    # summary(net, (BATCH_SIZE, 1, 1250, 1))
    # model_name = 'EEGNet_kd'
    # net = torch.load(f"./saved_models/{model_name}.pkl")
    # summary(net, (BATCH_SIZE, 1, 1250, 1))
    # model_name = 'EEGNetLarge'
    # net = torch.load(f"./saved_models/{model_name}.pkl")
    # summary(net, (BATCH_SIZE, 1, 1250, 1))
    # model_name = 'SampleNetSmall'
    # net = torch.load(f"./saved_models/{model_name}.pkl")
    # summary(net, (BATCH_SIZE, 1, 1250, 1))
    model_name = 'DemoNet'
    net = DemoNet()
    summary(net, (BATCH_SIZE, 1, 1250, 1))

    # fine tuning
    # net = torch.load(f"./saved_models/{model_name}8.pkl")
    # net = torch.load(f"./saved_models/{model_name}.pkl")

    # summary(net, (BATCH_SIZE, 1, 1250, 1))

    # Start dataset loading
    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]),
                           num_classes=num_classes)

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    epoch_num = EPOCH

    trainer = IEGMTrainer(net, optimizer, criterion, model_name)

    # trainer.test(testloader)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=200)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0008802139190561054)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=128)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='../tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    # print("device is --------------", device)

    main()