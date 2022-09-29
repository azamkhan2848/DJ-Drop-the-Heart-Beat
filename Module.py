import torch
import torch.nn as nn
import torch.nn.functional as F

class IEGMNet(nn.Module):
    def __init__(self):
        super(IEGMNet, self).__init__()
        # bs=128 lr=0.0008802139190561054
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(5, 1), stride=(1,1), padding=0),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(1,1), padding=0),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(5, 1), stride=(1,1), padding=0),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(4, 1), stride=(1,1), padding=0),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )
        
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=740, out_features=128)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=8)
        )

        self.classifier = nn.Linear(8, 8)

    def forward(self, input):
        conv1_output = self.conv1(input)

        conv2_output = self.conv2(conv1_output)

        conv3_output = self.conv3(conv2_output)

        conv4_output = self.conv4(conv3_output)

        conv4_output = conv4_output.view(-1,740)

        fc1_output = F.relu(self.fc1(conv4_output))
        fc2_output = self.fc2(fc1_output)

        out = self.classifier(fc2_output)

        return out


class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        # features=256, bs=128, lr=0.00017800615092407858,

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(6, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=740, out_features=256)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=8)
        )

        self.classifier = nn.Linear(8, 8)

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(-1,740)

        fc1_output = F.relu(self.fc1(conv5_output))
        fc2_output = self.fc2(fc1_output)
        out = self.classifier(fc2_output)

        return out

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        # bs=16, lr=0.023790790559621435, F1=32, D=8

        self.F1 = 8
        self.F2 = 16
        self.D = 2
        
        # Conv2d(in,out,kernel,stride,padding,bias)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (1, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        
        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16), padding=(0, 8), groups=self.D*self.F1, bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Linear(16*39, 8)

        self.classifier = nn.Linear(8, 8)

        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x)
        
        x = x.reshape(-1, 16*39)
        x = self.fc(x)
        out = self.classifier(x)

        return out

class EEGNetLarge(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        # bs=16, lr=0.023790790559621435, F1=32, D=8

        self.F1 = 32
        self.F2 = 16
        self.D = 8
        
        # Conv2d(in,out,kernel,stride,padding,bias)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (1, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        
        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16), padding=(0, 8), groups=self.D*self.F1, bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Linear(16*39, 8)

        self.classifier = nn.Linear(8, 8)

        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x)
        
        x = x.reshape(-1, 16*39)
        x = self.fc(x)
        out = self.classifier(x)

        return out


class SampleNetSmall(nn.Module):
    def __init__(self):
        super(SampleNetSmall, self).__init__()
        # features=256, bs=128, lr=0.00017800615092407858,

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(6, 1), stride=(2,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=10*76, out_features=8)
        )

        self.classifier = nn.Linear(8, 8)

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv_out = self.conv4(conv3_output)
        # print('conv out: {}'.format(conv_out.shape))
        conv_out = conv_out.view(-1,10*76)

        fc1_output = F.relu(self.fc1(conv_out))
        out = self.classifier(fc1_output)

        return out

class DemoNet(nn.Module):
    def __init__(self):
        super(DemoNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(6, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=740, out_features=10)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(-1,740)

        fc1_output = F.relu(self.fc1(conv5_output))
        fc2_output = self.fc2(fc1_output)
        return fc2_output

class SampleNetVerySmall(nn.Module):
    def __init__(self):
        super(SampleNetVerySmall, self).__init__()
        # features=256, bs=128, lr=0.00017800615092407858,

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(6, 1), stride=(2,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(4,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=10*77, out_features=8)
        )

        self.classifier = nn.Linear(8, 8)

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv_out = self.conv3(conv2_output)
        # print('conv out: {}'.format(conv_out.shape))
        conv_out = conv_out.view(-1,10*77)

        fc1_output = F.relu(self.fc1(conv_out))
        out = self.classifier(fc1_output)

        return out