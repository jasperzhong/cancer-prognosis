import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 5,
                stride=1,
            ),  #16x508x508
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2), #16x254x254
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 5,
                stride=1,
            ),  #32x250x250
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2), #32x125x125
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 4,
                stride=1,
            ),  #64x122x122
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2), #64x61x61
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = 4,
                stride=1,
            ),  #128x58x58
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2), #128x29xx29
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels = 128,
                out_channels = 256,
                kernel_size = 2,
                stride=1,
            ),  #256x28x28
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2), #256x14x14
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels = 256,
                out_channels = 512,
                kernel_size = 3,
                stride=1,
            ),  #512x12x12
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2), #512x6x6
        )


        self.fc1 = nn.Sequential(
            nn.Linear(512*6*6, 1024),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.out = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.out(x)
        return x