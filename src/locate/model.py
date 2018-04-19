import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 20,
                kernel_size = 5,
                stride=1
            ), #20x508x508
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=2) #20x254x254
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 20,
                out_channels = 50,
                kernel_size = 5,
                stride=1
            ), #50x250x250
            nn.BatchNorm2d(50),
            nn.MaxPool2d(kernel_size=2) #50x125x125
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 50,
                out_channels = 100,
                kernel_size = 4,
                stride=1
            ), #100x122x122
            nn.BatchNorm2d(100),
            nn.MaxPool2d(kernel_size=2) #100x61x61
        )

        self.conv3_4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 100,
                out_channels = 100,
                kernel_size = 3,
                stride=1
            ), #100x59x59
            nn.BatchNorm2d(100),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 100,
                out_channels = 300,
                kernel_size = 4,
                stride=1
            ), #300x56x56
            nn.BatchNorm2d(300),
            nn.MaxPool2d(kernel_size=2) #300x28x28
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels = 300,
                out_channels = 500,
                kernel_size = 3,
                stride=1
            ), #500x26x26
            nn.BatchNorm2d(500),
            nn.MaxPool2d(kernel_size=2) #500x13x13
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels = 500,
                out_channels = 800,
                kernel_size = 4,
                stride=1
            ), #800x10x10
            nn.BatchNorm2d(800),
            nn.MaxPool2d(kernel_size=2) #800x5x5
        )

        self.fc1 = nn.Sequential(
            nn.Linear(20000, 1000),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 100),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.out = nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_4(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x