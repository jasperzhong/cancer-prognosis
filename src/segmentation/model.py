import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 64,
            kernel_size = 3,
            stride=1,
            padding=1
        )#64x512x512
        self.conv2 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            stride=1,
            padding=1
        )#64x512x512
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2)#64x256x256

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )#128x256x256
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )#128x256x256
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2)#128x128x128

        self.conv5 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )#256x128x128
        self.conv6 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )#256x128x128
        self.conv7 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )#256x128x128
        self.maxpooling3 = nn.MaxPool2d(kernel_size=2)#256x64x64

        self.conv8 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )#512x64x64
        self.conv9 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )#512x64x64
        self.conv10 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )#512x64x64
        self.maxpooling4 = nn.MaxPool2d(kernel_size=2)#512x32x32

        self.conv11 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )#512x32x32
        self.conv12 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )#512x32x32
        self.conv13 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )#512x32x32

        self.conv14 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
        )#128x128x128
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear') #128x512x512
        self.conv15 = nn.Conv2d(
            in_channels=128,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0
        )#1x512x512

        self.conv16 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
        )#128x64x64
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear') #128x512x512
        self.conv17 = nn.Conv2d(
            in_channels=128,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0
        )#1x512x512

        self.conv18 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
        )#128x32x32
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear') #128x512x512
        self.conv19 = nn.Conv2d(
            in_channels=128,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0
        )#1x512x512



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpooling1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpooling2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        stage1 = x
        x = self.maxpooling3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        stage2 = x
        x = self.maxpooling4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        stage3 = x

        stage3 = self.conv18(stage3)
        stage3 = self.upsample16(stage3)
        stage3 = self.conv19(stage3)

        stage2 = self.conv16(stage2)
        stage2 = self.upsample8(stage2)
        stage2 = self.conv17(stage2)

        stage1 = self.conv14(stage1)
        stage1 = self.upsample4(stage1)
        stage1 = self.conv15(stage1)

        result = (stage1 + stage2 + stage3)/3
        return result
        