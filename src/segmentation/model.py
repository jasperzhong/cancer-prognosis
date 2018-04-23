import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 64,
                kernel_size = 3,
                stride=1,
                padding=1
            ),#64x512x512
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 64,
                out_channels = 64,
                kernel_size = 3,
                stride=1,
                padding=1
            ),#64x512x512
            nn.ReLU()
        )
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2)#64x256x256

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),#128x256x256
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),#128x256x256
            nn.ReLU()
        )
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2)#128x128x128

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),#256x128x128
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),#256x128x128
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),#256x128x128
            nn.ReLU()
        )
        self.maxpooling3 = nn.MaxPool2d(kernel_size=2)#256x64x64

        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),#512x64x64
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),#512x64x64
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),#512x64x64
            nn.ReLU()
        )
        self.maxpooling4 = nn.MaxPool2d(kernel_size=2)#512x32x32

        self.conv11 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1
            ),#1024x32x32
            nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1
            ),#1024x32x32
            nn.ReLU()
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1
            ),#1024x32x32
            nn.ReLU()
        )

        self.scores1 = nn.Conv2d(1024, 2, 1)
        self.scores2 = nn.Conv2d(512, 2, 1)
        self.scores3 = nn.Conv2d(256, 2, 1)
        
        self.upsample_8x = nn.ConvTranspose2d(2,2,8,4,2,bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(2, 2, 8)

        self.upsample_4x = nn.ConvTranspose2d(2,2,4,2,1,bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(2, 2, 4)

        self.upsample_2x = nn.ConvTranspose2d(2,2,4,2,1,bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(2, 2, 4)


    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        x = self.maxpooling1(x)

        x = self.conv3(x)
        #x = self.conv4(x)
        x = self.maxpooling2(x)

        x = self.conv5(x)
        #x = self.conv6(x)
        #x = self.conv7(x)
        s1 = x.clone()#256x128x128
        x = self.maxpooling3(x)

        x = self.conv8(x)
        #x = self.conv9(x)
        #x = self.conv10(x)
        s2 = x.clone()#512x64x64
        x = self.maxpooling4(x)

        x = self.conv11(x)
        #x = self.conv12(x)
        #x = self.conv13(x)
        s3 = x.clone()#1024x32x32

        s3 = self.scores1(s3)#2x32x32
        s3 = self.upsample_2x(s3) #2x64x64
        s2 = self.scores2(s2)#2x64x64
        s2 = s2 + s3 

        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)#2x128x128
        s = s1 + s2 

        s = self.upsample_8x(s) #2x512x512
        
        return s

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
