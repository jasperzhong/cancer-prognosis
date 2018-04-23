import cv2
import numpy
import torch
from torch.autograd import Variable

from data_loader import DataLoader
from utils import normalize, centralize

net = torch.load('/home/yuchen/Programs/cancer-prognosis/seg_model.pkl')

data_loader = DataLoader()

for img in data_loader.test():
    input = centralize(img)
    input = normalize(input)
    input = Variable(input)
    output = net(input)
    print(output)

