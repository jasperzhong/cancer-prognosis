import cv2
import numpy
import torch
from torch.autograd import Variable

from data_loader import DataLoader

net = torch.load('/home/yuchen/Programs/cancer-prognosis/seg_model.pkl')

data_loader = DataLoader()
 
