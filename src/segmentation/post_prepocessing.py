import cv2
import numpy as np
import torch
from torch.autograd import Variable

from data_loader import DataLoader
from utils import normalize, centralize

net = torch.load('/home/yuchen/Programs/cancer-prognosis/seg_model.pkl')

data_loader = DataLoader()

i = 0
#cv2.namedWindow("figure")
for img in data_loader.test():
    print(i)
    input = centralize(img)
    input = normalize(input)
    input = input[np.newaxis,np.newaxis,:,:]
    input = torch.FloatTensor(input)
    input = Variable(input).cuda()
    output = net(input)
    pred = output.max(1)[1].squeeze().cpu().data.numpy()
    print(pred.shape)
    cv2.imshow("figure",pred)
    cv2.waitKey(0)
    i += 1
