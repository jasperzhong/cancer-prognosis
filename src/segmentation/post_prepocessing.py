import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

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
  
    pred = pred / np.max(pred) *255
    pred = pred.astype(np.uint8)
    _ , pred = cv2.threshold(pred, 127,255, cv2.THRESH_BINARY)
    print(pred)
    img = img / np.max(img) * 255
    img = img.astype(np.uint8)
    cv2.imshow("figure",img)
    cv2.imshow("md",pred)
    cv2.waitKey(0)
    i += 1
