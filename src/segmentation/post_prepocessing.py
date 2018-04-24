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
for img, label in data_loader.test():
    print(i)
    input = centralize(img)
    input = normalize(input)
    input = input[np.newaxis,np.newaxis,:,:]
    input = torch.FloatTensor(input)
    input = Variable(input).cuda()
    output = net(input)
    output = F.log_softmax(output,dim=1)

    pred = output.max(dim=1)[1].data.squeeze().cpu().numpy()
    pred *= 255
    pred = pred.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20, 20)) 
    pred = cv2.erode(pred, kernel)

    pred, contours, hie = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            cv2.fillConvexPoly(pred, contour, 0)
    #_ , pred = cv2.threshold(pred, 20,255, cv2.THRESH_BINARY)
    img = img / np.max(img) * 255
    img = img.astype(np.uint8)
    cv2.imshow("figure",label)
    cv2.imshow("md",pred)
    cv2.waitKey(0)
    i += 1
