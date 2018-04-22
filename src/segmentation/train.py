import logging

import torch
import torch.optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

from model import CNN
from data_loader import DataLoader


EPOCH = 1000
LR = 0.0000003
BATCH_SIZE = 4

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("drive/GPU/train.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)

with open("drive/GPU/lowest_loss.txt", 'r') as file:
    lowest_loss = float(file.readline())
    epoch0 = int(file.readline())
print("Last Train: lowest_loss %f , epoch0 %d" % (lowest_loss, epoch0))
try:
    net = torch.load('drive/GPU/seg_model.pkl')
except:
    net = CNN()
net.cuda()


optimizer = torch.optim.SGD(net.parameters(),lr=LR, weight_decay=2.5, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.33)

loss_func = nn.NLLLoss2d()

data_loder = DataLoader()

for epoch in range(epoch0, epoch0+EPOCH):
    scheduler.step()
    train_step = 0
    test_step = 0
    train_loss = 0
    test_loss = 0
    for train, test in data_loder.gen(BATCH_SIZE, get_stored=True):
        for X, y in train:
            X = torch.FloatTensor(X[:,np.newaxis,:,:])
            y = torch.LongTensor(y)
            X = Variable(X).cuda()
            y = Variable(y).cuda()

            output = net(X)
            loss = loss_func(output, y)
            train_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1

        for X_, y_ in test:
            X_ = torch.FloatTensor(X_[:,np.newaxis,:,:])
            y_ = torch.LongTensor(y_)
            X_ = Variable(X_).cuda()
            y_ = Variable(y_).cuda()
            output = net(X_)
            loss = loss_func(output, y_)
            test_loss += loss.data[0]
            test_step += 1
            del output
       

    logger.info("Train: [Epoch %d] loss %.4f" % (epoch, train_loss/train_step))
    logger.info("Test: [Epoch %d] loss %.4f" % (epoch, test_loss/test_step))

    if epoch%5 == 0 and test_loss/test_step < lowest_loss:
        lowest_loss = test_loss/test_step 
        torch.save(net, 'drive/GPU/seg_model.pkl')
        with open("drive/GPU/lowest_loss.txt", 'w') as file:
            file.write(str(lowest_loss))
            file.write("\n")
            file.write(str(epoch))
        logger.info("[Model Update]")

