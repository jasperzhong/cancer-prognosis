import logging

import torch
import torch.optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

from model import CNN
from data_loder import DataLoader


EPOCH = 1000
LR = 0.0000003
BATCH_SIZE = 4

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("/home/yuchen/Programs/diagnosis/src/locate/train.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)

with open("/home/yuchen/Programs/diagnosis/src/locate/lowest_loss.txt", 'r') as file:
    lowest_loss = float(file.readline())
    epoch0 = int(file.readline())
print("Last Train: lowest_loss %f , epoch0 %d" % (lowest_loss, epoch0))
try:
    net = torch.load('/home/yuchen/Programs/diagnosis/src/locate/model.pkl')
except:
    net = CNN()
net.cuda()



optimizer = torch.optim.SGD(net.parameters(),lr=LR, weight_decay=2.5, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.33)

loss_func = nn.MSELoss()

data_loder = DataLoader()
train, test = data_loder.start(BATCH_SIZE)

train_loss = []
test_loss = []
for epoch in range(epoch0, epoch0+EPOCH):
    scheduler.step()
    step = 0
    losses = 0
    for X, y in train:
        X = torch.FloatTensor(X[:,np.newaxis,:,:])
        y = torch.FloatTensor(y)
        X = Variable(X).cuda()
        y = Variable(y).cuda()

        output = net(X)
        loss = loss_func(output, y)
        losses += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

    logger.info("Train: [Epoch %d] loss %.2f" % (epoch, losses/step))
    train_loss.append(losses/step)
    train, test = data_loder.reset(BATCH_SIZE)
    losses = 0
    step = 0
    for X, y in test:
        X = torch.FloatTensor(X[:,np.newaxis,:,:])
        y = torch.FloatTensor(y)
        X = Variable(X).cuda()
        y = Variable(y).cuda()
        output = net(X)
        loss = loss_func(output, y)
        losses += loss.data[0]
        step += 1
    logger.info("Test: [Epoch %d] loss %.2f" % (epoch, losses/step))
    test_loss.append(losses/step)

    if epoch%5 == 0 and losses/step < lowest_loss:
        lowest_loss = losses/step 
        torch.save(net, '/home/yuchen/Programs/diagnosis/src/locate/model.pkl')
        with open("/home/yuchen/Programs/diagnosis/src/locate/lowest_loss.txt", 'w') as file:
            file.write(str(lowest_loss.data))
            file.write("\n")
            file.write(str(epoch))
        logger.info("[Model Update]")


