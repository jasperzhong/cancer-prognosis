import logging

import torch
import torch.optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import FCN
from data_loader import DataLoader


EPOCH = 1000
LR = 0.001
BATCH_SIZE = 1

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("/home/yuchen/Programs/cancer-prognosis/train.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal(m.weight.data,mean=0,std=1)
        torch.nn.init.normal(m.bias.data,mean=0,std=1)

with open("/home/yuchen/Programs/cancer-prognosis/best.txt", 'r') as file:
    best = float(file.readline())
    epoch0 = int(file.readline())
print("Last Train: accu %f , epoch0 %d" % (best, epoch0))
try:
    net = torch.load('/home/yuchen/Programs/cancer-prognosis/seg_model.pkl')
except:
    net = FCN()
    net.apply(weights_init)
net.cuda()


optimizer = torch.optim.Adam(net.parameters(),lr=LR, weight_decay=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

loss_func = nn.NLLLoss2d(weight=torch.FloatTensor([1,8000]).cuda())

data_loder = DataLoader()
zeros = np.zeros((512,512))
for epoch in range(epoch0, epoch0+EPOCH):
    scheduler.step()
    train_step = 0
    test_step = 0
    train_loss = 0
    test_loss = 0
    train_accu = 0
    test_accu = 0
    for train, test in data_loder.gen(BATCH_SIZE, get_stored=True):
        net = net.train()
        for X, y in train:
            X = torch.FloatTensor(X[:,np.newaxis,:,:])
            y = torch.LongTensor(y)
            X = Variable(X).cuda()
            y = Variable(y).cuda()
            
            output = net(X)
            output = F.log_softmax(output,dim=1)
            loss = loss_func(output, y)
            train_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1

            label_pred = output.max(dim=1)[1].data.squeeze().cpu().numpy()
            label_true = y.data.squeeze().cpu().numpy()
            accu = sum(sum(label_pred == label_true))/512.0/512.0
            train_accu += accu
        

        net = net.eval()
        for X_, y_ in test:
            X_ = torch.FloatTensor(X_[:,np.newaxis,:,:])
            y_ = torch.LongTensor(y_)
            X_ = Variable(X_).cuda()
            y_ = Variable(y_).cuda()
            output = net(X_)
            output = F.log_softmax(output,dim=1)
            loss = loss_func(output, y_)
            test_loss += loss.data[0]
            test_step += 1

            label_pred = output.max(dim=1)[1].data.squeeze().cpu().numpy()
            label_true = y_.data.squeeze().cpu().numpy()
            accu = sum(sum(label_pred == label_true))/512.0/512.0
            test_accu += accu
            del output
        
    train_accu = train_accu/train_step
    test_accu = test_accu/test_step
    print("train_accu: %.4f | test_accu: %.4f" % (train_accu, test_accu))
    logger.info("Train: [Epoch %d] loss %.4f" % (epoch, train_loss/train_step))
    logger.info("Test: [Epoch %d] loss %.4f" % (epoch, test_loss/test_step))

    with open("/home/yuchen/Programs/cancer-prognosis/accu.txt", "a+") as file:
        file.write(str(train_accu)+"\n"+str(test_accu)+"\n")


    if  train_accu > best:
        best = train_accu 
        torch.save(net, '/home/yuchen/Programs/cancer-prognosis/seg_model.pkl')
        with open("/home/yuchen/Programs/cancer-prognosis/best.txt", 'w') as file:
            file.write(str(best))
            file.write("\n")
            file.write(str(epoch))
        logger.info("[Model Update]")

