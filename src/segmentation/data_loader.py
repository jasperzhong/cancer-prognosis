import os
import random

import numpy as np
import pydicom as pc
import pandas as pd 
import cv2

from utils import Batchgen

class DataLoader(object):
    def __init__(self):
        self.data_path = '/home/yuchen/Programs/cancer-prognosis/data/raw'
        self.data_size = None
        self.save_size = 100
        self.X = []
        self.y = []
        self.train_X = []
        self.train_y = []
        self.test_X = []
        self.test_y = []

    def gen(self, batch_size, get_stored=True):
        if not get_stored:
            self.get_data()    

        files = os.listdir('/home/yuchen/Programs/cancer-prognosis/data/train2')
        length = int(len(files)/2)

        for i in range(length):
            self.load(i)
            self.division()
            self.train_X = np.array(self.train_X).squeeze()
            self.train_y = np.array(self.train_y).squeeze()
            self.test_X = np.array(self.test_X).squeeze()
            self.test_y = np.array(self.test_y).squeeze()
            train = Batchgen(self.train_X, self.train_y, batch_size)
            test = Batchgen(self.test_X, self.test_y, batch_size)
            yield train, test
        raise StopIteration
    
    def test(self):
        files = os.listdir('/home/yuchen/Programs/cancer-prognosis/data/test')
        files.sort(key= lambda x:int(x[:-4]))
        for file in files:
            file = os.path.join('/home/yuchen/Programs/cancer-prognosis/data/test', file)
            dcm = pc.read_file(file)
            yield dcm.pixel_array
            
    def reset(self, batch_size):
        train = Batchgen(self.train_X, self.train_y, batch_size)
        test = Batchgen(self.test_X, self.test_y, batch_size)
        return train, test

    def centralize(self):
        for i in range(self.save_size):
            self.X[i] = self.X[i].astype('float64')
            mean = np.mean(self.X[i])
            self.X[i] -= mean
    
    def normalize(self):
        for i in range(self.save_size):
            max_ = np.max(self.X[i])
            self.X[i] = self.X[i] / max_

    def get_data(self):
        files = os.listdir(self.data_path)
        files.sort(key= lambda x:int(x[:-4]))
        self.data_size = len(files)
        
        df = pd.read_excel("/home/yuchen/Programs/cancer-prognosis/data/data.xlsx")
        contours = df["contour"]
        poses = df["pos"]

        i = 0
        cnt = 0
        for file, contour, pos in zip(files, contours, poses):
            self.read_dcm(os.path.join(self.data_path, file))
            self.read_xlsx(contour,pos)
            if  i == 100:
                self.centralize()
                self.normalize()
                self.save(cnt)
                self.X = []
                self.y = []
                i = 0
                cnt += 1
            i += 1


    def read_dcm(self, file):
        dcm = pc.read_file(file)
        self.X.append(dcm.pixel_array)
    
    def read_xlsx(self, contour, pos):
        new_label = np.zeros((512,512))
        new_contour = self.parse(contour, pos)
        cv2.fillConvexPoly(new_label, new_contour, 1)
        self.y.append(new_label)


    def parse(self, contour, pos):
        contour = contour.replace('[','')
        contour = contour.replace(']','')
        contour = contour.split(',')
        pos = pos.replace('[','')
        pos = pos.replace(']','')
        pos = pos.split(',')

        length = len(contour)
        new_contour = []
        for i in range(0, length, 3):
            contour[i] = contour[i].replace('\'','')
            
            pos[0] = pos[0].replace('\'','')
            pos[1] = pos[1].replace('\'','')
            try:
                contour[i] = (float(contour[i]) - float(pos[0]))/499*512
                contour[i+1] = contour[i+1].replace('\'','')
                contour[i+1] = (float(contour[i+1]) - float(pos[1]))/499*512
            except ValueError or IndexError:
                continue
            new_contour.append([[int(contour[i]), int(contour[i+1])]])
        return np.array(new_contour)
        

    def division(self):
        self.train_X = self.X[0:80]
        self.train_y = self.y[0:80]
        self.test_X = self.X[80:-1]
        self.test_y = self.y[80:-1]

    def save(self, i):
        np.save("/home/yuchen/Programs/cancer-prognosis/data/processed/X"+str(i)+".npy", self.X)
        np.save("/home/yuchen/Programs/cancer-prognosis/data/processed/y"+str(i)+".npy", self.y)
    
    def load(self, i):
        self.X = np.load("/home/yuchen/Programs/cancer-prognosis/data/train2/X"+str(i)+".npy")
        self.y = np.load("/home/yuchen/Programs/cancer-prognosis/data/train2/y"+str(i)+".npy")


