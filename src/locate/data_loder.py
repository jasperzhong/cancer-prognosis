import pydicom as pc
import pandas as pd 
import os
import random
import numpy as np

from utils import Batchgen

class DataLoader(object):
    def __init__(self):
        self.data_path = '/home/yuchen/Programs/diagnosis/data'
        self.data_size = -1
        self.X = []
        self.y = []
        self.train_X = []
        self.train_y = []
        self.test_X = []
        self.test_y = []

    def start(self, batch_size):
        self.get_data()
        self.centralize()
        self.normalize()
        self.division()
        self.train_X = np.array(self.train_X).squeeze()
        self.train_y = np.array(self.train_y).squeeze()
        self.test_X = np.array(self.test_X).squeeze()
        self.test_y = np.array(self.test_y).squeeze()

        train = Batchgen(self.train_X, self.train_y, batch_size)
        test = Batchgen(self.test_X, self.test_y, batch_size)
        return train, test

    def reset(self, batch_size):
        train = Batchgen(self.train_X, self.train_y, batch_size)
        test = Batchgen(self.test_X, self.test_y, batch_size)
        return train, test

    def centralize(self):
        """
        
        """
        pass
    
    def normalize(self):
        pass

    def get_data(self):
        files = os.listdir(self.data_path)
        files.sort(key= lambda x:int(x[:-4]))
        for file in files:
            if ".dcm" in file:
                file = os.path.join(self.data_path,file)
                self.read_dcm(file)
                self.data_size += 1
        
        self.read_xlsx("/home/yuchen/Programs/diagnosis/data.xlsx")
            
    def read_dcm(self, file):
        dcm = pc.read_file(file)
        self.X.append(dcm.pixel_array)
    
    def read_xlsx(self, file):
        df = pd.read_excel(file)
        df = df["pos"]
        df = df.astype(list)
        Y = df.as_matrix()
        for y in Y:
            self.y.append(self.parse(y))

    def parse(self, str):
        str = str.replace('[','')
        str = str.replace(']','')
        str = str.split(',')
        int_list = []
        for s in str:
            s = int(s)
            int_list.append(s)
        return (int_list[0], int_list[1])

    def division(self):
        '''
        start_index = random.randint(0, self.data_size)
        step = 800
        end_index = start_index + step
        if end_index > self.data_size:
            self.train_X.append(self.X[start_index:-1])
            self.train_y.append(self.y[start_index:-1])
            start_index = end_index%self.data_size
            self.train_X.append(self.X[0:start_index])
            self.test_y.append(self.y[0:start_index])

            self.test_X.append(self.X[start_index:end_index])
            self.test_y.append(self.y[start_index:end_index])
        else:
            self.train_X.append(self.X[start_index:end_index])
            self.train_y.append(self.y[start_index:end_index])
            self.test_X.append(self.X[end_index:-1])
            self.test_y.append(self.y[end_index:-1])
        '''
        self.train_X = self.X[0:800]
        self.train_y = self.y[0:800]
        self.test_X = self.X[800:-1]
        self.test_y = self.y[800:-1]
