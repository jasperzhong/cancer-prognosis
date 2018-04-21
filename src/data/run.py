"""script"""
import os
from shutil import copyfile

import pandas as pd

import config
from data_retriever import DataRetriver
from visualizer import Visualizer

def parse(contour, pos):
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
        contour[i] = (float(contour[i]) - float(pos[0]))/499*512
        contour[i+1] = contour[i+1].replace('\'','')
        contour[i+1] = (float(contour[i+1]) - float(pos[1]))/499*512

        new_contour.append([[int(contour[i]), int(contour[i+1])]])
    return new_contour

if not os.path.exists(r"D:\Program\diagnosis\data\data.xlsx"):
    data_retriever = DataRetriver(config.Config)
    data_retriever.start()



df = pd.read_excel(r"D:\Program\diagnosis\data\data.xlsx")

'''
paths = df["path"]
for i, path in enumerate(paths):
    copyfile(path, config.Config.train_data_path+"\\"+str(i)+".dcm")
'''

visualizer = Visualizer()
files = os.listdir(r"D:\Program\diagnosis\data\data")
files.sort(key = lambda x:int(x[:-4]))
print(files)
for file, contour, pos in zip(files, df["contour"], df["pos"]):
    contour = parse(contour, pos)
    visualizer.visualize(os.path.join(r"D:\Program\diagnosis\data\data\\",file), contour)

