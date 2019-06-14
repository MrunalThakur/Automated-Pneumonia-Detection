#3

import pandas as pd
import numpy as np
import tqdm


label_csv= pd.read_csv('D:\\BE Project\\datasets\\prepDicomData\\TrainLabels.csv')
labels= label_csv.Target
length= int(len(labels)/20)
L= 25980
R= 26680

for count in tqdm.tqdm(range(1)):
    with open("D:\\BE Project\\datasets\\prepDicomData\\TrainData\\temp",'w') as labelList:
        np.save("D:\\BE Project\\datasets\\prepDicomData\\TrainData\\TESTLabel",labels[L:R])
    L+= length
    R+= length