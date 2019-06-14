import tensorflow as tf
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import keras
import datetime
import tqdm
import time
import pandas as pd





imageList= []
DicomList= pd.read_csv("D:\\BE Project\\datasets\\prepDicomData\\DicomList.csv")  
#Dividing Dataset into 58 batches with 460 images each
length=int(len(DicomList)/20)
L= 0  
R= length



#Training Images= 26,220 Test Images= 460
with open("D:\\BE Project\\datasets\\prepDicomData\\TrainData\\logs.txt",'a') as logs:
    logs.write("\t\tSTART_TrainData: "+str(datetime.datetime.now())+'\n')    

for count in tqdm.tqdm(range(19)):
    with open("D:\\BE Project\\datasets\\prepDicomData\\TrainData\\logs.txt",'a') as logs:
        logs.write("Start_epoch_"+str(count)+"_TrainData: "+str(datetime.datetime.now())+'\n')    

    for imagePath in tqdm.tqdm(DicomList['pathName'][L:R]):
        loadImage= pydicom.dcmread(imagePath)
        pixel= loadImage.pixel_array
        #downsampling by selecting 1 o  f every 4 pixels i.e. 1024/4= 256
        pixel= pixel[::4,::4]
        pixel=pixel/255
        imageList.append(pixel) 
    
    with open("D:\\BE Project\\datasets\\prepDicomData\\TrainData\\temp",'w') as ImageList:
        np.savez_compressed("D:\\BE Project\\datasets\\prepDicomData\\TrainData\\Train"+str(count),imageList)
    with open("D:\\BE Project\\datasets\\prepDicomData\\TrainData\\logs.txt",'a') as logs:
        px= np.load("D:\\BE Project\\datasets\\prepDicomData\\TrainData\\Train"+str(count)+".npz")
        pi= px['arr_0'] 
        logs.write("End_epoch_"+str(count)+"_TrainData: "+str(datetime.datetime.now())+'\n\n'
            +"Images= "+str(len(pi))+'\n\n') 
    L+=length
    R+=length
    imageList=[]

with open("D:\\BE Project\\datasets\\prepDicomData\\TrainData\\logs.txt",'a') as logs:
    logs.write("\t\tEND_TrainData: "+str(datetime.datetime.now())+'\n'+'L= '+str(L)+' R= '+str(R))
