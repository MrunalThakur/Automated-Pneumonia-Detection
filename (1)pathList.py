#1

import pandas as pd
import os
import numpy as np
# dataset = pydicom.dcmread('D:\\BE Project\\datasets\\train Dicom')

# csvDicom= pd.read_csv('D:\\BE Project\\datasets\\TrainLabels.csv')
# LabelSet= pd.DataFrame(csvDicom,columns=['patientID','Target'])
# # print(LabelSet.isnull().any())

count= 0
PathDicom = 'D:\\BE Project\\datasets\\train Dicom'


DicomList = [] 
patientID= []
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        count +=1
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            DicomList.append(os.path.join(dirName,filename))
            patientID.append(filename)
'''
with open("D:\\BE Project\\datasets\\prepDicomData\\DicomList.npy",'w') as DL:
    np.save('D:\\BE Project\\datasets\\prepDicomData\\DicomList',DicomList)
'''
DicomDataFrame= pd.DataFrame(DicomList)
DicomDataFrame.to_csv('D:\\BE Project\\datasets\\prepDicomData\\DicomList.csv')
patientIdList= pd.DataFrame(patientID)
patientIdList.to_csv('D:\\BE Project\\datasets\\prepDicomData\\patientID.csv')