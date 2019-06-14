#2

import pandas as pd
import keras
import tqdm
labFile= pd.read_csv("D:\\BE Project\\datasets\\TrainLabels.csv")
patientIdList= pd.read_csv('D:\\BE Project\\datasets\\prepDicomData\\patientID.csv')



labList= labFile[['patientID','Target']]
labList.drop_duplicates(inplace=True)
labList= labList.set_index('patientID')
labList.to_csv("D:\\BE Project\\datasets\\prepDicomData\\TrainLabels.csv")

for x in tqdm.tqdm(range(26680)):
    patientIdList['Target'][x]=labList['Target'][patientIdList['patientID'][x]]

patientIdList.to_csv('D:\\BE Project\\datasets\\prepDicomData\\patientID.csv')
