from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

DataPath= "D:\\BE Project\\datasets\\prepDicomData\\TrainData\\Test0.npz"
LabelPath=  "D:\\BE Project\\datasets\\prepDicomData\\TrainData\\TESTLabel.npy"

px= np.load(DataPath)
TestData= px['arr_0']
TestData= np.array(TestData).reshape(-1,256,256,1)
TestLabels= np.load(LabelPath)

with open('Models/Model_Architecture.json') as arch:
    model= keras.models.model_from_json(arch.read())
    
model.compile(optimizer= 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics= ['accuracy'] )

model.load_weights('Models/trial.h5')
model.save('Models/pulmoX.model')
(loss,acc) = model.evaluate(TestData,TestLabels)
print("Loss: "+str(loss)+"\nAcc: "+str(acc))