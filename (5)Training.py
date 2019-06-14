import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
import tqdm
import numpy as np
import h5py
import time
from tensorflow.keras.callbacks import TensorBoard

'''
im= np.load( "D:\\BE Project\\datasets\\prepDicomData\\TrainData\\Train0.npz")
pxshape= im['arr_0'][0]
pxshape= np.array(pxshape).reshape(-1,256,256,1)
'''


dataBox= "D:\\BE Project\\datasets\\prepDicomData\\TrainData\\"
    


'''
images= np.load(dataBox + "Train"+str(0)+".npz")
px= images['arr_0']
px= np.array(px).reshape(-1,256,256,1)
labels= np.load(dataBox + "TrainLabel"+str(0)+".npy")

'''
images= np.load(dataBox + "Train"+str(0)+".npz")
px= images['arr_0']
px= np.array(px).reshape(-1,256,256,1)

# conv --> MaxPooling --> conv --> MaxPooling
model= Sequential()
model.add(Conv2D(112,(3,3),input_shape= px[0].shape, activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Conv2D(112,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

#Flatten--> Dense x 3
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(112,activation= 'relu'))
model.add(Dropout(0.3))                          #drops 30% photos
model.add(Dense(112,activation= 'relu'))
model.add(Dropout(0.3))                          
model.add(Dense(112,activation= 'relu'))
model.add(Dense(2,activation= 'softmax'))

model.compile(optimizer= 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics= ['accuracy'] )




#TensorBoard to compare performance graphically
tensorboard= TensorBoard(log_dir= './logs/pulmoX',histogram_freq=0,
                          write_graph=True, write_images=True)

'''
model.fit(px,labels,epochs= 3,callbacks= [tensorboard])
model.save_weights('Models/trial.h5')
(loss,acc)= model.evaluate(px,labels)
with open("D:\\BE Project\\CodeX\\trainLogs\\model_evaluate",'w') as modeval:
    modeval.write("iter "+str(0)+"\nLoss :"+str(loss)+" Acc :"+str(acc)+'\n')
'''

with open('Models/Model_Architecture.json','w') as arch:
    arch.write(model.to_json())

for count in tqdm.tqdm(range(1,20)):
    
    images= np.load(dataBox + "Train"+str(count)+".npz")
    px= images['arr_0']
    px= np.array(px).reshape(-1,256,256,1)
    labels= np.load(dataBox + "TrainLabel"+str(count)+".npy")
    model.fit(px,labels,epochs= 3,callbacks= [tensorboard])
    if count== 19:
        model.save('Models/pulmoX.model')
        model.load_weights('Models/trial.h5')
    else :
        weightMat= model.get_weights()
    (loss,acc)= model.evaluate(px,labels)
    with open("D:\\BE Project\\CodeX\\trainLogs\\model_evaluate",'a') as modeval:
        modeval.write("iter "+str(count)+"\nLoss :"+str(loss)+" Acc :"+str(acc)+'\n')
   


