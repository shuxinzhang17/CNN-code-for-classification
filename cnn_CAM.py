# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:45:47 2020
@author: Shuxin Zhang
"""
# 调用的包 ####################################################################
import numpy as np
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.io import loadmat
import h5py

from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
# from keras import regularizers
# from keras.utils import to_categorical
# from keras.metrics import categorical_accuracy
# from tensorflow.keras.models import load_model

# 加载数据，简单处理 ###########################################################
# mat = spio.loadmat(r'classification_MIC1.mat')
# NolPosData = mat['Data']
# NolPosData1 = np.array(NolPosData)
file = 'D:\\xiner\MachineLearning_Classification2021Oct\Class_hanningW10000_O60_240by1.mat'
file = 'C:\\Users\wrhsd\OneDrive\Desktop\Run\CNNclass_hanningW1000_Ovlp60_Matrix32by23.mat'
data = loadmat(file, mat_dtype=True)
NolPosData = data['Data']
NolPosData1 = np.array(NolPosData)

DATA = np.zeros((18900,32,23), dtype="float")
for i in range(0, 18899 + 1):
    T = NolPosData1[:,:,i]
    DATA[i,:,:] = T   
DATA1 = np.reshape(DATA,(18900,32,23,1))
# df = pd.read_csv('Label.csv')
# df = np.array(df)
# #df = np.load('df.npy')
labels = np.zeros((18900,9))
# #Onehot-encoding
for i in range(9):
    labels[i*2100:(i+1)*2100,i] = 1

# 将数据分为training set 和 test set #
#######################################
#split = train_test_split(df, DATA1, test_size=0.25,random_state=30)
split = train_test_split(DATA1, labels, test_size=0.25, random_state=42)
(trainImagesX, testImagesX, trainAttrX, testAttrX) = split
# normalize the label
trainY = (trainAttrX) 
testY = (testAttrX) 
###### trainImagesX, testImagesX, trainY, testY ##############
# 定义CNN结构和相关参数#####################################################
# objective = 'mean_squared_error'
objective = 'categorical_crossentropy'
optimizer = Adam(lr=1e-3, decay=1e-3/100)
def myCNN():
    # 定义你的CNN为顺序结构
    model = Sequential()
    # 16 x 3 x 3 卷积  #activation='relu' / activation=LeakyReLU(alpha=0.1)
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape= (32,23,1)))
    model.add(LeakyReLU(alpha=0.1))   
    model.add(Conv2D(16, (3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.1))
    
    # 32 x 3 x 3 卷积
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    # 64 x 3 x 3 卷积
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    # model.add(MaxPooling2D(pool_size=(2, 1)))
    # model.add(Dropout(0.3))
    
    # 128 x 3 x 3 卷积， 然后池化
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    # model.add(MaxPooling2D(pool_size=(2, 1)))
    # model.add(Dropout(0.3))
    
    # 256 x 3 x 3 卷积， 然后池化
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    # model.add(MaxPooling2D(pool_size=(2, 1)))
    # model.add(Dropout(0.5))
    
    # 展开
    # model.add(Flatten())
    # 全连接层 转成64 x 1，然后dropout
    # model.add(Dense(64))
    # model.add(LeakyReLU(alpha=0.1))     ?????????why relu after dense layer???????
    model.add(GlobalAveragePooling2D(data_format = 'channels_last'))
    model.add(Dropout(0.5))
    # 转换成 1 x 1
    model.add(Dense(9, activation='softmax'))
    print("Compiling model...")
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

print("Creating model:")
model = myCNN()
epochs = 50
batch_size = 16  # 20 / 50          # 52 #64 50 32

## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = [] 
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2, mode='auto')
       
# 定义CNN模型运行的参数，比如train和validation的比例，预测结果等 ###################
def run_myCNN():
    history = LossHistory()
    print("running model...")
    model.fit(trainImagesX, trainY, batch_size=batch_size, epochs=epochs,
              validation_data=(testImagesX, testY), verbose=2, shuffle=True, 
              callbacks=[history])   
    print("making predictions on test set...")
    predictions = model.predict(testImagesX, verbose=1)
    return predictions, history

# 输出loss的历史记录，以及测试机预测结果##########################################
predictions, history = run_myCNN()

# result = le_idx[np.argmax(predictions, axis = 1)]
# le_result = le_idx[np.argmax(y_test, axis = 1)]

loss = history.losses
loss = np.array(loss)
val_loss = history.val_losses
val_loss = np.array(val_loss)
acc = history.accuracy
acc = np.array(acc)
val_acc = history.val_accuracy
val_acc = np.array(val_acc)

# 画图#########################################################################
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,epochs)[0::20])
# plt.ylim(0, 30)
plt.xlabel('Epochs')
plt.ylabel('Categorical_crossentropy')
plt.title('Loss Trend')
plt.xlim(0, 100)
plt.ylim(0, 2.5)
plt.legend()
plt.grid(ls='--')
plt.show()

plt.axhline(y=90, color='gray', linestyle='--')
plt.plot(acc*100, 'blue', label='Training accuracy')
plt.plot(val_acc*100, 'red', label='Validation accuracy')
plt.xticks(range(0,epochs)[0::20])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('Epochs')
plt.ylabel('Score [%]')
plt.title('Accuracy Trend')
plt.legend()
plt.grid(ls='--')
plt.show()

# a=np.arange(-1,9,0.01)
# plt.plot(a,a,'r--')
# plt.scatter(testY*8,predictions*8)
# plt.xlabel('True elevation')
# plt.ylabel('Predicted elevation')
# plt.axis('scaled')
# plt.xlim(-1, 9)
# plt.ylim(-1, 9)
# sns.despine()  #去掉box
# plt.show()

##########################
model.save("D:\\xiner\\MachineLearning_Classification2021Oct\Classification_modelSave\Mymodel_CAM_W1000_Ovlp60.h5")
np.save(r'D:\xiner\MachineLearning_Classification2021Oct\Classification_modelSave\w10000o60_acc.npy',acc)
np.save(r'D:\xiner\MachineLearning_Classification2021Oct\Classification_modelSave\w10000o60_valacc.npy',val_acc)
# np.save('val_acc1.npy',val_acc)
# acc = np.load("D:\\xiner\\MachineLearning_Classification2021Oct\\Classification_modelSave\\w3000o25_acc.npy")

# model.save_weights("D:\\xiner\\MachineLearning2021Oct\\Classification_modelSave\\weights12.h5")
# load model from single file
# del model 
# model = load_model("Mymodel_w900ovlp75.h5")
# model.summary()
# # make predictions
# pred = model.predict(testImagesX, verbose=1)
# print(pred)
# val_loss, accuracy = model.evaluate(testImagesX, testY)
# print('val_loss', val_loss)
# print('val_accuracy', accuracy)

# plt.plot(accuracy*100, 'red', label='Validation accuracy')
# result = le_idx[np.argmax(pred, axis = 1)]

layer_4 = K.function([model.layers[0].input],[model.layers[-5].output])

pred = np.argmax(predictions, axis = 1)
test = np.argmax(testY, axis = 1)

for i in range(500):
    if pred[i] == test[i] and pred[i] == 0:
        example = testImagesX[i,:,:,:]
        example = np.expand_dims(example,axis = 0)
        f1 = layer_4([example])[0]
        f1 = np.squeeze(f1,axis = 0)
        weights = model.get_weights()[-2]
        # weights = np.squeeze(weights,axis = 1)
        heat_maps = np.zeros((32,23))
        for j in range(len(weights)):
            heat_maps = heat_maps + weights[j][pred[i]]*f1[:,:,j]
        heat_maps =(heat_maps-np.min(heat_maps))/(np.max(heat_maps)-np.min(heat_maps))
        plt.subplot(211)
        plt.imshow(example[0,:,:,0],cmap='jet')
        plt.title("{}th sample: Label: {}, Prediction: {} ".format(i,test[i],pred[i]))
        plt.subplot(212)
        plt.imshow(heat_maps,cmap = 'jet')
        # plt.colorbar()
        plt.title("CAM")
        # plt.tight_layout()
        plt.show()

    