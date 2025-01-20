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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from keras import regularizers
# from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from tensorflow.keras.models import load_model

# 加载数据，简单处理 ###########################################################
mat = spio.loadmat(r'CNNclass_hanningW1024_Ovlp0_Matrix32by9.mat')
NolPosData = mat['Data']
NolPosData1 = np.array(NolPosData)
# file = 'D:\\xiner\MachineLearning_Classification2021Oct\Overlap75\CNNclass_hanningW800_Ovlp75_Matrix24by47.mat'
# data = loadmat(file, mat_dtype=True)
# NolPosData = data['Data']
# NolPosData1 = np.array(NolPosData)

DATA = np.zeros((18900,32,9), dtype="float")
for i in range(0, 18899 + 1):
    T = NolPosData1[:,:,i]
    DATA[i,:,:] = T   
DATA1 = np.reshape(DATA,(18900,32,9,1))
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
# objective = 'binary_crossentropy'
objective = 'categorical_crossentropy'
optimizer = Adam(lr=1e-3, decay=1e-3/100)
def myCNN():
    # 定义你的CNN为顺序结构
    model = Sequential()
    # 16 x 3 x 3 卷积  #activation='relu' / activation=LeakyReLU(alpha=0.1)
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape= (32,9,1)))
    model.add(LeakyReLU(alpha=0.1))   
    model.add(Conv2D(16, (3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.1))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    
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
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # 128 x 3 x 3 卷积， 然后池化
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # 256 x 3 x 3 卷积， 然后池化
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    # 展开
    model.add(Flatten())
    # 全连接层 转成64 x 1，然后dropout
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    # 转换成 1 x 1
    # model.add(Dense(1, activation='linear'))
    model.add(Dense(9, activation='softmax'))
    print("Compiling model...")
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss=objective, optimizer=optimizer, metrics=['categorical_accuracy'])
    model.summary()
    return model

print("Creating model:")
model = myCNN()
epochs = 1
batch_size = 20  # 52 #64 50 32

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

early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2, mode='auto')
       
# 定义CNN模型运行的参数，比如train和validation的比例，预测结果等###################
def run_myCNN():
    history = LossHistory()
    print("running model...")
    model.fit(trainImagesX, trainY, batch_size=batch_size, epochs=epochs,
              validation_data=(testImagesX, testY), verbose=1, shuffle=True, 
              callbacks=[history, early_stopping])   
    print("making predictions on test set...")
    predictions = model.predict(testImagesX, verbose=1)
    return predictions, history

# 输出loss的历史记录，以及测试机预测结果##########################################
predictions, history = run_myCNN()

loss = history.losses
loss = np.array(loss)
val_loss = history.val_losses
val_loss = np.array(val_loss)
acc = history.accuracy
acc = np.array(acc)
val_acc = history.val_accuracy
val_acc = np.array(val_acc)

##########################
# model.save("D:\\xiner\\MachineLearning_Classification2021Oct\\Classification_modelSave\\Mymodel_w800o75.h5")
# model.save("mymodel_w800o75.h5")
# np.save('acc1.npy',acc)
# np.save('val_acc1.npy',val_acc)
