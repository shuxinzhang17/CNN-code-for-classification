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
# mat = spio.loadmat(r'classification_MIC1.mat')
# NolPosData = mat['Data']
# NolPosData1 = np.array(NolPosData)
file = 'D:\\xiner\Research2019\MachineLearning_ClassificationOneEar_2021Oct\Class_hanningW1000_O60_32by23.mat'
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
# objective = 'binary_crossentropy'
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
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    
    # 128 x 3 x 3 卷积， 然后池化
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    
    # 256 x 3 x 3 卷积， 然后池化
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    
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
    model.summary()
    return model

print("Creating model:")
model = myCNN()
epochs = 100
batch_size = 20  # 20 / 50          # 52 #64 50 32

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
       
# 定义CNN模型运行的参数，比如train和validation的比例，预测结果等###################
def run_myCNN():
    history = LossHistory()
    print("running model...")
    model.fit(trainImagesX, trainY, batch_size=batch_size, epochs=epochs,
              validation_data=(testImagesX, testY), verbose=2, shuffle=True, 
              callbacks=[history, early_stopping])   
    print("making predictions on test set...")
    predictions = model.predict(testImagesX, verbose=1)
    return predictions, history

# 输出loss的历史记录，以及测试机预测结果##########################################
predictions, history = run_myCNN()
# result = le_idx[np.argmax(predictions, axis = 1)]
# le_result = le_idx[np.argmax(y_test, axis = 1)]

loss = history.losses
loss = np.array(loss)
# Loss = np.sqrt(loss)
val_loss = history.val_losses
val_loss = np.array(val_loss)
# Val_loss = np.sqrt(val_loss)
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



######### save #############
##########################
# model.save("D:\\xiner\\MachineLearning_Classification2021Oct\Mymodel_w1000o60_16.h5")
# np.save(r'D:\xiner\MachineLearning_Classification2021Oct\w1000o60_acc17.npy',acc)
# np.save(r'D:\xiner\MachineLearning_Classification2021Oct\w1000o60_valacc17.npy',val_acc)

# np.save('testImagesX_confusion.npy',testImagesX)
# np.save('testY_confusion.npy',testY)
# np.save('predictions_confusion.npy',predictions)


# ######## Confusion matrix ######################
# from sklearn.metrics import confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# import itertools
# from matplotlib import rcParams

# model = load_model('Mymodel_cm2.h5')
# predictions = np.load('predictions_cm2.npy')
# testY = np.load('testY_cm2.npy')
# testImagesX = np.load('testImagesX_cm2.npy')
# batch_size = 20

# def plot_confusion_matrix(cm,
#                           target_names,
#                           title='Confusion matrix',
#                           # cmap=plt.cm.Greys,  # set confusion matrix color 
#                             cmap=plt.cm.Greens,
#                           normalize=True):
   
 
#     accuracy = np.trace(cm) / float(np.sum(cm))
#     misclass = 1 - accuracy

#     if cmap is None:
#         cmap = plt.get_cmap('Blues')

#     plt.figure(figsize=(15, 12)) #inch
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title,fontweight='bold')
#     plt.colorbar()

#     if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names, rotation=0)
#         plt.yticks(tick_marks, target_names)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:0.4f}".format(cm[i, j]),
#                       horizontalalignment="center",
#                       color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                       horizontalalignment="center",
#                       color="white" if cm[i, j] > thresh else "black")


#     plt.tight_layout()
#     plt.ylabel('True label',fontsize=20)
#     plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass),fontsize=20)
#     #这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
#  	#plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
#     plt.show()
# # 显示混淆矩阵
# def plot_confuse(model, x_val, y_val):
#     predictions = model.predict_classes(x_val,batch_size=batch_size)
#     truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
#     conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
#     plt.figure()
#     # plot_confusion_matrix(conf_mat, normalize=True,target_names=label,title='Confusion Matrix')
#     plot_confusion_matrix(conf_mat, normalize=False,target_names=label,title='Confusion Matrix')
# #=========================================================================================
# #最后调用这个函数即可。 test_x是测试数据，test_y是测试标签（这里用的是One——hot向量）
# #labels是一个列表，存储了你的各个类别的名字，最后会显示在横纵轴上。
# #比如这里我的labels列表
# label=['CN','CC','CO','ON','OC','OO','NN','NC','NO']
# plot_confuse(model,testImagesX,testY)



# # ######ROC曲线#########
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# DATA1, labels = load_digits(return_X_y=True)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# nb = GaussianNB()
# nb.fit(trainAttrX, trainY)
# predicted_probas = nb.predict_proba(testImagesX)
# # The magic happens here
# import matplotlib.pyplot as plt
# import scikitplot as skplt
# skplt.metrics.plot_roc(testY, predictions)
# plt.show()

################################################
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

# import numpy as np
# acc = np.load('acc_Dropout0110.npy')
# valacc = np.load('valacc_Dropout0110.npy')
# import scipy.io as io
# io.savemat('acc_Dropout0110.mat',{'acc':acc})
# io.savemat('valacc_Dropout0110.mat',{'valacc':valacc}) 

