from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random
import pandas as pd
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('tf')
OUTPUT_CLASS = 5
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)
# print(y_train.shape)
# print(x_train[0 : 2])
# print(x_train[0 : 2].shape)
# print(y_train[0 : 2])
#
# plt.imshow(x_train[0])
#
# img_x = 28
# img_y = 28
#
# x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
# x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train = x_train / 255
# x_test = x_test / 255
#
# print("y_test:", y_test)
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)
# print("y_test:", y_test)
VALIDATION_SPLIT = 0.10
Prefix = 'all_exception_tsf_kjw_del_repeat_train'
FileName = Prefix  + '.csv'

''' 读取csv文件数据,DataX为输入，DataY为标签'''
DataX = []
DataY = []
DataX = pd.read_csv(FileName, usecols=[1, 2, 3, 4, 5])
DataY = pd.read_csv(FileName, usecols=[6])
DataX = DataX.values.tolist()
DataY = DataY.values.tolist()

'''以相同的顺序打乱DataX和DataY'''
random.Random(4).shuffle(DataX)
random.Random(4).shuffle(DataY)
DataX = np.array(DataX)
X0 = DataX.shape[0]
X1 = DataX.shape[1]
DataY = np.array(DataY)
DataX = DataX.astype('float')
DataX = DataX / 100
DataY = DataY
DataX = DataX.reshape(X0, 1, X1, 1)

'''将DataY 转化为OUTPUT类别矩阵'''
DataY = np_utils.to_categorical(DataY, OUTPUT_CLASS)

print(DataX)
print(DataY)
print('DataX.shape:', DataX.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(1,1), activation = 'relu', input_shape=(1, X1, 1)))
model.add(MaxPool2D(pool_size=(1,1), strides=(1,1)))
model.add(Conv2D(64, kernel_size=(1,1), activation= 'relu'))
model.add(MaxPool2D(pool_size=(1,1), strides=(1,1)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(OUTPUT_CLASS, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(DataX, DataY, batch_size=128, epochs = 1000,verbose = 1, validation_split = VALIDATION_SPLIT)

model.save(Prefix + '.h5')
