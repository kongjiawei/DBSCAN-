# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:29:07 2019

@author: 孔嘉伟
"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
import pandas as pd
from keras.utils import np_utils
import random
import scipy.io as scio

Prefix = 'Train_for_Anomaly'
#FileName = Prefix  + '.csv'
N_HIDDEN = 64
INPUT_SHAPE = 3
OUTPUT_CLASS = 5
BATCH_SIZE = 512

VALIDATION_SPLIT = 0.0

if (__name__ == '__main__'):
    '''
    从mat文件中获取数据
    '''


    DataFile = "DeleteNormal_of_DBSCAN_Simulate_2020_1_5.mat"
    Data_dict = scio.loadmat(DataFile) #type is dict
    Data = Data_dict['allexceptiontsfkjwdelrepeattrainsortedsimulateDBSCANFinal']
    # X1 = np.array(Data[723:1059,[2,3]]) # mat: 724-1059
    # print('X1:',)
    # Y1 = np.array(Data[723:1059,6])
    # X2 = np.array(Data[1882:2420,[2,3]])
    # Y2 = np.array(Data[1882:2420,6])
    #
    # X = np.concatenate((X1,X2))
    # Y = np.concatenate((Y1,Y2))
    # Y = np.array(Y)

    X1 = np.array(Data[723:779, [1, 2, 3]])
    X2 = np.array(Data[786:1032, [1, 2, 3]])
    X4 = np.array(Data[1882:2066, [1, 2, 3]])

    Y1 = np.array(Data[723:779, 6])
    Y2 = np.array(Data[786:1032, 6])
    Y4 = np.array(Data[1882:2066, 6])

    X = np.concatenate((X1, X2, X4))
    Y = np.concatenate((Y1, Y2, Y4))
    DataX = X.tolist()
    DataY = Y.tolist()

    '''以相同的顺序打乱DataX和DataY'''
    random.Random(4).shuffle(DataX)
    random.Random(4).shuffle(DataY)
    DataX = np.array(DataX)
    DataY = np.array(DataY)
    # DataY = DataY - 1; # start from 0; 1,2 -> 0,1
    print('DataX:',DataX)
    print("DataY:",DataY)
    print("DataX.shape:", DataX.shape)

    '''将DataY 转化为OUTPUT类别矩阵'''
    DataY = np_utils.to_categorical(DataY, OUTPUT_CLASS)



    '''TrainX.TrainY进行训练,先定义层数'''
    X_train = DataX
    Y_train = DataY
    model = Sequential()
    model.add(Dense(N_HIDDEN, input_shape = (INPUT_SHAPE, )))
    model.add(Activation('sigmoid'))
    model.add(Dense(N_HIDDEN))
    model.add(Activation('relu'))
    model.add(Dense(OUTPUT_CLASS))
    model.add(Activation('softmax'))
    model.summary()

    '''确定损失函数，开始训练'''
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
    history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE,
                        epochs = 1100, verbose = 1, validation_split = VALIDATION_SPLIT)
    output = model.predict(X_train)
    print(output)
    print(output.shape)
    model.save(Prefix + '.h5')