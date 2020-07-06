# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:17:51 2019

@author: 孔嘉伟
"""

from keras.models import load_model
import pandas as pd
import numpy as np
import scipy.io as scio
#from Classify_Training import 

Prefix = 'Train_for_Anomaly'
# FileName = Prefix + '.csv'

'''读取测试集的数据'''
# DataX = []
# DataY = []
# DataX = pd.read_csv(FileName, usecols = [1,2,3,4,5])
# DataY = pd.read_csv(FileName, usecols = [6])
# DataX = DataX.values.tolist()
# DataY = DataY.values.tolist()
DataFile = "DeleteNormal_of_DBSCAN_Simulate_2020_1_5.mat"
Data_dict = scio.loadmat(DataFile)  # type is dict
Data = Data_dict['allexceptiontsfkjwdelrepeattrainsortedsimulateDBSCANFinal']
# '''train'''
# X1 = np.array(Data[723:779, [2, 3]])
# X2 = np.array(Data[786:1032, [2, 3]])
# X4 = np.array(Data[1882:2366, [2, 3]])
#
# Y1 = np.array(Data[723:779, 6])
# Y2 = np.array(Data[786:1032, 6])
# Y4 = np.array(Data[1882:2366, 6])

'''test'''
X1 = np.array(Data[779:786, [1,2, 3]])
X2 = np.array(Data[1032:1059, [1,2,3]])
X4 = np.array(Data[2366:2400, [1,2,3]])

Y1 = np.array(Data[779:786, 6])
Y2 = np.array(Data[1032:1059, 6])
Y4 = np.array(Data[2366:2400, 6])

X = np.concatenate((X1, X2, X4))
Y = np.concatenate((Y1, Y2, Y4))
DataX = X.tolist()
DataY = Y.tolist()

DataX = np.array(DataX)
DataY = np.array(DataY)

'''载入模型并预测'''
model = load_model(Prefix + '.h5')
predict = model.predict(DataX)

'''计算精确度'''
predict_to_value = np.argmax(predict,axis = 1)
print('predict_to_value:', predict_to_value)
print('DataY:',DataY )
count_right = 0
count_sum = len(predict_to_value)
for i in range(count_sum):
    if(predict_to_value[i] == DataY[i]):#DataY是嵌套的
        count_right = count_right + 1

accuracy = count_right / count_sum
print('accuracy:',accuracy)
resultfile = open("predict_results.dat", "w+")
for elements in predict_to_value:
    resultfile.write(str(elements)+'\n')






 