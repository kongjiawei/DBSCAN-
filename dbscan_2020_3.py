# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:44:10 2019

@author: 孔嘉伟
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import time
from sklearn import metrics


'''从mat文件中获取数据'''
import  scipy.io as scio

DataFile = "DeleteNormal_of_DBSCAN_Simulate_2020_1_5.mat"
Data_dict = scio.loadmat(DataFile) #type is dict
Data = Data_dict['allexceptiontsfkjwdelrepeattrainsortedsimulateDBSCANFinal']
X1 = np.array(Data[0:1059,[2,3]]) # mat: 1-1059, 1059 points
print('X1:',)
Y1 = np.array(Data[0:1059,6])
X2 = np.array(Data[2420:8420,[2,3]])
Y2 = np.array(Data[2420:8420,6]) #mat 2421:8420

X = np.concatenate((X1,X2))
Y = np.concatenate((Y1,Y2))
Y = np.array(Y)

print('length of data:', len(X), len(Y), 'Y:', Y, 'len 0', len(Y[Y[:] == 0]), 'len1', len(Y[Y[:] == 1]),'len2',len(Y[Y[:] ==2]))
'''preprocessing
scaler：可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据。
'''
scaler = preprocessing.StandardScaler().fit(X)
print('original data X1:', X1, 'X2:',X2)
X = scaler.transform(X)

eps = np.arange(0.01,0.2,0.01) #eps参数从0.2开始到4，每隔0.2进行一次
min_samples=np.arange(2,20,1)#min_samples参数从2开始到20

best_Classify_right_point_rate= 0
best_Classify_right_positive_rate = 0
best_Classify_right_negative_rate = 0
best_score_eps=0
best_score_min_samples=0
best_silhouette_score = 0
silhouette_score =0

# for i in eps:
#     for j in min_samples:
#             #print('try: eps:', i, 'min_samples:', j, 'time:', time.time())
#             db = DBSCAN(eps=i, min_samples=j).fit(X)
#             labels= db.labels_#得到DBSCAN预测的分类便签
#             print("labels:", labels)
#             #计算准确率，选取最高值
#             #当数据聚类数目较少，定为异常
#             labels_unique = np.unique(labels) #删除重复项,便于检索,unique函数会从小到大排序
#             print("labels_unique:", labels_unique)
#             for m in range(1, len(labels_unique)):      #从1开始，因为第一个数据是-1，代表异常数据，不做判断
#                 if(len(labels[labels[:] == labels_unique[m]]) < len(labels) * 0.01):
#                     labels[labels[:] == labels_unique[m]] = -1
#                 # else:
#                 #     labels[labels[:] == labels_unique[m]] = 0
#             try:
#                 silhouette_score = metrics.silhouette_score(X, labels)
#                 print("silhouette_score:",silhouette_score)
#             except Exception as e:
#                 print("")
#             else:
#                 if(silhouette_score > best_silhouette_score):
#                     best_silhouette_score = silhouette_score
#                     best_score_eps = i
#                     best_score_min_samples = j
# print('best_silhouette_score:',best_silhouette_score, 'best_score_eps:',best_score_eps,'best_score_min_samples:',best_score_min_samples)
#



# if (Classify_right_negative_rate > best_Classify_right_negative_rate and Classify_right_positive_rate > 0.9): ##对positive rate做一个限定，防止出现比较极端的情况，大部分的正常点都被分为异常。
#     best_Classify_right_point_rate = Classify_right_point_rate
#     best_Classify_right_positive_rate = Classify_right_positive_rate
#     best_Classify_right_negative_rate = Classify_right_negative_rate
#     best_score_eps = i
#     best_score_min_samples = j
#
# print('best_Classify_right_point_rate:',best_Classify_right_point_rate, 'best_Classify_right_positive_rate:',best_Classify_right_positive_rate, 'best_Classify_right_negative_rate:',best_Classify_right_negative_rate,  'best_score_eps:',best_score_eps,'best_score_min_samples:',best_score_min_samples)


time1 = time.time()
y = DBSCAN(eps = 0.04, min_samples = 12, metric= 'euclidean').fit(X)
last_time = time.time() - time1
print(last_time)
print('X:',X)
plt.scatter(X[:,1], X[:,0], marker = 'o', c = Y)
plt.show()
labels = y.labels_
labels_unique = np.unique(labels)  # 删除重复项
for m in range(1, len(labels_unique)):
    if (len(labels[labels[:] == labels_unique[m]]) < len(labels) * 0.01):
        labels[labels[:] == labels_unique[m]] = -1
    else:
        labels[labels[:] == labels_unique[m]] = 0

plt.scatter(X[:,1], X[:,0], marker = 'o', c = labels)
print('y_labels:',y.labels_)
print('y_labels:',labels)

#与标签进行对比，计算准确率
for m in range(0, len(Y)):
    Y[m] = 0 if(Y[m] == 0) else -1

Classify_right_point = 0
Classify_right_positive = 0
Classify_right_negative = 0
for m in range(len(Y)):
    if(Y[m] == labels[m]):
        Classify_right_point = Classify_right_point + 1
    if(Y[m] == 0 and Y[m] == labels[m]):
        Classify_right_positive = Classify_right_positive + 1
    if (Y[m] == -1 and Y[m] == labels[m]):
        Classify_right_negative = Classify_right_negative + 1

Classify_right_point_rate = Classify_right_point/len(Y)
Classify_right_positive_rate = Classify_right_positive / len(Y[Y[:] == 0])
Classify_right_negative_rate = Classify_right_negative/ len(Y[Y[:] == -1])
print('Classify_right_point_rate:',Classify_right_point_rate, 'Classify_right_positive_rate:',Classify_right_positive_rate, ' Classify_right_negative_rate', Classify_right_negative_rate )

plt.show()
