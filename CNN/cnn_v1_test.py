from keras.models import load_model
import pandas as pd
import numpy as np
#from Classify_Training import

Prefix = 'all_exception_tsf_kjw_del_repeat_train'
File = 'all_exception_tsf_kjw_del_repeat_test'
FileName = File + '.csv'


'''读取测试集的数据'''
DataX = []
DataY = []
DataX = pd.read_csv(FileName, usecols = [1,2,3,4,5])
DataY = pd.read_csv(FileName, usecols = [6])
DataX = DataX.values.tolist()
DataY = DataY.values.tolist()
DataX = np.array(DataX)
DataY = np.array(DataY)
DataX = DataX.astype('float')
DataX = DataX / 100
X0 = DataX.shape[0]
X1 = DataX.shape[1]
DataX = DataX.reshape(X0, 1, X1, 1)

'''载入模型并预测'''
model = load_model(Prefix + '.h5')
predict = model.predict(DataX)

'''计算精确度'''
predict_to_value = np.argmax(predict,axis = 1)
print(predict_to_value)
count_right = 0
count_sum = len(predict_to_value)
for i in range(count_sum):
    if(predict_to_value[i] == DataY[i][0]):#DataY是嵌套的
        count_right = count_right + 1

accuracy = count_right / count_sum
print('accuracy:',accuracy)
resultfile = open("predict_results.dat", "w+")
for elements in predict_to_value:
    resultfile.write(str(elements)+'\n')
