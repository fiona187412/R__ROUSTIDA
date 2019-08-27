import numpy as np
import pandas as pd
import time
import os
import Eva

from scipy import stats



def load_data(filename):
    '''
    读取原始数据集
    :param filename:
    :return:
    '''
    data = pd.read_csv(filename, header=None)
    data = np.array(data, dtype=float)
    return data

def impute(data, cate):
    '''
    对离散属性，布尔属性使用众数，对连续属性使用平均值填补
    :param data: 待填补数据集
    :param cate:  属性类别列表
    :return: 填补完成地数据集
    '''
    nan_index = np.argwhere(np.isnan(data))  # 遍历数据集找到仍未填补的缺失值索引
    impute = np.nanmean(data, axis=0)  # 保存每个特征的mean
    for i in cate:
        impute[i] = stats.mode(data[:, i])[0][0]  # 把categories的众数覆盖到mean中去
    for index in nan_index:
        data[index[0], index[1]] = impute[index[1]]  # 使用均值填补
    return data



if __name__=="__main__":
    # cate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15] #ZOO
    # cate = [1, 2, 3, 4, 5]   #Acute
    # cate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  #lymphography
    # cate = [1, 2, 5, 6, 8, 10, 12]   #heart
    cate = [1, 2, 4, 5, 6, 7, 8]   #Contraceptive
    path = "RSIHISFA/impute-results/contraceptive/imp-10"  # 读取数据集文件夹
    result1 = []  # 保存缺失值个数，填补时间，填补率
    print('读取数据')
    for info in os.listdir(path):
        name = info
        domain = os.path.abspath(path)
        info = os.path.join(domain, info)
        train_data = load_data(info)
        P2 = impute(train_data, cate)
        result = pd.DataFrame(P2)
        result.to_csv('RSHISFA+MIX/Contraceptive/10/(RSHISFA+MIX)' + name, header=False, index=False)  # 保存填补结果


