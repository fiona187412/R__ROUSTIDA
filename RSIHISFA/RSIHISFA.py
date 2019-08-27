import numpy as np
import pandas as pd
import os
import time
#======================================读取数据=====================
def load_data(filename):
    '''
    转换数据，将原文件中缺失值的 "NA" 使用 inf 替换
    :param filename:读取文件名称
    :return: 转换后的待填补数据
    '''
    data = pd.read_csv(filename, header=None)  #读取文件 csv格式
    data = data.replace(np.nan, np.inf)  #使用NA替换缺失值
    data = np.array(data)
    return data


#===============================根据决策属性划分数据集为子集============
def split_S(incomplete_data):
    '''
    划分不完备信息系统为子信息系统
    :param incomplete_data: 不完备信息系统
    :return: 子系统组成的列表
    '''
    m, n = incomplete_data.shape
    D_value = np.unique(incomplete_data[:, -1])  #找出决策属性的值域
    num_sub_S = len(D_value)    #统计需要划分几个子系统
    sub_S = [[] for _ in range(num_sub_S)]  #创建空的子系统
    for index, value in enumerate(D_value) :   #遍历决策属性值
        for i in range(m):   #遍历样本
            if incomplete_data[i, -1] == value:
                sub_S[index].extend(incomplete_data[i])   #子系统组成的list，后续使用需要转换成 array形式

    for i in range(len(sub_S)):
        length = int(len(sub_S[i])/n)
        sub_S[i] = np.array((sub_S[i])).reshape(length, n)
    return sub_S


#============================离散属性距离=======================
def vd_categoric(S, x, y, axis):
    '''
    计算样本 x y 离散属性的距离
    :param S: 当前样本所在信息系统
    :param x: 样本 x
    :param y: 样本 y
    :param axis: 离散型属性所在列
    :return: 样本 x y 离散属性距离
    '''
    inf_index = np.isinf(S)
    S[inf_index] = np.nan
    P = np.nanmax(S[:, axis])#当前信息系统离散属性列最大的取值
    if np.isnan(x[axis]) or np.isnan(y[axis]):
        distance = 1
    else:
        distance = np.abs(x[axis] - y[axis]) / (np.abs(P - 1)+1e-10)  # 离散属性距离公式， 防止分母为零 人为添加 e-10
    return distance

#===============================布尔属性距离===============================
def vd_boolean(x, y, axis):
    '''
    计算样本 x y 布尔属性的距离
    :param x: 样本 x
    :param y: 样本 y
    :param axis: 布尔属性所在列
    :return: 样本 x, y之间布尔属性距离
    '''
    if x[axis] == y[axis]:    #两个样本的当前属性值相同
        distance = 0
    else:
        distance = 1
    return distance


#=============================连续属性距离=================================
def vd_numerical(S, x, y, axis):
    '''
    计算样本 x y 连续属性的距离
    :param S: 当前样本所在信息系统
    :param x: 样本 x
    :param y: 样本 y
    :param axis: 连续属性所在列
    :return: 样本 x, y之间连续属性距离
    '''
    inf_index = np.isinf(S)
    S[inf_index] = np.nan
    sigam = np.nanstd(S[:, axis], axis=0)  #得到当前信息系统连续属性的标准差
    if np.isnan(x[axis]) or np.isnan(y[axis]):
        distance = 1
    else:
        distance = np.abs(x[axis] - y[axis]) / ((4 * sigam) + 1e-10)  #连续属性距离公式, 防止分母为零 人为添加的e-10
    return distance

#============================混合距离======================================
def HD(S, x, y, attribute_axis):
    '''
    计算样本 x y 之间的混合距离
    :param S: 当前所在信息系统
    :param x: 样本 x
    :param y: 样本 y
    :param attribute_axis: 不同类属性所在列组成的列表， 其中 0 表示布尔属性
                                                          1 表示离散属性
                                                          2 表示连续属性
                                                          3 表示决策属性
    :return: 样本 x y 混合距离
    '''
    distance_temp = []  #初始一个一维的 保存每个属性的距离,用于求混合距离
    for index, value in enumerate(attribute_axis):  #遍历属性类别列表,根据属性类别计算相应距离
        if value == 0:   #布尔属性
            distance_temp.append(vd_boolean(x, y, index))
        elif value == 1:   #离散属性
            distance_temp.append(vd_categoric(S, x, y, index))
        elif value == 2:       #连续属性
            distance_temp.append(vd_numerical(S, x, y, index))
        elif value == 3:  #决策属性距离为0 ,不纳入考虑范围内
            distance_temp.append(0)
    distance = np.sqrt(sum(np.square(distance_temp)))

    return distance

#=================================距离矩阵=============================
def D_matrix(S, att_axis, Mas, Mos):
    '''
    计算距离矩阵 D
    :param data: 不完备信息系统
    :return: 距离矩阵 D
    '''
    m, n = S.shape
    D = np.zeros((m, m))  #初始化空的D矩阵,大小为 m x m
    for i in range(m):  # 遍历整个数据集
         for row in range(1, m): #从第二个样本开始遍历
             dist = HD(S, S[i], S[row], att_axis)
             D[i][row] = dist
             D[row][i] = D[i][row]
         D[i][i] = 100    #对角线令为-1 方便区分
    for index, value in enumerate(Mos):   #筛选不满足条件的相似对象
      for index2, values2 in enumerate(Mos[1:], start=1):
          if Mas[index] in Mas[index2]:
              D[value][values2] = 100
          elif Mas[index2] in Mas[index]:
              D[values2][value] = 100
    return D

#=================================缺失对象集合===========================
def MOS(S):
    '''
    计算信息系统S的遗失对象集
    :return: 存在缺失对象的样本序号（index）组成的集合
    '''
    Mos = np.unique(np.where(np.isinf(S))[0])
    Mos = list(Mos)
    return Mos

#================================缺失对象的缺失属性集合========================
def MAS(S, MOS):
    '''
    计算对象 Xi的遗失属性集
    :return: 缺失对象的缺失属性集合
    '''
    mas = []
    for i in MOS:
        mas.append(np.unique(np.where(np.isinf(S[i]))))
    return mas

#===============================产生新的填补后信息系统=========================
def generate_S(S, Mos, Mas, D):
    '''
    进行填补训练
    :return:
    '''
    for index, value in enumerate(Mos):  # 遍历缺失对象集得到对象索引和对应的值
        Min = np.argmin(D[value])     #返回最小值所在的位置
        mas_index = Mas[index]   #记录当前缺失对象的缺失属性
        S[value, mas_index] = S[Min, mas_index]                  #填补缺失值为最小值所对应样本的值
    return S

def train(S):
    '''

    :param S:
    :param Mos:
    :param Mas:
    :param D:
    :return:
    '''
    sub_S = split_S(S)
    for i in range(len(sub_S)):
        Mos = MOS(sub_S[i])
        Mas = MAS(sub_S[i], Mos)
        D = D_matrix(sub_S[i], att_axis, Mas, Mos)
        sub_S[i] = generate_S(sub_S[i], Mos, Mas, D)
    SSS = np.vstack((sub_S[i] for i in range(len(sub_S))))
    return SSS

if __name__ == '__main__':
    att_axis = [1, 1, 1, 1, 3]  #属性类别列表
    path = "datasets"  # 读取数据集文件夹
    result1 = []  # 保存缺失值个数，填补时间，填补率
    for info in os.listdir(path):
        name = info
        domain = os.path.abspath(path)
        info = os.path.join(domain, info)
        print('读取数据')
        DATA = load_data(info)
        SSS = train(DATA)
        print(SSS)
        result = pd.DataFrame(SSS)
        result.to_csv("impute-results/(NEW)-" + name, header=False, index=False)
    # S = sub_S[0]
    # x = S[1]
    # y = S[2]
    # print("x:", x)
    # print("y:", y)
    # d1 = vd_boolean(x, y, 3)
    # print("布尔距离:", d1)
    #
    # d2 = vd_categoric(S, x, y, 2)
    # print("离散距离:", d2)
    #
    # w = S[:, 2]
    # print(w)
    # mmm = np.array([np.nan, 0, 0])
    # print("mmmmm", np.nanmax(mmm))
    # print(max(w))
    #
    # ddd = vd_categoric(S, x, y, 0)
    # print("dddd", ddd)
    #
    # d3 = vd_numerical(S, x, y, 1)
    # print("连续距离:", d3)
    # #

    #
    # att_axis = [1, 2, 1, 0, 3]
    # hd = HD(S, x, y, att_axis)
    # print("混合距离:", hd)
    #
    # # att_axis = [1, 2, 1, 0, 3]
    # Mos = MOS(S)
    # print("MOS:", Mos)
    #
    # Mas = MAS(S, Mos)
    # print("MAS:", Mas)
    # #
    # D = D_matrix(S, att_axis, Mas, Mos)
    # print(D)
    #
    # S1 = generate_S(S, Mos, Mas, D)
    # print(S1)

    # sub_S = split_S(S)
    # for i in range(len(sub_S)):
    #     print("子信息系统", i + 1)
    #     print(sub_S[i])
    #
    #     Mos = MOS(sub_S[i])
    #     print("子系统", i + 1, "的缺失对象集合MOS: ", Mos)
    #     Mas = MAS(sub_S[i], Mos)
    #     print("子系统", i + 1, "缺失对象的缺失属性集合MAS：", Mas)
    #
    #     D = D_matrix(sub_S[i], att_axis, Mas, Mos)
    #     print("子系统", i + 1, "距离矩阵D: ")
    #     print(D)
    #     print("填补后的信息系统----------------------------------------------------------------------------")
    #     S1 = generate_S(sub_S[i], Mos, Mas, D)
    #     print(S1)
    #     print("----------------------------------------------------------------------------")