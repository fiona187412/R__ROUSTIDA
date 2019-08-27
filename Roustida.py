import numpy as np
import pandas as pd
import time
import os
import Eva

def load_data(filename):
    '''
    转换数据，将原文件中缺失值的 "?" 使用 nan 替换
    :param filename:读取文件名称
    :return: 转换后的待填补数据
    '''
    data = pd.read_csv(filename, header=None)  #读取文件 csv格式
    data = data.replace(np.nan, np.inf)  #使用NA替换缺失值
    missing_count = np.isinf(data).sum()  #统计缺失值个数
    data = np.array(data, dtype=float)
    return data, missing_count


def m_matrix(data):
    '''
    计算 扩充差异矩阵M
    :param data:待填补的数据
    :return: 扩充可分辨 M 矩阵
    '''
    M = [[[] for _ in range(data.shape[0])] for _ in range(data.shape[0])]   #初始化空的M矩阵
    for i in range(data.shape[0]):  # 遍历整个数据集
        for row in range(1, data.shape[0]): #从第二个样本开始遍历
            value_set = []  #初始矩阵M中的元素为集合
            for j in range(data.shape[1]):  # 遍历整个属性集
                M[i][row] = value_set
                if ((data[i][j] != data[row][j]) and (data[i][j] != np.inf) and (
                        data[row][j] != np.inf)):  # 寻找符合M矩阵定义的下标
                    value_set.append(j)
            M[i][row] = value_set
        M[i][i] = [-1]  # 对角线令为-1 方便区分
        for p in range(data.shape[0]):
            if p < i:
                M[i][p] = [-0.12345]  # 对M矩阵的下三角令为-0.12345 方便区别
    return M

def MOS(incomple_data):
    '''
    计算信息系统S的遗失对象集
    :return: 存在缺失对象的样本序号（index）组成的集合
    '''
    Mos = np.unique(np.where(np.isinf(incomple_data))[0])
    Mos = list(Mos)
    return Mos

def MAS(incomple_data, MOS):
    '''
    计算对象 Xi的遗失属性集
    :return: 缺失对象的缺失属性集合
    '''
    mas = []
    for i in MOS:
        mas.append(np.unique(np.where(np.isinf(incomple_data[i]))))
    return mas


def NS(M_matrix, MOS):
    '''
    计算对象 Xi 的无差别对象集
    :return:

    '''
    ns = [[] for _ in range(len(MOS))]
    l = MOS  # 记录MOS的序号
    for index, value in enumerate(l):
        for j in range(train_data.shape[0]):
            if M_matrix[value][j] == [] or M_matrix[j][value] == []:
                ns[index].append(j)
    return ns

def generate_S(S):
    '''
    进行填补训练
    :return:
    '''
    Mos = MOS(S)  #初始化第一次的缺失对象集合
    Mas = MAS(S, Mos)   #初始化第一次的缺失对象的缺失属性集合
    M = m_matrix(S)   #初始化扩充差异矩阵
    ns = NS(M, Mos)   #计算缺失对象的无差异对象集合
    S2 = S.copy()       #复制一个不完备信息表，后续查找缺失值使用
    for index, value in enumerate(Mos):  # 遍历遗失对象集得到对象索引和对应的值
        for k in Mas[index]:  # 遍历当前对象的遗失属性集中的属性
            m = list(np.unique(S2[ns[index], k]))  #得到原本不完备信息系统的无差别对象的同一属性取值去掉重复的
            if len(m) == 1:   #如果当前无差别对象取值相同
                if m[0] == np.inf:     #无差别对象的取值均为 *
                    S[value, k] == np.inf   #则原信息系统修改缺失值为 *
                    print("对象", value + 1, "属性", k + 1, "填补值为空，填补失败！")
                else:
                    S[value, k] = m[0]   #若无差别对象取值不为 *，则赋值为当前值
                    print("对象", value + 1, "属性", k + 1, "填补成功！")
            else:
                if len(m) > 2:   #如果无差别对象取值不唯一，大于2种以上
                    S[value, k] == np.inf      #修改缺失值为 *
                    print("对象", value + 1, '属性', (k + 1), '存在多个填补值，填补失败！')
                elif np.inf in m and len(m) == 2: #无差别对象取值只有两种，一个为空一个不为空
                    S[value, k] = m[0]    #赋值为不为空的取值
                    print("对象", value + 1, "属性", k + 1, "填补成功...")
                else:
                    S[value, k] == np.inf    #其他情况，均填补为 *
                    print("对象", value + 1, '属性', (k + 1), '填补失败！')

    return S

def train(train_data):
    flag = True      #设置修改标志
    while(flag):      #循环判定条件
        Second = generate_S(train_data)    #得到填补后的第二个信息系统
        if (Second == train_data).all():   #如果新的信息系统和上一次没有发生变化
            flag = False    #修改标志为 false
        else:
            train_data = Second    #使用当前的再次进行训练
    Second = pd.DataFrame(Second)
    Second = np.array(Second.replace(np.inf, np.nan), dtype=float)
    return Second

if __name__ == '__main__':
    start = time.time()      #算法运行开始计时
    path = "DataSets/Zoo/missing-1"   #读取数据集文件夹
    result1 = []   #保存缺失值个数，填补时间，填补率
    for info in os.listdir(path):
        name = info
        domain = os.path.abspath(path)
        info = os.path.join(domain, info)
        print('读取数据')
        train_data, missing_count = load_data(info)
        # print("缺失个数:", missing_count)
        # m = missing_count / (train_data.shape[0] * train_data.shape[1])
        # print("缺失率：", m)
        st = time.time()
        P2 = train(train_data)
        enen = time.time()
        print(name, "所花费时间：", enen - st)
        print('填补完成')
        result = pd.DataFrame(P2)
        result.to_csv('RA/Zoo---/imp-RA-1/(RA)' + name, header=False, index=False)    #保存填补结果
        end = time.time()
        print("填补时间：", end - start)
        cr = Eva.CR(P2, missing_count, train_data)
        print("填补率：", cr)
        result1.append([cr, end - start])
    result_all = pd.DataFrame(result1)
    result_all.to_csv("RA/Zoo---/(RA)-results-" + name, header=False, index=False)



