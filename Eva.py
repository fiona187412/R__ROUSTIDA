import numpy as np
import time
import os
import pandas as pd

def load_data(filename):
    '''
    转换数据，将原文件中缺失值的 "?" 使用 nan 替换
    :param filename:读取文件名称
    :return: 转换后的待填补数据
    '''
    data = pd.read_csv(filename, header=None)

    data = data.replace('?', np.nan)
    missing_count = sum(data.isnull().sum())
    data = np.array(data, dtype=float)
    return data, missing_count


def org_data(filename):
    '''
    读取原始数据集
    :param filename:
    :return:
    '''
    data = pd.read_csv(filename, header=None)
    data = np.array(data, dtype=float)
    return data


def correct(result, incomplete_data, org_data, missing_count):
    '''
    填补正确率
    :param result: 使用算法填补后的结果
    :param missing_count: 缺失值个数
    :param incomplete_data : 包含缺失值的数据集
    :param org_data: 原始完整的数据集
    :return:正确率
    '''
    acc = 0  #记录正确填补个数
    nan_index = np.argwhere(np.isnan(incomplete_data))   #保存包含缺失值的索引
    for index in nan_index:
        if result[index[0], index[1]] == org_data[index[0], index[1]]:  # 使用均值填补
            acc += 1
    accuracy = acc / missing_count
    return accuracy

def CR(result, missing_count, incomplete_data):
    '''
    填补率
    :param result:
    :param missing_count:
    :param incomplete_data:
    :return:
    '''
    nan_index = np.argwhere(np.isnan(incomplete_data))  # 遍历数据集找到仍未填补的缺失值索引
    acc = 0
    # for index in nan_index:
    #     if result[index[0], index[1]].isna():
    #         acc += 1
    acc = sum(sum(np.isnan(result)))
    print("未填补的个数：", acc)

    cr = 1-(acc / missing_count)
    return cr



if __name__ == '__main__':
    complete_data = org_data("test/zoo-pre.csv")
    impute_path = 'RA/Zoo---/imp-RA-1'
    # IMPUT_NAME = 'RSIHISFA-1'
    IMPUT_NAME2 = 'RA-1'
    incomplete_path = 'DataSets/Zoo/missing-1'
    LIST = []
    List = []
    for info in os.listdir(impute_path):
        name1 = info
        domain = os.path.abspath(impute_path)
        info = os.path.join(domain, info)
        # print('读取填补后的数据集', name1)
        results = org_data(info)
        for info in os.listdir(incomplete_path):
            name2 = info
            domain = os.path.abspath(incomplete_path)
            info = os.path.join(domain, info)
            # print('读取待填补的数据集', name2)
            incomplete_data, missing = load_data(info)

            acc = correct(results, incomplete_data, complete_data, missing)
            # print("缺失值个数：", missing)
            # print("填补正确率：", acc)
            LIST.append([name1, name2, missing, acc])
    for i in (0, 8, 16, 24, 32, 40, 48):
        List.append(LIST[i])
    print(List)
    List = pd.DataFrame(List)
    List.to_csv("acc/zoo/zoo-results-" + IMPUT_NAME2, header=False, index=False)
        # result_all = pd.DataFrame(LIST[0, 8, 16,  ])
        # result_all.to_csv("acc/liver/liver-results-" + IMPUT_NAME, header=False, index=False)
        # print("-------------------------------------------------")
