import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import os
import pandas as pd
from sklearn.model_selection import cross_val_score

def load_data(filename):
    '''
    假设这是鸢尾花数据,csv数据格式为：
    0,5.1,3.5,1.4,0.2
    0,5.5,3.6,1.3,0.5
    1,2.5,3.4,1.0,0.5
    1,2.8,3.2,1.1,0.2
    每一行数据第一个数字(0,1...)是标签,也即数据的类别。
    '''
    data = np.genfromtxt(filename, delimiter=',')
    x = data[:, 0:-1]  # 数据特征
    y = data[:, -1].astype(int)  # 标签
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)  # 标准化
    # 将数据划分为训练集和测试集，test_size=.3表示30%的测试集
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)
    # print(x_train.shape)
    # print(y_train.shape)
    return x_std, y


def svm_c(x_train, y_train):
    # rbf核函数，设置数据权重
    svc = SVC(kernel='rbf', class_weight='balanced',)
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train)
    # 十则交叉验证准确率
    scores = cross_val_score(grid, x_train, y_train, cv=10, scoring='accuracy')
    print(scores)
    average = sum(scores)/10
    print(average)
    return average
if __name__ == '__main__':
    for i in range(2, 11):
        path = 'RSHISFA+MIX/Contraceptive/' + str(i)
        result1 = []  # 保存缺失值个数，填补时间，填补率
        for info in os.listdir(path):
            name = info
            domain = os.path.abspath(path)
            info = os.path.join(domain, info)
            print('读取数据', path, name)

            acc = svm_c(*load_data(info))

            result1.append(acc)
        result1 = pd.DataFrame(result1)
    # result1.to_csv("classification_acc/Lym/RSHISFA/-" + name, header=False, index=False)
        result1.to_csv("classification_acc/contraceptive/RSHISFA/-" + name, header=False, index=False)
        

