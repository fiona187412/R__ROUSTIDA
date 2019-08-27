# R__ROUSTIDA
## ROUSTIDA 代码 Roustdia.py
## RSHISFA 代码  RSHISFA.py
## DataSets 数据集
* 包含5个数据集ZOO、Acute_Inflammations、Contraceptive、Heart、Lym
** 每个数据集都包含10个子文件，表示使用10个不同随机数做种子仅对条件属性产生一定的缺失比例后的缺失数据集
## RA 
* 表示使用ROUSTIDA算法填补后的数据集
## ROUSTIDA+MIX
* 表示使用ROUSTIDA算法填补后使用混合填补方式辅助填补后的完整数据集
## RSHISFA+MIX
* 表示使用RSHISFA算法填补后使用混合填补方式辅助填补后得到的完整数据集
## acc 填补准确率
## Classification_acc 分类准确率
## SVM分类 svm-1.py
## MIX_IMPUTE 代码 MIX_IMPUTE.py
* 对两种算法填补后仍未填补的数据集，根据属性类型进行辅助填补，离散用众数、连续用均值。
## EVA.py 
* 验证填补正确率
