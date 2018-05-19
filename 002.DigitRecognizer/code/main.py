# -*- coding: utf-8 -*-
# @Time    : 2018/5/18 上午9:35
# @Author  : ZZZ
# @Email   : zuxinxing531@pingan.com.cn
# @File    : main.py
# @Software: Kaggle
# @Descript: main

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train = pd.read_csv("../input/train.csv")
    pred_X = pd.read_csv("../input/test.csv")

    # train = train[:1000]
    source_X = train.drop("label",axis=1)
    source_y = train.loc[:,'label']

    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(source_X,
                                                        source_y,
                                                        train_size=.8)

    print '原始数据集特征：', source_X.shape, '训练数据集特征：', train_X.shape, '测试数据集特征：', test_X.shape

    print '原始数据集标签：', source_y.shape, '训练数据集标签：', train_y.shape, '测试数据集标签：', test_y.shape

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler


    model = GradientBoostingClassifier(n_estimators=50)
    # model = LogisticRegression()
    model.fit(train_X,train_y)

    print model.score(test_X, test_y)

    pred_Y = model.predict(pred_X)
    pred_Y = pred_Y.astype(int)

    id = pd.Series([i for i in range(1,pred_Y.shape[0]+1)])
    predDf = pd.DataFrame(
        {'ImageId': id,
         'Label': pred_Y})
    print predDf.shape
    print predDf.head()
    predDf.to_csv('../output/pred.csv', index=False)