# -*- coding: utf-8 -*-
# @Time    : 2018/5/17 下午4:25
# @Author  : ZZZ
# @Email   : zuxinxing531@pingan.com.cn
# @File    : main.py
# @Software: Kaggle
# @Descript: main

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fillData(pd_data):
    pd_data['Age'] = pd_data['Age'].fillna(pd_data['Age'].mean())
    pd_data['Fare'] = pd_data['Fare'].fillna(pd_data['Fare'].mean())
    pd_data['Embarked'] = pd_data['Embarked'].fillna('S')
    pd_data['Cabin'] = pd_data['Cabin'].fillna('U')
    return pd_data
if __name__ == "__main__":
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    # 将训练数据,测试数据合并
    full = train.append(test, ignore_index=True)
    fillData(full)

    # 对Sex进行0-1表示
    sex_mapDict = {'male': 1, 'female': 0}
    full['Sex'] = full['Sex'].map(sex_mapDict)

    # 对embarked数据进行one-hot编码
    embarkedDf = pd.DataFrame()
    embarkedDf = pd.get_dummies(full['Embarked'], prefix='Embarked')
    full = pd.concat([full, embarkedDf], axis=1)
    full = full.drop('Embarked', axis=1)

    # 对Pclass数据进行one-hot编码
    pclassDf = pd.DataFrame()
    pclassDf = pd.get_dummies(full['Pclass'], prefix='Pclass')
    full = pd.concat([full, pclassDf], axis=1)
    full.drop('Pclass', axis=1, inplace=True)

    # 删除'PassengerId', 'Name', 'Cabin', 'Ticket'
    # full = full.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

    # 数据相关性分析
    corrDf = full.corr()
    print corrDf['Survived'].sort_values(ascending=False)

    # 构建特征
    full_X = pd.concat([pclassDf,  # 客舱等级
                        full['Fare'],  # 船票价格
                        embarkedDf,  # 登船港口
                        full['Sex']  # 性别
                        ], axis=1)
    full_X = full.drop(['Survived','PassengerId', 'Name', 'Cabin', 'Ticket'],axis=1)

    # 拆分数据
    sourceNum = 891
    source_X = full_X.loc[0:sourceNum - 1,:]
    source_y = full.loc[0:sourceNum - 1,'Survived']

    pred_X = full_X.loc[sourceNum:,:]
    print source_X.shape,pred_X.shape

    from sklearn.model_selection import train_test_split

    train_X, test_X, train_y, test_y = train_test_split(source_X,
                                                        source_y,
                                                        train_size=.8)

    print '原始数据集特征：', source_X.shape,'训练数据集特征：', train_X.shape,'测试数据集特征：', test_X.shape

    print '原始数据集标签：', source_y.shape,'训练数据集标签：', train_y.shape,'测试数据集标签：', test_y.shape


    from sklearn.model_selection import GridSearchCV
    # 第1步：导入算法
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    # 数据归一化
    scaler = StandardScaler()
    X_scaled = scaler.fit(train_X).transform(train_X)
    test_X_scaled = scaler.fit(test_X).transform(test_X)
    print X_scaled.head()

    param_grid = {'n_estimators': [100, 120, 140, 160], 'learning_rate': [0.05, 0.08, 0.1, 0.12], 'max_depth': [3, 4]}
    grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)
    grid_search.fit(X_scaled, train_y)
    print grid_search.best_params_, grid_search.best_score_

    # 第2步：创建模型：逻辑回归（logisic regression）
    # model = LogisticRegression()
    # model = RandomForestClassifier(n_estimators=1000)
    # model = SVC()
    # model = GradientBoostingClassifier(n_estimators= 100,learning_rate)

    # 第3步：训练模型
    # model.fit(X_scaled, train_y)

    # print model.score(test_X_scaled, test_y)
    # pred_Y = model.predict(pred_X)
    # pred_Y = pred_Y.astype(int)
    #
    # passenger_id = full.loc[sourceNum:,'PassengerId']
    # predDf = pd.DataFrame(
    #     {'PassengerId': passenger_id,
    #      'Survived': pred_Y})
    # print predDf.shape
    # print predDf.head()
    # predDf.to_csv('../output/titanic_pred.csv', index=False)