# -*- coding: utf-8 -*-
# @Time    : 2018/5/20 下午4:12
# @Author  : ZZZ
# @Email   : zuxinxing531@pingan.com.cn
# @File    : learn_001.py
# @Software: Kaggle
# @Descript: learn_001

import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt

FTRAIN = '../input/training.csv'
FTEST = '../input/test.csv'

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them
    print '-'*60
    print (df.count())

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def singleHiddenLayer(input_shape):
    model = Sequential()
    model.add(Dense(100, activation='relu',input_shape=input_shape))
    model.add(Dense(30))
    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(optimizer=sgd,loss='mse')
    return model

def cnnNet1(input_shape):
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (2, 2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (2, 2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dense(30))

    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(optimizer=sgd, loss='mse')
    return model


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 96, 96,1)
    return X, y

if __name__ == "__main__":
    model = singleHiddenLayer((9216,))
    # model = cnnNet1(input_shape = (96,96,1))
    print model.summary()

    X, y = load()
    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
        y.shape, y.min(), y.max()))

    # from sklearn.model_selection import train_test_split
    #
    # train_X, test_X, train_y, test_y = train_test_split(X,
    #                                                     y,
    #                                                     train_size=.8)
    # print '原始数据集特征：', X.shape, '训练数据集特征：', train_X.shape, '测试数据集特征：', test_X.shape
    #
    # print '原始数据集标签：', y.shape, '训练数据集标签：', train_y.shape, '测试数据集标签：', test_y.shape

    fit_history = model.fit(X,y,epochs=100,batch_size=16,verbose=1,validation_split=0.2)

    print fit_history.history
    # train_loss = np.array([i["train_loss"] for i in fit_history.history])
    # valid_loss = np.array([i["valid_loss"] for i in fit_history.history])
    plt.plot(fit_history.history['loss'], linewidth=3, label="train")
    plt.plot(fit_history.history['val_loss'], linewidth=3, label="valid")
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # plt.ylim(1e-3, 1e-2)
    plt.yscale("log")
    plt.show()

    X, _ = load2d(test=True)
    y_pred = model.predict(X)

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)

    plt.show()