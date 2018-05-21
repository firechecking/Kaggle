# -*- coding: utf-8 -*-
# @Time    : 2018/5/18 下午5:11
# @Author  : ZZZ
# @Email   : zuxinxing531@pingan.com.cn
# @File    : main_vgg.py
# @Software: Kaggle
# @Descript: main_vgg

import tensorflow as tf
import numpy as np
import pandas as pd

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers.core import Dense,Flatten
from keras.models import Sequential
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def nn_base(input,trainable):
    base_model = VGG16(include_top=False,input_shape = input)
    for layer in base_model.layers:
        layer.trainable = trainable
    return base_model.input,base_model.get_layer('block5_conv3').output

def fc_net(vgg16_output):
    x = Flatten()(vgg16_output)
    x = Dense(32,activation='relu')(x)
    x = Dense(10,activation='softmax')(x)
    return x

def loadCSVfile(file):
    tmp = np.loadtxt(file, dtype=np.str, delimiter=",")
    data = tmp[1:,1:].astype(np.float)#加载数据部分
    label = tmp[1:,0].astype(np.float)#加载类别标签部分
    return data, label #返回array类型的数据

def defineModel(input_shape):
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(30, activation='relu', name='fc1')(x)
    # x = Dense(100, activation='relu', name='fc2')(x)
    x = Dense(10, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='vgg16')
    return model

if __name__ == "__main__":
    # inp,vgg16_output = nn_base((56, 56, 3),trainable=False)
    # x_categories = fc_net(vgg16_output)
    #
    # model = Model(inputs=inp, outputs=x_categories)  # basemodel共28层
    # print model.summary()

    type = "run"

    # a = np.array([[1, 5, 5, 2],
    #               [9, 6, 2, 8],
    #               [3, 7, 9, 1]])
    # print(np.argmax(a, axis=1))

    if type == "run":
        test = np.loadtxt('../input/test.csv', dtype=np.str, delimiter=",")
        test = test[1:, :].astype(np.float)  # 加载数据部分
        test = test.reshape(-1, 28, 28, 1)
        print test.shape
    else:
        source_X, source_y = loadCSVfile('../input/train.csv')
        source_X = source_X.reshape(-1, 28, 28, 1)

        from sklearn.model_selection import train_test_split
        train_X, test_X, train_y, test_y = train_test_split(source_X,
                                                            source_y,
                                                            train_size=.8)
        from keras.utils.np_utils import to_categorical

        train_y = to_categorical(train_y, num_classes=10)
        test_y = to_categorical(test_y,num_classes=10)
        print '原始数据集特征：', source_X.shape, '训练数据集特征：', train_X.shape, '测试数据集特征：', test_X.shape

        print '原始数据集标签：', source_y.shape, '训练数据集标签：', train_y.shape, '测试数据集标签：', test_y.shape

    model = defineModel((28,28,1))
    print model.summary()

    model.load_weights('../output/weights-cnn-4-49.hdf5')
    checkpoint = ModelCheckpoint('../output/weights-cnn-4-{epoch:02d}.hdf5',
                                 save_weights_only=True)
    adam = Adam(lr=0.00001)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if type == "run":
        result = model.predict(test,batch_size=1)
        result = np.argmax(result,axis=1)
        print result

        predDf = pd.DataFrame(
            {'ImageId': [i for i in range(1,result.shape[0]+1)],
             'Label': result})
        print predDf.shape
        print predDf.head()
        predDf.to_csv('../output/pred_vgg.csv', index=False)
    else:
        model.fit(train_X, train_y,epochs=50, batch_size=32,callbacks=[checkpoint])  # starts training
        score = model.evaluate(test_X, test_y, batch_size=128)
        print score


    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 22
    # s - loss: 4.3394 - acc: 0.7015
    # Epoch
    # 2 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 23
    # s - loss: 0.0714 - acc: 0.9781
    # Epoch
    # 3 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 23
    # s - loss: 0.0444 - acc: 0.9860
    # Epoch
    # 4 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 23
    # s - loss: 0.0376 - acc: 0.9882
    # Epoch
    # 5 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 23
    # s - loss: 0.0327 - acc: 0.9900
    # Epoch
    # 6 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 23
    # s - loss: 0.0296 - acc: 0.9907
    # Epoch
    # 7 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 23
    # s - loss: 0.0235 - acc: 0.9929
    # Epoch
    # 8 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 23
    # s - loss: 0.0262 - acc: 0.9919
    # Epoch
    # 9 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 25
    # s - loss: 0.0190 - acc: 0.9943
    # Epoch
    # 10 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 25
    # s - loss: 0.0223 - acc: 0.9933
    # Epoch
    # 11 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 24
    # s - loss: 0.0174 - acc: 0.9947
    # Epoch
    # 12 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 24
    # s - loss: 0.0206 - acc: 0.9944
    # Epoch
    # 13 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 24
    # s - loss: 0.0160 - acc: 0.9958
    # Epoch
    # 14 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 24
    # s - loss: 0.0215 - acc: 0.9942
    # Epoch
    # 15 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 24
    # s - loss: 0.0145 - acc: 0.9962
    # Epoch
    # 16 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 24
    # s - loss: 0.0153 - acc: 0.9960
    # Epoch
    # 17 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 28
    # s - loss: 0.0136 - acc: 0.9963
    # Epoch
    # 18 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 36
    # s - loss: 0.0215 - acc: 0.9948
    # Epoch
    # 19 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 41
    # s - loss: 0.0135 - acc: 0.9969
    # Epoch
    # 20 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 32
    # s - loss: 0.0200 - acc: 0.9959
    # Epoch
    # 21 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 40
    # s - loss: 0.0134 - acc: 0.9971
    # Epoch
    # 22 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 40
    # s - loss: 0.0160 - acc: 0.9970
    # Epoch
    # 23 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 38
    # s - loss: 0.0163 - acc: 0.9964
    # Epoch
    # 24 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 39
    # s - loss: 0.0142 - acc: 0.9969
    # Epoch
    # 25 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 38
    # s - loss: 0.0137 - acc: 0.9972
    # Epoch
    # 26 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 36
    # s - loss: 0.0126 - acc: 0.9978
    # Epoch
    # 27 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 32
    # s - loss: 0.0344 - acc: 0.9951
    # Epoch
    # 28 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 35
    # s - loss: 0.0201 - acc: 0.9966
    # Epoch
    # 29 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 33
    # s - loss: 0.0214 - acc: 0.9970
    # Epoch
    # 30 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 33
    # s - loss: 0.0179 - acc: 0.9973
    # Epoch
    # 31 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 31
    # s - loss: 0.0279 - acc: 0.9963
    # Epoch
    # 32 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 31
    # s - loss: 0.0260 - acc: 0.9968
    # Epoch
    # 33 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 30
    # s - loss: 0.0323 - acc: 0.9965
    # Epoch
    # 34 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 30
    # s - loss: 0.0329 - acc: 0.9963
    # Epoch
    # 35 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 30
    # s - loss: 0.0283 - acc: 0.9971
    # Epoch
    # 36 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0274 - acc: 0.9967
    # Epoch
    # 37 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0274 - acc: 0.9971
    # Epoch
    # 38 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 30
    # s - loss: 0.0330 - acc: 0.9966
    # Epoch
    # 39 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0395 - acc: 0.9964
    # Epoch
    # 40 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0365 - acc: 0.9967
    # Epoch
    # 41 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0379 - acc: 0.9964
    # Epoch
    # 42 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0440 - acc: 0.9962
    # Epoch
    # 43 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0468 - acc: 0.9961
    # Epoch
    # 44 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0513 - acc: 0.9957
    # Epoch
    # 45 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0358 - acc: 0.9968
    # Epoch
    # 46 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 31
    # s - loss: 0.0588 - acc: 0.9957
    # Epoch
    # 47 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0501 - acc: 0.9963
    # Epoch
    # 48 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0442 - acc: 0.9965
    # Epoch
    # 49 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0520 - acc: 0.9962
    # Epoch
    # 50 / 50
    # 33600 / 33600[ == == == == == == == == == == == == == == ==] - 29
    # s - loss: 0.0644 - acc: 0.9953
    # 8320 / 8400[ == == == == == == == == == == == == == == >.] - ETA: 0
    # s[0.29800680641856159, 0.98011904761904767]