from __future__ import print_function

import csv
import datetime
import os
import os.path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import copy
import itertools
import pprint

from keras.datasets import fashion_mnist
from keras.layers import InputLayer, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.utils import plot_model
from pandas import Series, DataFrame
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def main():
    network_template = {
        3:[None, "1", "2", "4", "8", "16", "32", "64", "128", "256"],  # 3層目
        2:[None, "1", "2", "4", "8", "16", "32", "64", "128", "256"],  # 2層目
        1:["1", "2", "4", "8", "16", "32", "64", "128", "256"],# 1層目
    }

    input_layer = 11
    output_layer = 10
    epoch = "10000"
    parent_path = "test1" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
    os.mkdir(parent_path)

    train_tset_data = create_test_data(parent_path)

    for layer3 in network_template[3]:
        for layer2 in network_template[2]:
            for layer1 in network_template[1]:
                network = ""
                if layer3 is None:
                    if layer2 is None:
                        network = [layer1]
                    else:
                        network = [layer1, layer2]
                else:
                    if layer2 is None:
                        continue
                    else:
                        network = [layer1, layer2, layer3]

                print(network)
                learn(network=network, epoch=epoch, input_layer=input_layer, output_layer=output_layer,
                  parent_path=parent_path, train_and_test_data=train_tset_data)

def create_test_data(save_path):
    wine_data_set = pd.read_csv("sorted-redwine-data.csv", header=0)

    x_test = DataFrame()
    y_test = DataFrame()
    x_train = DataFrame()
    y_train = DataFrame()

    # 分類に偏りがあるので、各評価ごとにテストデータとトレーニングデータを抽出
    group = wine_data_set.groupby("quality")
    for key, value in group:
        vx = DataFrame(value.drop("quality", axis=1))
        vy = DataFrame(value["quality"])
        vx_train, vx_test, vy_train, vy_test = train_test_split(vx, vy, test_size=0.05)
        x_train = pd.concat([x_train, vx_train])
        y_train = pd.concat([y_train, vy_train])
        x_test = pd.concat([x_test, vx_test])
        y_test = pd.concat([y_test, vy_test])

    x_testCopy = x_test.copy()
    y_testCopy = y_test.copy()
    x_trainCopy = x_train.copy()
    y_trainCopy = y_train.copy()

    test = pd.concat([x_testCopy, y_testCopy], axis=1)
    train = pd.concat([x_trainCopy, y_trainCopy], axis=1)
    test.to_csv(save_path + "/test.csv", index=False)
    train.to_csv(save_path + "/train.csv", index=False)

    return x_train, y_train, x_test, y_test

def learn(network, epoch, input_layer, output_layer,train_and_test_data, parent_path="."):
    x_train, y_train, x_test, y_test = train_and_test_data

    # CSVファイルの読み込み
    savePath = parent_path + "/dir" + str(input_layer) + "-" + "-".join(network) + "-" + str(output_layer) + "--" + epoch
    os.mkdir(savePath)

    # データの整形
    x_train = x_train.astype(np.float)
    x_test = x_test.astype(np.float)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # ニューラルネットワークの実装①
    model = Sequential()

    nodeNumbers = network.copy()

    # 一個目の隠れ層と入力層
    model.add(Dense(int(nodeNumbers[0]), activation='relu', input_dim=input_layer))
    model.add(Dropout(0.2))

    # 二個目以降の隠れ層
    for nodeNumber in nodeNumbers[1:-1]:
        model.add(Dense(int(nodeNumber), activation='relu'))
        model.add(Dropout(0.2))

    # 出力層
    model.add(Dense(output_layer, activation='softmax'))

    # モデルを要約するらしい。ちょっと意味がわからないです。
    model.summary()

    # Early Stopping のコールバック作成
    es = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=1000, mode="min")
    fpath = savePath+'/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
    mc = ModelCheckpoint(filepath=fpath, monitor='val_loss', save_best_only=True, mode='auto')

    model.compile(loss="mean_squared_error", optimizer=RMSprop(), metrics=["accuracy"])

    # ニューラルネットワークの学習
    history = model.fit(x_train, y_train, batch_size=500, epochs=int(epoch), verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[es, mc])
    # history = model.fit(x_train, y_train,batch_size=200,epochs=30,verbose=1,validation_data=(x_test, y_test))

    # ニューラルネットワークの推論
    score = model.evaluate(x_test, y_test, verbose=1)
    print("\n")
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # testResult = getPredictAllDatas(model, test)
    # trainResult = getPredictAllDatas(model, test)
    #
    # result.to_csv(savePath + "/predict.csv")
    # testResult.to_csv(savePath + "/testPredict.csv")
    # trainResult.to_csv(savePath + "/trainPredict.csv")

    # 学習履歴をプロット
    save_history_fig(history, savePath)
    print(str(network) + " 学習完了")


def getPredictAllDatas(model, dataFrameDatas):
    label = dataFrameDatas.columns.values
    datas = DataFrame(dataFrameDatas.drop("quality", axis=1))
    predictList = []
    for data in datas.values.tolist():
        array = np.array(data)
        predict = model.predict_classes(array.reshape(1, -1), batch_size=1, verbose=0).tolist()
        predictList = predictList + predict

    result = dataFrameDatas.copy()
    result["predict"] = predictList
    return result

    # 学習履歴のグラフ化に関する参考資料
    # http://aidiary.hatenablog.com/entry/20161109/1478696865


def save_history_fig(history, savePath):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(savePath + "/acc.png")
    plt.close()

    # 損失の履歴をプロット
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.savefig(savePath + "/loss.png")
    plt.close()


def mPrint(obj):
    print("type:\t" + str(type(obj)))
    print("value:\t" + str(obj))


main()
print("完了しました")
