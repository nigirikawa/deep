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
    # 出力フォルダを作成
    parent_path = "test1_" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
    os.mkdir(parent_path)

    input_layer = 11  # 入力層
    output_layer = 10 # 出力層
    epoch = "100"     # 学習回数
    
    # 隠れ層のノードの組み合わを作成
    # layerListは1階層のみから3階層まで含めた全組み合わせ
    network_template = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    layerList = []
    
    for i in range(1, 4):
        layerList.extend(
            itertools.product(network_template, repeat=i)
        )

    # 隠れ層のノードの組み合わせ確認用
    # pprint.pprint(layerList)

    network_out_template = [True, False]
   
    # トレーニングデータとテストデータに分割
    train_tset_data = create_test_data(parent_path)
    
    for layers in layerList:
        network = list(layers)
        
        network_out_list = list(itertools.product(network_out_template, repeat = len(layers)))

        for network_out in network_out_list:
            network_out = list(network_out)
            print(" ---- 隠れ層のノードパターン -----")
            print(network)
            print(network_out)
            
            learn(network=network, network_out=network_out,epoch=epoch, input_layer=input_layer, output_layer=output_layer,
                      parent_path=parent_path, train_and_test_data=train_tset_data) 

# テストデータとトレーニングデータを作成
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

    # 作成したニュートラルネットワークの構成を文字列形式で出力
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

# 学習結果を画像に出力
def save_history_fig(history, savePath):
    fig, (pltL, pltR) = plt.subplots(ncols=2, figsize=(10,4), sharex=True)

    # 精度の履歴をプロット
    pltL.plot(history.history['acc'])
    pltL.plot(history.history['val_acc'])
    pltL.set_title('model accuracy')
    pltL.set_xlabel('epoch')
    pltL.set_ylabel('accuracy')
    pltL.set_ylim([0.0,1.0])
    pltL.legend(['acc', 'val_acc'], loc='lower right')

    # 損失の履歴をプロット
    pltR.plot(history.history['loss'])
    pltR.plot(history.history['val_loss'])
    pltR.set_title('model loss')
    pltR.set_xlabel('epoch')
    pltR.set_ylabel('loss')
    pltR.legend(['loss', 'val_loss'], loc='lower right')

    fig.savefig(savePath + '/plot.png')

main()
print("完了しました")
