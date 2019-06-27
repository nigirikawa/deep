from __future__ import print_function

import os
import datetime
import csv
import os.path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.utils import plot_model
from pandas import Series, DataFrame
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DEEP_NETWORK = "121-7-10"
epoch = "10000"
savePath = "dir" + DEEP_NETWORK + "--" + epoch + "--" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
os.mkdir(savePath)

def main():
    # CSVファイルの読み込み
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

    # mPrint(x_test)
    # mPrint(y_test)
    # mPrint(x_train)
    # mPrint(y_train)

    x_testCopy = x_test.copy()
    y_testCopy = y_test.copy()
    x_trainCopy = x_train.copy()
    y_trainCopy = y_train.copy()


    # データの整形
    x_train = x_train.astype(np.float)
    x_test = x_test.astype(np.float)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # ニューラルネットワークの実装①
    model = Sequential()

    nodeNumbers = DEEP_NETWORK.split("-")

    model.add(Dense(int(nodeNumbers[0]), activation='relu', input_shape=(11,)))
    model.add(Dropout(0.2))

    for nodeNumber in nodeNumbers[1:-1]:
        model.add(Dense(int(nodeNumber), activation='relu'))
        model.add(Dropout(0.2))

    model.add(Dense(int(nodeNumbers[-1]), activation='softmax'))

    # モデルを要約するらしい。ちょっと意味がわからないです。
    model.summary()

    model.compile(loss="mean_squared_error", optimizer=RMSprop(), metrics=["accuracy"])

    # model.load_weights("weight_1000_500_500_500_30000.txt")

    # ニューラルネットワークの学習
    history = model.fit(x_train, y_train, batch_size=200, epochs=int(epoch), verbose=1,
                        validation_data=(x_test, y_test))
    # history = model.fit(x_train, y_train,batch_size=200,epochs=30,verbose=1,validation_data=(x_test, y_test))

    # ニューラルネットワークの推論
    score = model.evaluate(x_test, y_test, verbose=1)
    print("\n")
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # 10段階評価したいワインの成分を設定
    # sample = [7.9, 0.35, 0.46, 5, 0.078, 15, 37, 0.9973, 3.35, 0.86, 12.8]
    sample = [0, 0, 0, 0, 0, 0, 0, 1, 7, 0, 0]
    print("\n")
    print("--サンプルワインのデータ--")

    print(sample)

    # ポイント：ワインの成分をNumpyのArrayにしないとエラーが出る
    sample = np.array(sample)
    print(type(sample))
    predict = model.predict_classes(sample.reshape(1, -1), batch_size=1, verbose=0)

    print("\n")
    print("--予測値--")
    print(predict)
    print("--正解値--")
    print(str(8))
    print("\n")

    model_json = model.to_json();
    file = open(savePath + "/model.json", "w")
    file.write(model_json)
    file.close()
    model.save_weights(savePath + "/weight.txt")

    print(model_json)

    print("-----------------------------------------------------------------------------------")
    print("全データ判定")
    print("-----------------------------------------------------------------------------------")
    result = getPredictAllDatas(model, wine_data_set)

    test = pd.concat([x_testCopy, y_testCopy], axis=1)
    train = pd.concat([x_trainCopy, y_trainCopy], axis=1)
    test.to_csv(savePath + "/test.csv", index=False)
    train.to_csv(savePath + "/train.csv", index=False)
    testResult = getPredictAllDatas(model, test)
    trainResult = getPredictAllDatas(model, test)

    result.to_csv(savePath + "/predict.csv")
    testResult.to_csv(savePath + "/testPredict.csv")
    trainResult.to_csv(savePath + "/trainPredict.csv")

    # 学習履歴をプロット
    save_history_fig(history)


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


def save_history_fig(history):
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

    # 損失の履歴をプロット
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.savefig(savePath + "/loss.png")


def mPrint(obj):
    print("type:\t" + str(type(obj)))
    print("value:\t" + str(obj))


main()
print("完了しました")
