# from __future__ import print_function

import csv

import pandas as pd
from pandas import Series,DataFrame

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

# import keras
# from keras.datasets import fashion_mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import RMSprop
# from keras.optimizers import Adam
# from keras.utils import plot_model

import os.path

main():
#CSVファイルの読み込み
    wine_data_set = pd.read_csv("winequality-red.csv",sep=";",header=0)

    group = wine_data_set.groupby("quality")
    for key, value in group:
        if key == 3 :
            print(type(value))
            Data

    #説明変数(ワインに含まれる成分)
    x = DataFrame(wine_data_set.drop("quality",axis=1))

    #目的変数(各ワインの品質を10段階評価したもの)
    y = DataFrame(wine_data_set["quality"])

    # print(x)
    # print(y)
    # #説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
    # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.05)
    #
    # #データの整形
    # x_train = x_train.astype(np.float)
    # x_test = x_test.astype(np.float)
    #
    # y_train = keras.utils.to_categorical(y_train,10)
    # y_test = keras.utils.to_categorical(y_test,10)
    #
    # #ニューラルネットワークの実装①
    # model = Sequential()
    #
    # model.add(Dense(1000, activation='relu', input_shape=(11,)))
    # model.add(Dropout(0.2))
    #
    # model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.2))
    #
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.2))
    #
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))
    #
    # model.add(Dense(10, activation='softmax'))
    #
    # # モデルを要約するらしい。ちょっと意味がわからないです。
    # model.summary()
    #
    # model.compile(loss="mean_squared_error", optimizer=RMSprop(), metrics=["accuracy"])
    #
    # # model.load_weights("weight_1000_500_500_500_30000.txt")
    #
    # #ニューラルネットワークの学習
    # history = model.fit(x_train, y_train,batch_size=200,epochs=10000,verbose=1,validation_data=(x_test, y_test))
    # #history = model.fit(x_train, y_train,batch_size=200,epochs=30,verbose=1,validation_data=(x_test, y_test))
    #
    # #ニューラルネットワークの推論
    # score = model.evaluate(x_test,y_test,verbose=1)
    # print("\n")
    # print("Test loss:",score[0])
    # print("Test accuracy:",score[1])
    #
    # #10段階評価したいワインの成分を設定
    # #sample = [7.9, 0.35, 0.46, 5, 0.078, 15, 37, 0.9973, 3.35, 0.86, 12.8]
    # sample = [0, 0, 0, 0, 0, 0, 0, 1, 7, 0, 0]
    # print("\n")
    # print("--サンプルワインのデータ--")
    #
    # print(sample)
    #
    # #ポイント：ワインの成分をNumpyのArrayにしないとエラーが出る
    # sample = np.array(sample)
    # predict = model.predict_classes(sample.reshape(1,-1),batch_size=1,verbose=0)
    #
    # print("\n")
    # print("--予測値--")
    # print(predict)
    # print("\n")
    #
    # model_json = model.to_json();
    # file = open("model.json","w")
    # file.write(model_json)
    # file.close()
    # model.save_weights("weight_1000_100_64_32_30000.txt")
    #
    # print(model_json)
    #
    #
    #
    # #学習履歴のグラフ化に関する参考資料
    # #http://aidiary.hatenablog.com/entry/20161109/1478696865
#
# def plot_history(history):
#     # print(history.history.keys())
#
#     # 精度の履歴をプロット
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.legend(['acc', 'val_acc'], loc='lower right')
#     plt.show()
#
#     # 損失の履歴をプロット
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend(['loss', 'val_loss'], loc='lower right')
#     plt.show()

main()#
# # 学習履歴をプロット
# plot_history(history)
