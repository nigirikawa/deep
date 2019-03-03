from __future__ import print_function

import pandas as pd
from pandas import Series, DataFrame

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam

#CSVファイルの読み込み
wine_data_set = pd.read_csv("../data/winequality-red.csv", sep=";", header=0)

# dropは行列の削除を行う、axis 0 の時行削除、1の時列削除
# x: 説明変数
# y: 目的変数
x = DataFrame(wine_data_set.drop("quality", axis=1))
y = DataFrame(wine_data_set["quality"])

# 学習用データとテストデータを分ける
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.05)

#データ整形
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test,10)


# ニューラルネットワーク実装
model = Sequential()

# 層の追加
# Denseは全結合の層
# units 出力の数
# activation 活性化関数
# input_shape 最初の層のみ指定、入力層の形指定
model.add(Dense(500, activation="relu", input_shape=(11,)))

# 二層目
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))

# 三層目
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))

# 出力層
model.add(Dense(10, activation='softmax'))

# 過学習を防ぐ為にドロップアウト
model.add(Dropout(0.2))
model.summary()

# コンパイル　オプティマイザとか色々指定。ここ後で調べなきゃダメ
model.compile(loss='mean_squared_error',optimizer=RMSprop(),metrics=['accuracy'])

# 学習 epochが学習回数
history =  model.fit(x_train, y_train, batch_size=200, epochs=5000, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)

print("¥n")
print("Test loss:", score[0])
print("Test accuracy", score[1])


# 評価するワインの成分
sample = [7.9, 0.35, 0.46, 5, 0.078, 15, 37, 0.9973, 3.35, 0.86, 12.8]

print("¥n")
print("--サンプルワインのデータ")

print(sample)

# kerasの引数の型にあわせるために sampleをnumpyのArrayに変換
sample = np.array(sample)
predict = model.predict_classes(sample.reshape(1, -1), batch_size=1, verbose=0)

print("¥n")
print("--予測値--")
print(predict)
print("¥n")


# 学習履歴の出力
def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()

# 学習履歴をプロット
plot_history(history)