from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import pandas as pd

def main():
    result = pd.read_csv("result_processed.csv", encoding="ms932")
    result = result.drop(columns=["ファイル名", "入力", "隠れ層1", "隠れ層2", "隠れ層3", "出力", "隠れ層の数", "loss", "acc", "val_acc"])
    group = result.groupby("directory")

    dict = {}

    for key, value in group:
        if key not in dict:
            dict[key] = value.drop(columns="directory").values.tolist()
        else :
            dict[key].append(value.values.tolist())

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    plt.figure()

    x = list(dict.keys())
    values = list(dict.values())


    for value in values:
        y = []
        z = []

        for val in sorted(value, key=lambda x: x[0]):
            y.append(val[0])
            z.append(val[1])


        plt.plot(y, z, marker=".")
    # plt.scatter(y, z, marker=".")


    plt.show()


    Y,Z = np.meshgrid(y, z)

    # print(Y)
    # print(Z)




    # x, y = x.ravel(), y.ravel()
    # dx, dy, dz = 0.1, 0.1, dict.values()[1]

    # ax.bar3d(x, y, 0, dx, dy, dz)
    # plt.show()

def test():
    x = np.arange(-3.0, 3.0, 0.1)# 1次元リスト
    y = np.arange(-3.0, 3.0, 0.1)# 1次元リスト

    X, Y = np.meshgrid(x, y)

    Z = func1(X, Y)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

    print(type(X))
    print(type(Y))
    print(type(Z))

    ax.plot_wireframe(1, 1, 1)
    plt.show()

def func1(x, y):
    return x ** 2 + y ** 2

main()
