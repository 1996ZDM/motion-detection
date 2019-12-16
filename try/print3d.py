import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# 构造需要显示的值
def print3d(state, counter):
    X = [x[0] for x in state]  # X轴的坐标
    Y = [x[1] for x in state]  # Y轴的坐标
    # print("x",X)
    # 设置每一个（X，Y）坐标所对应的Z轴的值，在这边Z（X，Y）=X+Y
    Z = np.zeros(shape=(len(state), 1))
    for i in range(len(state)):
        print("eeeee",str(state[i]))
        print("cconur",counter[state[i]])
        Z[i] = counter[state[i]]
    Z = [x[0] for x in Z]
    print("zzz",Z)

    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X, Y, Z, c='r', label="state")

    # 绘制图例
    ax.legend(loc='best')

    # 添加坐标轴
    ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
    ax.set_xlabel('Y', fontdict={'size': 10, 'color': 'red'})
    ax.set_xlabel('Z', fontdict={'size': 10, 'color': 'red'})

    plt.show()
