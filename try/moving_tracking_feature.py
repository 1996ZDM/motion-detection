'''
function:


input:tracked optical point,include forebackground and background
feature point

output: state vector s(i,k)=(L^2,tan0)
when K frame,state of eature point i
L^2 = (xk-xk-1)^2+(yk-yk-1)^2
tan0 = (yk-yk-1)/(xk-xk-1)
from collections import Counter

'''
from decimal import Decimal
from decimal import getcontext
from collections import Counter
import print3d as pd
getcontext().prec = 5


# 计算每一帧的特征点的状态向量
def subtract_background(track):
    state = []
    for item in track:
        if not len(item) > 1:
            continue
        else:
            x, y = item[-1]
            z, s = item[-2]
            x_z = x - z
            if not x_z:
                x_z = 0.000001
            L = Decimal(str(pow((x - z), 2))) + Decimal(str(pow((y - s), 2)))
            tan = Decimal(str(y - s)) / Decimal(str(x_z))
            state.append((L, tan))
    print("state",state)
    counter = Counter(state)
    print("state", counter)
    pd.print3d(state,counter)
    return state
