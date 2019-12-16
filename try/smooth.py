# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
def smooth(img):
    #img = cv2.imread('D:/code/vehicle-speed-check/tsm.jpg')  # 测试图片
    H = img.shape[0]
    W = img.shape[1]

    img3 = np.zeros((H, W, 3), np.uint8)  # 3*3邻域平滑后的图像
    imgmid = np.zeros((H, W, 3), np.uint8)  # 3*3邻域内取中值的图像

    tmpImg = np.zeros((H + 2, W + 2, 3), np.uint8)  # 扩充之后的图像
    for i in range(H):
        for j in range(W):
            tmpImg[i + 1, j + 1] = img[i, j]  # 将测试图片复制到扩充之后的图像中

    for i in range(H):
        for j in range(W):
            S = []
            for x in range(3):
                for y in range(3):  # 3*3邻域
                    # S[x * 3 + y] = tmpImg[i + x, j + y, 0]
                    S.append(tmpImg[i + x, j + y, 0])
            img3[i, j, 0] = sum(S) // 9  # 取平均值
            img3[i, j, 1] = img3[i, j, 0]
            img3[i, j, 2] = img3[i, j, 0]
            # 冒泡排序，只要排到中间一个值，即4，因此x范围是8->3
            for x in range(8, 3, -1):  # 从8减少到3，每次减少1,因为每循环一次就排好一个数
                for y in range(x):
                    if S[y + 1] > S[y]:  # 小的数往后移
                        temp = S[y]
                        S[y] = S[y + 1]
                        S[y + 1] = temp
            imgmid[i, j, 0] = S[4]  # 取中间值
            imgmid[i, j, 1] = imgmid[i, j, 0]
            imgmid[i, j, 2] = imgmid[i, j, 0]
    return imgmid


# cv2.imshow("origin",img)  # 绘制第一幅图片
#
# # 3*3邻域
#
# cv2.imshow("3.3",img3)
#
# # 邻域中值替换之后
#
# cv2.imshow("med",imgmid)
# cv2.waitKey()


