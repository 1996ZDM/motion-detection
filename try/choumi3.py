#!/usr/bin/python
# coding:utf8

import numpy as np
import cv2

step=10

if __name__ == '__main__':
    cam = cv2.VideoCapture('D:/code/vehicle-speed-check/data/mobilecars.avi')
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用Gunnar Farneback算法计算密集光流
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        # 绘制线
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)

        line = []
        for l in lines:
            if l[0][0]-l[1][0]>3 or l[0][1]-l[1][1]>3:
                line.append(l)

        cv2.polylines(img, line, 0, (0,255,255))
        cv2.imshow('flow', img)

        ch = cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()