import numpy as np
import cv2
im = cv2.imread('car.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
im2, contors, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contors[0]

img = cv2.drawContours(im, contors, -1, (0, 255, 0), 2)
cv2.imshow('res', img)

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
res = cv2.drawContours(im,[box],0,(0,0,255),2)
cv2.imshow('detect',res)
k=cv2.waitKey(0)
if k ==27:
    cv2.destroyAllWindows()