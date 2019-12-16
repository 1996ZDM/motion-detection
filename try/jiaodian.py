import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('d:/code/vehicle-speed-check/ttt.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,20,0.06,10)
# 返回的结果是[[ 311., 250.]] 两层括号的数组。
corners = np.int0(corners)
print(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
plt.imshow(img),plt.show()
cv2.waitKey()
