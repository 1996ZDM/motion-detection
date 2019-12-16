import numpy as np
import cv2

cap = cv2.VideoCapture("D:/code/vehicle-speed-check/cars.mp4")
mog = cv2.createBackgroundSubtractorMOG2()

while (1):
    ret, frame = cap.read()
    fgmask = mog.apply(frame)
    cv2.imshow('original',frame)
    cv2.imshow('frame', fgmask)
    if cv2.waitKey(40) & 0xff == ord("q"):
        break;

cv2.destroyAllWindows()
cap.release()