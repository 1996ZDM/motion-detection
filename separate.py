import numpy as np
import cv2

# read the video
cap = cv2.VideoCapture('cars.mp4')

# create the subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=100, detectShadows=False)


def getPerson(image, opt=1):

    # get the front mask
    mask = fgbg.apply(frame)

    # eliminate the noise
    line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5), (-1, -1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, line)
    cv2.imshow("mask", mask)

    # find the max area contours
    out, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        if area < 150:
            continue
        rect = cv2.minAreaRect(contours[c])
        cv2.ellipse(image, rect, (0, 255, 0), 2, 8)
        cv2.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)
    return image, mask


while True:
    ret, frame = cap.read()
    cv2.imwrite("input.png", frame)
    cv2.imshow('input', frame)
    result, m_ = getPerson(frame)
    cv2.imshow('result', result)
    k = cv2.waitKey(50)&0xff
    if k == 27:
        cv2.imwrite("result.png", result)
        cv2.imwrite("mask.png", m_)

        break
cap.release()
cv2.destroyAllWindows()