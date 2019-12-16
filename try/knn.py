import cv2


datapath = "D:/code/vehicle-speed-check/data/people.avi"
bs = cv2.createBackgroundSubtractorKNN(detectShadows = False)#背景减除器，设置阴影检测
#训练帧数
history=20
bs.setHistory(history)
frames=0
camera = cv2.VideoCapture(datapath)
count = 0


while True:
    ret, frame = camera.read()    # ret=True/False,判断是否读取到了图片
    if ret==True:
        fgmask = bs.apply(frame)  # 计算前景掩码，包含 前景的白色值 以及 阴影的灰色值
        if frames < history:
            frames += 1
            continue

        #对原始帧膨胀去噪，
        th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        #前景区域形态学处理
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,3)), iterations = 2)
        #绘制前景图像的检测框
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
        #对轮廓设置最小区域，筛选掉噪点框
            if cv2.contourArea(c) > 1000:
                #获取矩形框边界坐标
                (x,y,w,h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)
        #cv2.imwrite("frame%d.jpg" % count, fgmask) #保存处理后的每一帧图片，JPEG格式的图片
        cv2.imshow("mog", fgmask)
        cv2.imshow("thresh", th)
        cv2.imshow("diff", frame & cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR))
        cv2.imshow("detection", frame)
        count += 1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break
camera.release()
cv2.destroyAllWindows()
