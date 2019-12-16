import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="./cars.mp4", help="path to input video file")
ap.add_argument("-t", "--algorithm", type=str, default="m", help="OpenCV object tracker type")
args = vars(ap.parse_args())
class pedestrain():
    """行人类"""
    #每个行人都由roi，id，一个卡尔曼滤波器组成
    def __init__(self, id,frame,track_window):
        self.id = id
        ##SET ROI
        x,y,w,h = track_window
        self.track_window = track_window
        self.roi = cv2.cvtColor(frame[y:y+h, x:x+w],cv2.COLOR_BGR2HSV)
        roi_hist  = cv2.calcHist([self.roi],[0],None,[16],[0,180])
        self.roi_hist = cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        ##SET KALMAN
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03
        self.measurement = np.array((2,1),np.float32)

        self.prediction = np.zeros((2,1),np.float32)
        self.term_crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)

        self.center = None
        self.update(frame)

    def __del__(self):
        print("pedestrain %d destoryed"%self.id)

    def update(self,frame):
        print("update %d"%self.id)
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        back_project = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)

        if args.get("algorithm") == "c":
            ret,self.track_window=cv2.CamShift(back_project,self.track_window,self.term_crit)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            self.center = center(pts)
            cv2.polylines(frame,[pts],True,255,1)

        if not args.get("algorithm") or args.get("algorithm") == "m":
            ret,self.track_window =cv2.meanShift(back_project,self.track_window,self.term_crit)
            x,y,w,h = self.track_window
            self.center = center([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1)

        self.kalman.correct(self.center)
        prediction = self.kalman.predict()
        cv2.circle(frame,(int(prediction[0]),int(prediction[1])),4,(0,255,0),-1)

        #fake shadow
        cv2.putText(frame,"id:%d->%s"%(self.id,self.center),(11,(self.id+1)*25+1),cv2.FONT_HERSHEY_PLAIN,0.6,(0,0,0),1,cv2.LINE_AA)
        #actual info
        cv2.putText(frame,"id :%d-》%s"%(self.id,self.center),(10,(self.id+1)*25),cv2.FONT_HERSHEY_PLAIN,0.6,(0,255,0),1,cv2.LINE_AA)

