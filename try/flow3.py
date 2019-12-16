# encoding:utf-8
'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

import cv2
import numpy as np

import moving_tracking_feature as mtf

# from common import anorm2, draw_str

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=300,
                      qualityLevel=0.2,
                      minDistance=7,
                      blockSize=7)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


class App:
    # 构造方法，初始化一些参数和视频路径
    def __init__(self, video_src):
        self.track_len = 15
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.track_moving_feature = []

    # 光流运行方法
    def run(self):
        while True:
            ret, frame = self.cam.read()  # 读取视频帧
            if ret == True:
                frame = cv2.medianBlur(frame, 3, None)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
                # frame_gray = cv2.bilateralFilter(frame, 9, 75, 75)
                vis = frame.copy()

                if len(self.tracks) > 0:  # 检测到角点后进行光流跟踪
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                           **lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None,
                                                            **lk_params)  # 当前帧跟踪到的角点及图像和前一帧的图像作为输入 来找到前一帧的角点位置
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # 得到角点回溯与前一帧实际角点的位置变化关系
                    #print("abs", abs(p0 - p0r).reshape(-1, 2), d)
                    good = d < 1  # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                    new_tracks = []
                    # print("p1", p1)
                    trackpoint = p1.reshape(-1, 2)
                    trackpoint = trackpoint.tolist()
                    print("tack", trackpoint)
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):  # 将跟踪正确的点列入成功跟踪点
                        if not good_flag:
                            trackpoint.remove([x, y])
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

                    self.tracks = new_tracks


                    estimateSpeed(self.tracks, frame)
                    mark_track = new_tracks
                    print("track keypoint", self.tracks)

                    # for item, (x, y) in zip(mark_track, [np.float32(tr[-1]) for tr in mark_track]):
                    #     # print("item", item)
                    #     # temp = item
                    #     # flag =0
                    #     # for i in range(len(temp)-2):
                    #     #     z,t = temp[i]
                    #     #     if(x >z)&(y> t):
                    #     #         flag=flag+1
                    #     #         break
                    #     #     else:
                    #     #         mark_track.remove(item)
                    #     #         break
                    #     xlist = []
                    #     ylist = []
                    #
                    #     # for (x,y) in item:
                    #     #     print(x,y)
                    #     #     xlist.append(x)
                    #     #     ylist.append(y)
                    #
                    #     # plt.xlabel('X')
                    #     # plt.ylabel('Y')
                    #     # plt.scatter(xlist, ylist)
                    #     # plt.show()
                    #     # break;
                    #
                    # self.tracks = mark_track

                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False,
                                  (0, 255, 0))  # 以上一振角点为初始点，当前帧跟踪到的点为终点划线
                    # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
                    # 聚类

                if self.frame_idx % self.detect_interval == 0:  # 每5帧检测一次特征点
                    mask = np.zeros_like(frame_gray)  # 初始化和视频大小相同的图像
                    mask[:] = 255  # 将mask赋值255也就是算全部图像的角点
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:  # 跟踪的角点画圆
                        cv2.circle(mask, (x, y), 8, 0, -1)
                        # get moving tracking feartue，
                    state = mtf.subtract_background(self.tracks)

                    # shi-Tomasi角点检测
                    corner = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # 像素级别角点检测
                    # 亚像素角点检测
                    p = cv2.cornerSubPix(frame_gray, corner, (11, 11), (-1, -1), criteria)


                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            print(",x,y",[x,y])
                            self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break


def estimateSpeed(track, frame):
    vehicle = []
    # for tr in track:
    mask1 = np.zeros_like(frame)  # 初始化和视频大小相同的图像
    mask1[:] = 0  # 将mask赋值255也就是算全部图像的角点
    cv2.polylines(mask1, [np.int32(tr) for tr in track], False,
                  (255, 255, 255))
    cv2.imshow("zdm",mask1)
    # location1, location2
    # d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # # ppm = location2[2] / carWidht
    # ppm = 8.8
    # d_meters = d_pixels / ppm
    # # print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    # fps = 18
    # speed = d_meters * fps * 3.6
    # return speed


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = "D:/code/vehicle-speed-check/data/mobilecars.avi"

    App(video_src).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
