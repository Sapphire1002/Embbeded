#OpenCV中的Lucas-Kanade光流(動態)
import numpy as np
import cv2

# ShiTomasi corner detection的参数
feature_params = dict(maxCorners=800,
                      qualityLevel=0.01,
                      minDistance=7,
                      blockSize=7)

# 光流法参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=1,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture('road_back1.avi')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height, FPS))
color = np.random.randint(0, 255, (500, 3))
###########################################################
#fourcc = 0x00000021 #存取影片
#cv2.VideoWriter_fourcc('H', '2', '6', '4')
#videoWriter = cv2.VideoWriter('./LK_output.mp4', fourcc , 30, (width, height))
###########################################################

ret, old_frame = cap.read()  # 取出视频的第一帧
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # 灰階化

#p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

while (cap.isOpened()):
    ret, new_frame = cap.read()
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params) #特徵點

    if ret==True :
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params) # 计算光流以获取点的新位置
        # good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel() #座標
            c, d = old.ravel() #座標

            cv2.line(new_frame, (a, b), (c, d), (255,0,0), 2)
            cv2.circle(new_frame, (a, b), 3, (0,255,0), -1)

        old_gray = new_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        cv2.namedWindow("new_frame",0)
        cv2.resizeWindow("new_frame", 720, 480)
        cv2.imshow('new_frame', new_frame)

        if cv2.waitKey(10) == 27 :
            break

        #videoWriter.write(new_frame) #輸出影片 要等...
    else :
        break

cap.release()
#videoWriter.release()
cv2.destroyAllWindows()
