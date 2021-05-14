#OpenCV中的LK光流(靜態)
import numpy as np
import cv2

old_frame = cv2.imread('lena.jpg') # 取出视频的第一帧
new_frame = cv2.imread('lena2.jpg') # 取出视频的第二帧

# ShiTomasi corner detection的参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# 光流法参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  #灰階化
new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  #灰階化

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params) #找邊角

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame) # 为绘制创建掩码图片

while True:
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params) # 计算光流以获取点的新位置
    # 选择good points
    good_old = p0[st == 1]
    good_new = p1[st == 1]
    
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel() #座標
        c,d = old.ravel() #座標

        circle_frame = cv2.circle(old_frame,(a,b),1, (0,255,0), -1)
        mask = cv2.line(mask, (a,b),(c,d),(255,0,0), 2)

    img = cv2.add(circle_frame,mask) #加上圓形

    cv2.namedWindow('new_frame',cv2.WINDOW_NORMAL)
    cv2.namedWindow('new_frame1',cv2.WINDOW_NORMAL)
    cv2.namedWindow('new_frame2',cv2.WINDOW_NORMAL)
    cv2.imshow('new_frame', img)
    cv2.imshow('new_frame1', old_frame)
    cv2.imshow('new_frame2', new_frame)
    k = cv2.waitKey(30)  # & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
