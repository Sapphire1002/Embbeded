#OpenCV中的稠密光流 Farneback (靜態)
import numpy as np
import cv2

def draw_flow(im,flow,step=16):
    h,w = im.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int) 
    fx,fy = flow[y,x].T
 
    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)

    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(255,0,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    return vis

old_frame = cv2.imread('lena.jpg') # 取出視頻的第一幀
new_frame = cv2.imread('lena2.jpg') # 取出視頻的第二幀

old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY) #灰階化
new_gray = cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY) #灰階化

while True:
    #返回一个两通道的光流向量，实际上是每个点的像素位移值
    flow = cv2.calcOpticalFlowFarneback(old_gray,new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #old_frame = new_frame

    cv2.namedWindow('new_frame',cv2.WINDOW_NORMAL)
    cv2.namedWindow('new_frame1',cv2.WINDOW_NORMAL)
    cv2.namedWindow('new_frame2',cv2.WINDOW_NORMAL)
    cv2.imshow('new_frame', draw_flow(old_gray,flow))
    cv2.imshow('new_frame1', old_frame)
    cv2.imshow('new_frame2', new_frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()