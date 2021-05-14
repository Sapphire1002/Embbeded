#OpenCV中的稠密光流 Farneback (動態)
import cv2
import numpy as np

def draw_flow(im,flow,step=16):
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1).astype(int)
    fx,fy = flow[y,x].T
 
    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)
 
    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(255,0,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    #cv2.imshow("vis",vis)
    return vis
 
#cap = cv2.VideoCapture(0)
#ret,im = cap.read()

cap = cv2.VideoCapture('ATN-1036_CH0220190622115005-R.avi')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height, FPS))

ret,im = cap.read() #第一幀
old_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
 
fourcc = 0x00000021#cv2.VideoWriter_fourcc('H', '2', '6', '4')
videoWriter = cv2.VideoWriter('./Farneback_output.mp4', fourcc , 30, (width, height)) # 建立 VideoWriter 物件，輸出影片至 output.avi

while (cap.isOpened()):
    ret,im = cap.read() #第二幀
    
    if ret==True :
        new_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
    
        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) # compute flow
 
        old_gray = new_gray # 第一幀 = 第二幀

        videoWriter.write(draw_flow(new_gray,flow)) #輸出影片 要等...

        cv2.imshow('Optical flow',draw_flow(new_gray,flow))

        if cv2.waitKey(10) == 27:
            break
    else :
        break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()