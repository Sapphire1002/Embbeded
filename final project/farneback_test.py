import numpy as np
import cv2 as cv
 
def draw_flow(im,flow,step=16):
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1).astype(int)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines =  np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines =  np.int32(lines)

    # create image and draw
    vis = cv.cvtColor(im,cv.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv.circle(vis,(x1,y1),1,(0,255,0), -1)
    return vis


path = ''
cap = cv.VideoCapture(cv.samples.findFile("test.mp4"))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None,0.5, 3, 15, 3, 5, 1.2, 0)
    cv.imshow('Optical flow',draw_flow(next,flow))
    prvs = next
    if cv.waitKey(33) == 27:
        break
    
