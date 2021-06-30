# -*- coding: utf-8 -*-
# OpenCV中的稠密光流 Farneback (動態)
import cv2
import numpy as np


def draw_flow(gray, flow, step=5):
    h,w = gray.shape[:2]
    y,x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx,fy = flow[y, x].T

    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    
    print(ang)
    
    # create image and draw
    #vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    Drow_img = np.zeros_like(gray)
    for (x1,y1),(x2,y2) in lines:
        #cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
        if (mag[y1][x1] > 5) :#and (gray[y1][x1] > 200) : #強度大於自訂範圍
            print(ang[y1][x1]*180/np.pi/2)
            if( 80 < (ang[y1][x1]*180/np.pi/2)) and ((ang[y1][x1]*180/np.pi/2) < 180):
                Drow_img[y1][x1] = 255
            #if(0 < ang[y1][x1] < 1):
            #cv2.line(vis,(x1,y1),(x2,y2),(0,0,255),2)
            
            
    cv2.imshow("Drow_img",Drow_img)
    return Drow_img
 
#cap = cv2.VideoCapture(0)
#ret,im = cap.read()

cap = cv2.VideoCapture('./last.avi')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height, FPS))

ret,im = cap.read() # 第一幀
old_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
 
# fourcc = 0x00000021  # cv2.VideoWriter_fourcc('H', '2', '6', '4')
# videoWriter = cv2.VideoWriter('C:/Users/Yeeder/Desktop/road_detection/Farneback_output_town.mp4', fourcc , 30, (width, height)) # 建立 VideoWriter 物件，輸出影片至 output.avi

while (cap.isOpened()):
    ret,im = cap.read() #第二幀
    
    if ret==True :
        cv2.imshow('im',im)
        new_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        #ret, thresh = cv2.threshold(new_gray.copy(), 160,255, cv2.THRESH_BINARY)
        #cv2.imshow('threshold',thresh)
        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) # compute flow
        old_gray = new_gray # 第一幀 = 第二幀
        # videoWriter.write(draw_flow(new_gray,flow)) #輸出影片 要等...
        cv2.imshow('Optical flow',draw_flow(new_gray,flow))
        if cv2.waitKey(1) == ord('q'):
            break
        elif cv2.waitKey(1) == ord('p'):
            while cv2.waitKey(1) != ord(' '):
                pass
    else :
        break

cap.release()
# videoWriter.release()
cv2.destroyAllWindows()
