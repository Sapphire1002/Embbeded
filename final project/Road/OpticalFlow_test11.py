# OpenCV中的稠密光流 Farneback (動態)
# RGB來判斷，再用擴張取輪廓
import cv2
import numpy as np
# ---------------------------------------------------------------
def draw_flow(im,flow,step=16):
    h,w = im.shape[:2] #切片長寬
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1).astype(int)
    fx,fy = flow[y,x].T
 
    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines+0.5)
 
    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.circle(vis,(x1,y1),1,(0,0,255), -1)
    #cv2.imshow("vis",vis)
    return vis
# ---------------------------------------------------------------
cap = cv2.VideoCapture('road_back7.avi')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height, FPS))
# ---------------------------------------------------------------
ret,im = cap.read() #第一幀
old_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
# ---------------------------------------------------------------
# fourcc = 0x00000021 #存取影片
# cv2.VideoWriter_fourcc('H', '2', '6', '4')
# videoWriter = cv2.VideoWriter('./Farneback_RGB_dilate_5.mp4', fourcc , 30, (width, height)) # 建立 VideoWriter 物件，輸出影片至 output.avi
# ---------------------------------------------------------------
while (cap.isOpened()):
    ret,im = cap.read() #第二幀
    # print(im[400][78]) #[B][G][R] [47][46][40]坑洞顏色 ~ [56][56][47]
    # cv2.circle(im,(78,400),1,(0,0,255), 3)
    if ret==True :
        # ---------------------------------------------------------------
        new_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        im_buf = np.zeros_like(new_gray)
        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang  = cv2.cartToPolar(flow[..., 0], flow[..., 1]) #笛卡爾坐標轉換為極坐標，獲得 極軸(X軸)
        ang_buf = ang.astype('int8') 
        mag_buf = mag.astype('int8')
        old_gray = new_gray # 第一幀 = 第二幀
        # ---------------------------------------------------------------
        for y in range(479) :
            for x in range(719) :
                im_R = im[y][x][2]
                im_G = im[y][x][1]
                im_B = im[y][x][0]
                if (60 > im_R > 39) and (60 > im_G > 45) and (60 > im_R > 45) and (mag_buf[y][x] > 4):
                    im_buf[y][x] = 255
                else :
                    im_buf[y][x] = 0
        # ---------------------------------------------------------------
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(im_buf,kernel,iterations = 1)
        # ret, binary = cv2.threshold(im_buf,0,255,cv2.THRESH_BINARY)
        ret, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im,contours,-1,(0,255,255),1)

        for c in contours:
            if (cv2.contourArea(c) > 200):
                (x, y, w, h) = cv2.boundingRect(c)# 檢測輪廓
                cv2.rectangle(im, (x,y), (x+w,y+h), (255,255,0), 2)# 繪製檢測結果

        # ---------------------------------------------------------------
        cv2.imshow("im",im)
        cv2.imshow('im_buf',im_buf)
        # ---------------------------------------------------------------
        if cv2.waitKey(10) == 27:
            break
        # videoWriter.write(im) #輸出影片 要等...
    else :
        break

cap.release()
# videoWriter.release()
cv2.destroyAllWindows()