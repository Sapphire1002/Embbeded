# OpenCV中的稠密光流 Farneback (動態)
# HSV做顯示
import cv2
import numpy as np
# ---------------------------------------------------------------
def draw_flow(im,flow,step=8):
    h,w = im.shape[:2] #切片長寬
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1).astype(int)
    fx,fy = flow[y,x].T
 
    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)
 
    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.circle(vis,(x1,y1),1,(0,0,255), -1)
    return vis
# ---------------------------------------------------------------
cap = cv2.VideoCapture('road_back6.avi')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height, FPS))
# ---------------------------------------------------------------
ret,im = cap.read() #第一幀
old_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
# ---------------------------------------------------------------
hsv = np.zeros_like(im) #掩蓋
hsv[..., 1] = 255 #每一行第一列為255 (飽和度)
# ---------------------------------------------------------------
# fourcc = 0x00000021 #存取影片
#cv2.VideoWriter_fourcc('H', '2', '6', '4')
#videoWriter = cv2.VideoWriter('./Farneback_output.mp4', fourcc , 30, (width, height)) # 建立 VideoWriter 物件，輸出影片至 output.avi
# ---------------------------------------------------------------
while (cap.isOpened()):
    ret,im = cap.read() #第二幀
    if ret==True :
        # ---------------------------------------------------------------
        new_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, angle  = cv2.cartToPolar(flow[..., 0], flow[..., 1]) #笛卡爾坐標轉換為極坐標，獲得 極軸(X軸) 和 極角(方位)
        angle_buf=angle.astype('int8') #陣列(整數)
        mag_buf=mag.astype('int8') #陣列(整數)
        # ---------------------------------------------------------------
        hsv[..., 0] = angle * 180 / np.pi / 2 #(色調) (pi=3.141592)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) #(明度)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # ---------------------------------------------------------------
        cv2.namedWindow('frame3',0)
        cv2.resizeWindow('frame3', 640, 480)
        cv2.imshow("frame3",im)
        cv2.namedWindow("frame1",0)
        cv2.resizeWindow("frame1", 1920, 1080)
        cv2.imshow('frame1',draw_flow(new_gray,flow))
        cv2.namedWindow("frame2",0)
        cv2.resizeWindow("frame2", 640, 480)
        cv2.imshow('frame2', rgb)
        # ---------------------------------------------------------------
        if cv2.waitKey(10) == 27:
            break
        old_gray = new_gray # 第一幀 = 第二幀
        #videoWriter.write(draw_flow(new_gray,flow)) #輸出影片 要等...
    else :
        break
cap.release()
#videoWriter.release()
cv2.destroyAllWindows()