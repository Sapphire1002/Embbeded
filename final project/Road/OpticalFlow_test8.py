# OpenCV中的稠密光流 Farneback(動態)
# 利用HSV顏色來判斷
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
        cv2.line(vis,(x1,y1),(x2,y2),(255,0,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0),-1)
    return vis
# ---------------------------------------------------------------
cap = cv2.VideoCapture('road_back3.avi')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #影片(寬)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #影片(高)
FPS = int(cap.get(cv2.CAP_PROP_FPS)) #影片(品質)
print("Image Size: %d x %d , %d" % (width, height, FPS))
# ---------------------------------------------------------------
ret,im = cap.read() #第一幀
old_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化 
# ---------------------------------------------------------------
hsv = np.zeros_like(im) #掩蓋
hsv[..., 1] = 255 #每一行第一列為255 (飽和度)
# ---------------------------------------------------------------
#fourcc = 0x00000021 #存取影片
#cv2.VideoWriter_fourcc('H', '2', '6', '4')
#videoWriter = cv2.VideoWriter('./FindPothole_HSV.mp4', fourcc , 30, (width, height)) # 建立 VideoWriter 物件，輸出影片至 output.avi
# ---------------------------------------------------------------
while (cap.isOpened()):
    ret,im = cap.read() #第二幀
    if ret==True :
        # ---------------------------------------------------------------
        new_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) #獲得Farneback
        OpticalFlow_image = draw_flow(new_gray,flow) #繪畫出光流
        # ---------------------------------------------------------------
        mag, angle  = cv2.cartToPolar(flow[..., 0], flow[..., 1]) #笛卡爾坐標轉換為極坐標，獲得(強度)(方向)
        hsv[..., 0] = angle * 180 / np.pi / 2 #(色調) (pi=3.141592)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) #(明度)
        # ---------------------------------------------------------------
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) #計算用
        rgb_buf = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) #輸出比對用
        # ---------------------------------------------------------------
        cropImg_HSV = hsv[200:480,0:719] #擷取視窗[y,x]
        cropImg_RGB_Image = rgb[200:480,0:719] #擷取視窗[y,x]
        RGB_Image_buf = rgb_buf[200:480,0:719] #擷取視窗[y,x] 輸出比對用
        # ---------------------------------------------------------------
        for y in range(279) :
            for x in range(718) :
                HSV_Hue_buf = cropImg_HSV[y][x][0] #[y][x][Hue]
                HSV_Lightness_buf = cropImg_HSV[y][x][2] #[y][x][Lightness]
                if (45 < HSV_Hue_buf < 55) and (5 < HSV_Lightness_buf < 50) :
                    cropImg_RGB_Image[y][x][0] = 255
                    cropImg_RGB_Image[y][x][1] = 255
                    cropImg_RGB_Image[y][x][2] = 255
                else:
                    cropImg_RGB_Image[y][x][0] = 0
                    cropImg_RGB_Image[y][x][1] = 0
                    cropImg_RGB_Image[y][x][2] = 0
        # ---------------------------------------------------------------
        cropImg_RGB_Image_Buf = cropImg_RGB_Image[0:279,0:719,0]
        ret, binary = cv2.threshold(cropImg_RGB_Image_Buf,0,255,cv2.THRESH_BINARY)

        ret, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        im_buf = im[200:480,0:719] #用來繪畫輪廓
        cv2.drawContours(im_buf,contours,-1,(0,0,255),1)
        cv2.drawContours(cropImg_RGB_Image,contours,-1,(0,0,255),1)

        for c in contours:
            if cv2.contourArea(c) > 3:
                (x, y, w, h) = cv2.boundingRect(c)# 檢測輪廓
                cv2.rectangle(im_buf, (x,y), (x+w,y+h), (255,255,0), 2)# 繪製檢測結果
        # ---------------------------------------------------------------
        cv2.imshow("Image",im) #原始影片
        cv2.imshow('OpticalFlow',OpticalFlow_image) #光流影片
        cv2.imshow('RGB_old_Image', RGB_Image_buf) #原RGB影片
        cv2.imshow('RGB_new_Image', cropImg_RGB_Image) #RGB影片
        # ---------------------------------------------------------------
        if cv2.waitKey(10) == 27:
            break

        old_gray = new_gray # 第一幀 = 第二幀
        #videoWriter.write(im) #輸出影片 要等...
    else :
        break

cap.release()
#videoWriter.release()
cv2.destroyAllWindows()