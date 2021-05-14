# OpenCV中的稠密光流 Farneback(動態)
# 利用一行角度平均來判斷
# !!!(問題)!!!
import cv2
import numpy as np

video_y = 280
video_x = 719
video_length = 302

# lists = [[[[0] for _ in range(3)]for j in range(2)]for k in range(1)] #三維維lists
lists = np.zeros((302, 280, 719), dtype=np.int) #[z][y][x]
data = list()
data_out = list()
# print(lists[2][2][2])
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

cap = cv2.VideoCapture('road_back4.avi')
ret,im = cap.read() #第一幀
old_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
Gus_old_gray_ = cv2.GaussianBlur(old_gray,(15,15),1.5) 
hsv = np.zeros_like(im) #掩蓋
hsv[..., 1] = 255 #每一行第一列為255 (飽和度)
#fourcc = 0x00000021 #存取影片
#cv2.VideoWriter_fourcc('H', '2', '6', '4')
#videoWriter = cv2.VideoWriter('./FindPothole_one_angle.mp4', fourcc , 30, (width, height)) # 建立 VideoWriter 物件，輸出影片至 output.avi
z =0
while (cap.isOpened()):
    ret,im = cap.read() #第二幀
    if ret==True :
        new_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        Gus_old_gray_ = cv2.GaussianBlur(new_gray,(15,15),1.5) 
        Crop_Image = new_gray[200:480,0:719] #擷取視窗[y,x]
        TD_Image = np.zeros_like(Crop_Image) #用來做二質化圖片
        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) #獲得Farneback
        print(flow)
        OpticalFlow_image = draw_flow(new_gray,flow) #繪畫出光流
        mag, angle  = cv2.cartToPolar(flow[..., 0], flow[..., 1]) #笛卡爾坐標轉換為極坐標，獲得(強度)(方向)




        hsv[..., 0] = angle * 180 / np.pi / 2 #(色調) (pi=3.141592)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) #(明度)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) #計算用

        angle_buf = angle * 180 / np.pi / 2
        Crop_angle = angle_buf[200:480,0:719] #擷取視窗[y,x]
        # mag_buf = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Crop_mag = mag_buf[200:480,0:719] #擷取視窗[y,x]

        for y in range(280):
            for x in range(719):
                lists[z][y][x] = Crop_angle[y][x]
        z += 1
        
                    
        cv2.imshow("Image",im) #原始影片
        # cv2.imshow("TD_Image",TD_Image) 
        # cv2.imshow('OpticalFlow',OpticalFlow_image) #光流影片

        if cv2.waitKey(10) == 27:
            break

        old_gray = new_gray # 第一幀 = 第二幀
        #videoWriter.write(im) #輸出影片 要等...
    else :
        break
cap.release()
#videoWriter.release()
cv2.destroyAllWindows()

for y in range(280):
    for x in range(719):
        for z in range(302):
            data.append(lists[z][y][x])
        DATA_BUFF = sum(data)/302
        data_out.append(DATA_BUFF)
        data.clear()
# print(len(data_out)) #201320個
# str = ''
# str4 = str.join(data_out)
# fp = open("angle_data.txt","w")
# fp.write(str4)
# fp.close()

