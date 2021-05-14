# OpenCV中的稠密光流 Farneback(動態)
# 利用每行角度平均來判斷
import cv2
import numpy as np

ang_grade = 5
mag_grade_min = 40
mag_grade_max = 60

data_buf1=list();data_buf2=list();data_buf3=list();data_buf4=list()
data_buf5=list();data_buf6=list();data_buf7=list();data_buf8=list()

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

cap = cv2.VideoCapture('road_back3.avi')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #影片(寬)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #影片(高)
FPS = int(cap.get(cv2.CAP_PROP_FPS)) #影片(品質)
print("Image Size: %d x %d , %d" % (width, height, FPS))
ret,im = cap.read() #第一幀
old_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化 
hsv = np.zeros_like(im) #掩蓋
hsv[..., 1] = 255 #每一行第一列為255 (飽和度)
#fourcc = 0x00000021 #存取影片
#cv2.VideoWriter_fourcc('H', '2', '6', '4')
#videoWriter = cv2.VideoWriter('./FindPothole_angle.mp4', fourcc , 30, (width, height)) # 建立 VideoWriter 物件，輸出影片至 output.avi

while (cap.isOpened()):
    ret,im = cap.read() #第二幀
    if ret==True :
        new_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        Crop_Image = new_gray[200:480,0:719] #擷取視窗[y,x]
        TD_Image = np.zeros_like(Crop_Image) #用來做二質化圖片

        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) #獲得Farneback
        OpticalFlow_image = draw_flow(new_gray,flow) #繪畫出光流

        mag, angle  = cv2.cartToPolar(flow[..., 0], flow[..., 1]) #笛卡爾坐標轉換為極坐標，獲得(強度)(方向)
        hsv[..., 0] = angle * 180 / np.pi / 2 #(色調) (pi=3.141592)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) #(明度)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) #計算用

        angle_buf = angle * 180 / np.pi / 2
        Crop_angle = angle_buf[200:480,0:719] #擷取視窗[y,x]
        mag_buf = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        Crop_mag = mag_buf[200:480,0:719] #擷取視窗[y,x]
        ######################################## 算8個空間角度平均
        for y in range(279) :
            for x in range(89) :
                data_buf1.append(angle_buf[y+200][x])
                data_buf2.append(angle_buf[y+200][x+90])
                data_buf3.append(angle_buf[y+200][x+180])
                data_buf4.append(angle_buf[y+200][x+270])
                data_buf5.append(angle_buf[y+200][x+360])
                data_buf6.append(angle_buf[y+200][x+450])
                data_buf7.append(angle_buf[y+200][x+540])
                data_buf8.append(angle_buf[y+200][x+630])
        angle_out1 = sum(data_buf1)/24831;data_buf1.clear()
        angle_out2 = sum(data_buf2)/24831;data_buf2.clear()
        angle_out3 = sum(data_buf3)/24831;data_buf3.clear()
        angle_out4 = sum(data_buf4)/24831;data_buf4.clear()
        angle_out5 = sum(data_buf5)/24831;data_buf5.clear()
        angle_out6 = sum(data_buf6)/24831;data_buf6.clear()
        angle_out7 = sum(data_buf7)/24831;data_buf7.clear()
        angle_out8 = sum(data_buf8)/24831;data_buf8.clear()
        print(angle_out1,angle_out2,angle_out3,angle_out4,angle_out5,angle_out6,angle_out7,angle_out8)
        ########################################
        for y in range(279) :
            for x in range(89) :
                #不可接受的範圍
                if (Crop_angle[y][x] < (angle_out1-ang_grade)) or (Crop_angle[y][x] > (angle_out1+ang_grade)): #角度限制 
                    if (mag_grade_min < Crop_mag[y][x] < mag_grade_max) : #強度限制
                        TD_Image[y][x] = 255
                #可接受範圍        
                else : 
                    TD_Image[y][x] = 0
                #不可接受的範圍
                if (Crop_angle[y][x+90] < (angle_out2-ang_grade)) or (Crop_angle[y][x+90] > (angle_out2+ang_grade)):
                    if (mag_grade_min < Crop_mag[y][x+90] < mag_grade_max) :
                        TD_Image[y][x+90] = 255
                #可接受範圍        
                else : 
                    TD_Image[y][x+90] = 0
                #不可接受的範圍
                if (Crop_angle[y][x+180] < (angle_out3-ang_grade)) or (Crop_angle[y][x+180] > (angle_out3+ang_grade)):
                    if (mag_grade_min < Crop_mag[y][x+180] < mag_grade_max) :
                        TD_Image[y][x+180] = 255
                #可接受範圍        
                else : 
                    TD_Image[y][x+90] = 0
                #不可接受的範圍
                if (Crop_angle[y][x+270] < (angle_out4-ang_grade)) or (Crop_angle[y][x+270] > (angle_out4+ang_grade)):
                    if (mag_grade_min < Crop_mag[y][x+270] < mag_grade_max) :
                        TD_Image[y][x+270] = 255
                #可接受範圍        
                else : 
                    TD_Image[y][x+270] = 0
                #不可接受的範圍
                if (Crop_angle[y][x+360] < (angle_out5-ang_grade)) or (Crop_angle[y][x+360] > (angle_out5+ang_grade)):
                    if (mag_grade_min < Crop_mag[y][x+360] < mag_grade_max) :
                        TD_Image[y][x+360] = 255
                #可接受範圍        
                else : 
                    TD_Image[y][x+360] = 0
                #不可接受的範圍
                if (Crop_angle[y][x+450] < (angle_out6-ang_grade)) or (Crop_angle[y][x+450] > (angle_out6+ang_grade)):
                    if (mag_grade_min < Crop_mag[y][x+450] < mag_grade_max) :
                        TD_Image[y][x+450] = 255
                #可接受範圍        
                else : 
                    TD_Image[y][x+450] = 0
                #不可接受的範圍
                if (Crop_angle[y][x+540] < (angle_out7-ang_grade)) or (Crop_angle[y][x+540] > (angle_out7+ang_grade)):
                    if (mag_grade_min < Crop_mag[y][x+540] < mag_grade_max) :
                        TD_Image[y][x+540] = 255
                #可接受範圍        
                else : 
                    TD_Image[y][x+540] = 0
                #不可接受的範圍
                if (Crop_angle[y][x+630] < (angle_out8-ang_grade)) or (Crop_angle[y][x+630] > (angle_out8+ang_grade)):
                    if (mag_grade_min < Crop_mag[y][x+630] < mag_grade_max) :
                        TD_Image[y][x+630] = 255
                #可接受範圍        
                else : 
                    TD_Image[y][x+630] = 0
    
        ret, contours, hierarchy = cv2.findContours(TD_Image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        im_buf = im[200:480,0:719] #用來繪畫輪廓
        cv2.drawContours(im_buf,contours,-1,(0,0,255),1)

        for c in contours:
            if cv2.contourArea(c) > 50:
                (x, y, w, h) = cv2.boundingRect(c)# 檢測輪廓
                cv2.rectangle(im_buf, (x,y), (x+w,y+h), (255,255,0), 2)# 繪製檢測結果

        cv2.imshow("Image",im) #原始影片
        cv2.imshow("TD_Image",TD_Image) 
        cv2.imshow('OpticalFlow',OpticalFlow_image) #光流影片

        if cv2.waitKey(10) == 27:
            break

        old_gray = new_gray # 第一幀 = 第二幀
        #videoWriter.write(im) #輸出影片 要等...
    else :
        break

cap.release()
#videoWriter.release()
cv2.destroyAllWindows()