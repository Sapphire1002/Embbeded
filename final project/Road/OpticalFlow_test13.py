# OpenCV中的稠密光流 Farneback(動態)
# 利用兩點直線方程式到Foe距離，並且用角度顏色區分
import cv2
import numpy as np
# -----------------------------------------------------------
def lines (im,flow,step=8):
    h,w = im.shape[:2] #切片長寬
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1).astype(int)
    # print(len(x)) = 2070601(1919*1079)(x*y)
    fx,fy = flow[y,x].T
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines+0.5)
    return lines
# -----------------------------------------------------------
def draw_flow(im,lines):
    for (x1,y1),(x2,y2) in lines:
        cv2.line(im,(x1,y1),(x2,y2),(255,0,0),1)
        cv2.circle(im,(x1,y1),1,(0,255,0), -1)
    return im
# -----------------------------------------------------------
def draw_RGB(im,lines) :
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    Foe_x = 1080 ; Foe_y = 30 #假設Foe (960,30)
    cv2.circle(vis,(Foe_x,Foe_y),1,(0,255,255),20) #Foe
    for (x1,y1),(x2,y2) in lines:
        mag = np.sqrt(np.square(x2-x1) + np.square(y2-y1)) #得到運動的位移長度
        fx = x2-x1 ; fy = y2-y1 #變化量
        ang = np.arctan2(fy,fx)*180/np.pi/2 #角度(0~90)
        if (mag > 4) :
            if (x1 != x2) and (y1 != y2) :
                # ------------------------------------------
                A = np.array([[x1,1],[x2,1]]) #係數
                B = np.array([y1,y2]).reshape(2, 1) #常數
                A_inv = np.linalg.inv(A)
                ans = A_inv.dot(B)
                a = ans[0]; b = ans[1]
                F1 = a*Foe_x+b-Foe_y
                # ------------------------------------------
                if (80 > F1 > -80): #範圍裡
                    # cv2.circle(vis,(x1,y1),1,(255,0,0),2) #B
                    cv2.circle(vis,(x1,y1),1,(0,0,0),1) #BLOCK
                elif (F1 == 0): #等於
                    # cv2.circle(vis,(x1,y1),1,(0,255,0),2) #G
                    cv2.circle(vis,(x1,y1),1,(0,0,0),1) #BLOCK
                else: #不在範圍裡
                    if (30 >= ang > 0) :
                        cv2.circle(vis,(x1,y1),1,(0,128,255),2) #橘色
                    elif (60 >= ang > 30) :
                        cv2.circle(vis,(x1,y1),1,(204,102,0),2) #藍色
                    elif (90 >= ang > 60) :
                        cv2.circle(vis,(x1,y1),1,(153,153,0),2) #藍色
                    elif (0 >= ang > -30) :
                        cv2.circle(vis,(x1,y1),1,(0,255,255),2) #深黃
                    elif (-30 >= ang > -60) :
                        cv2.circle(vis,(x1,y1),1,(102,255,102),2) #綠色
                    elif (-60 >= ang > -90) :
                        cv2.circle(vis,(x1,y1),1,(255,102,255),2) #紫色
                    else :
                        cv2.circle(vis,(x1,y1),1,(0,0,255),2) #R
            else :
                cv2.circle(vis,(x1,y1),1,(255,0,0),2) #B
        else :
            cv2.circle(vis,(x1,y1),1,(0,0,0),1) #BLOCK
            # cv2.circle(vis,(x1,y1),1,(255,0,0),2) #B
    return vis
# -----------------------------------------------------------
cap = cv2.VideoCapture('ATN-1036_CH0220190622115005-R.avi')
# cap = cv2.VideoCapture('ATN-1036_CH0120190624105513-R.avi')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height,FPS))
# -----------------------------------------------------------
ret,im = cap.read() #第一幀
old_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
old_gray = cv2.GaussianBlur(old_gray,(11,11),0) #高斯濾波(11x11)
# -----------------------------------------------------------
fourcc = 0x00000021 #存取影片
cv2.VideoWriter_fourcc('H', '2', '6', '4')
videoWriter = cv2.VideoWriter('dist_Foe_4.mp4',fourcc,30,(width,height))
# -----------------------------------------------------------
#設定初始狀態
show_im = False
show_line = False
show_RGB = False
write_video = False
# -----------------------------------------------------------
while (cap.isOpened()):
    ret,im = cap.read() #第二幀
    if ret==True :
        new_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        im_draw = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        new_gray = cv2.GaussianBlur(new_gray,(11,11),0) #高斯濾波(11x11)
        # ---------------------------------------------------------------
        flow = cv2.calcOpticalFlowFarneback(old_gray,new_gray,None,0.5,3,15,3,7,1.5,0) #位移矢量 d   
        lines_buf = lines(im_draw,flow)
        old_gray = new_gray #第一幀=第二幀(圖像)  
        # ---------------------------------------------------------------
        cv2.imshow("frame_im",im)
        if show_line:
            cv2.imshow("frame_line",draw_flow(im,lines_buf))
        if show_RGB:
            cv2.imshow("frame_RGB",draw_RGB(im_draw,lines_buf))
        if write_video:
            videoWriter.write(draw_RGB(im_draw,lines_buf)) #輸出影片
        # ---------------------------------------------------------------
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_line = not show_line
            print('show_line is', ['off', 'on'][show_line])
        if ch == ord('2'):
            show_RGB = not show_RGB
            print('show_RGB is', ['off', 'on'][show_RGB])
        if ch == ord('3'):
            write_video = not write_video
            print('write_video is', ['off', 'on'][write_video])
        # ---------------------------------------------------------------
    else :
        break
cap.release()
videoWriter.release()
cv2.destroyAllWindows()


