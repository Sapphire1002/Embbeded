# OpenCV中的稠密光流 Farneback(動態)
# 利用兩點直線方程式到Foe距離找到非共平面及平面
# 找出斑馬線白色與黃色
# 白色
import cv2
import numpy as np
# -----------------------------------------------------------
def lines (im,flow,step=4):
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
    # vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
    Eis = np.zeros_like(gray) #掩蓋
    Foe_x = 1080 ; Foe_y = 30 #假設Foe (960,30)
    # cv2.circle(vis,(Foe_x,Foe_y),1,(0,255,255),20) #Foe
    for (x1,y1),(x2,y2) in lines:
        mag = np.sqrt(np.square(x2-x1) + np.square(y2-y1)) #得到運動的位移長度
        # fx = x2-x1 ; fy = y2-y1 #變化量
        # ang = np.arctan2(fy,fx)*180/np.pi/2 #角度(0~90)
        if (mag > 3) : #強度大於自訂範圍
            if (x1 != x2) and (y1 != y2) : #解聯立方程式(不能有無解和無窮多解)
                # ------------------------------------------
                im_5R = im[y1][x1][2];im_5G = im[y1][x1][1];im_5B = im[y1][x1][0]
                # ------------------------------------------
                #解聯立方程式
                A = np.array([[x1,1],[x2,1]]) #係數
                B = np.array([y1,y2]).reshape(2, 1) #常數
                A_inv = np.linalg.inv(A)
                ans = A_inv.dot(B)
                a = ans[0]; b = ans[1]
                F1 = a*Foe_x+b-Foe_y
                # ------------------------------------------
                if (70 > F1 > -70): #範圍裡
                    Eis[y1][x1] = 0
                elif (F1 == 0): #等於
                    Eis[y1][x1] = 0
                else: #不在範圍裡
                    if (224> im_5R >214) and (229> im_5G >219) and (228> im_5B >218) : #whiteline
                        Eis[y1][x1] = 255
                # ------------------------------------------
            else :
                Eis[y1][x1] = 0
        else :
            Eis[y1][x1] = 0
    return Eis
# -----------------------------------------------------------
# cap = cv2.VideoCapture('ATN-1036_CH0220190622115005-R.avi')
cap = cv2.VideoCapture('Edit_video_4.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width,height,FPS))
# -----------------------------------------------------------
ret,im = cap.read() #第一幀
old_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
# old_gray = cv2.GaussianBlur(old_gray,(11,11),0) #高斯濾波(11x11)
# -----------------------------------------------------------
# fourcc = 0x00000021 #存取影片
# cv2.VideoWriter_fourcc('H', '2', '6', '4')
# videoWriter = cv2.VideoWriter('dist_Foe_4_B.mp4',fourcc,30,(width,height),False) #黑白用
# videoWriter = cv2.VideoWriter('dist_Foe_4_B.mp4',fourcc,30,(width,height)) #RGB用
# -----------------------------------------------------------
show_im=False; show_line=False; show_RGB=False; write_video=False; write_image=False #設定初始狀態
# -----------------------------------------------------------
while (cap.isOpened()):
    ret,im = cap.read() #第二幀
    if ret==True :
        new_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        im_draw = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        # new_gray = cv2.GaussianBlur(new_gray,(11,11),0) #高斯濾波(11x11)
        # ---------------------------------------------------------------
        flow = cv2.calcOpticalFlowFarneback(old_gray,new_gray,None,0.5,3,15,3,7,1.5,0) #位移矢量   
        old_gray = new_gray #第一幀=第二幀(圖像)  
        lines_buf = lines(im_draw,flow)
        draw_flow_buf = draw_flow(im_draw,lines_buf) #畫線
        draw_RGB_buf = draw_RGB(im,lines_buf) #RGB判斷
        # ---------------------------------------------------------------
        kernel = np.ones((10,10),np.uint8)
        draw_RGB_buf = cv2.dilate(draw_RGB_buf,kernel,iterations=1) #擴張
        ret, draw_RGB_buf = cv2.threshold(draw_RGB_buf,0,255,cv2.THRESH_BINARY)
        ret, contours, hierarchy = cv2.findContours(draw_RGB_buf,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im,contours,-1,(0,255,255),2)
        for c in contours:
            if (cv2.contourArea(c) > 400):
                (x, y, w, h) = cv2.boundingRect(c)# 檢測輪廓
                cv2.rectangle(im, (x,y), (x+w,y+h), (255,255,0), 2)# 繪製檢測結果
        # ---------------------------------------------------------------
        cv2.imshow("frame_im",im)
        if show_line:
            cv2.imshow("frame_line",draw_flow_buf)
            # cv2.imshow("frame_line",draw_flow(im_draw,lines_buf))
            # None
        if show_RGB:
            cv2.imshow("frame_RGB",draw_RGB_buf)
            # cv2.imshow("frame_RGB",draw_RGB(im,lines_buf))
            # None
        if write_image:
            # cv2.imwrite('OpticalFlow_test14_im.jpg', im) #輸出圖片
            # cv2.imwrite('OpticalFlow_test14_RGB.jpg', draw_RGB(im,lines_buf)) #輸出圖片
            None
        if write_video:
            # videoWriter.write(im) #輸出影片
            # videoWriter.write(draw_RGB_buf) #輸出影片
            None
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
            write_image = not write_image
            print('write_image is', ['off', 'on'][write_image])
        if ch == ord('4'):
            write_video = not write_video
            print('write_video is', ['off', 'on'][write_video])
        # ---------------------------------------------------------------
    else:
        break
cap.release()
# videoWriter.release()
cv2.destroyAllWindows()


