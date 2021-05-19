# OpenCV中的稠密光流 Farneback(動態)
# 利用兩點直線方程式到Foe距離找到非共平面及平面
# 先分割出白色跟黃色，再用Foe距離來判斷，並且用紅色來輸出
# 2019/9/12做到白色，黃色還沒
import cv2
import numpy as np
# -----------------------------------------------------------
def lines (im,flow,step=8):
    h,w = im.shape[:2] #切片長寬
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1).astype(int)
    fx,fy = flow[y,x].T
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines+0.5)
    return lines
# -----------------------------------------------------------
#繪畫光流
def draw_flow(im,lines): #輸入灰階
    for (x1,y1),(x2,y2) in lines:
        cv2.line(im,(x1,y1),(x2,y2),(255,0,0),1)
        cv2.circle(im,(x1,y1),1,(0,255,0), -1)
    cv2.imshow("draw_flow",im)
# -----------------------------------------------------------
#分割白色物體
def Segmentation(im): #輸入RGB IMAGE
    im_BGR = cv2.split(im) #切片RGB
    Segmentation_out = im.copy()
    # -----------------------------------------------
    # 設定參數
    yellow_R = 214; yellow_G = 195; yellow_B = 129
    # white_R  = 219; white_G  = 224; white_B  = 223 #原
    white_R  = 220; white_G  = 224; white_B  = 223 #改
    grade_R = 25; grade_G = 29; grade_B = 28 #範圍等級
    # -----------------------------------------------
    ret,thresh_WR_0 = cv2.threshold(im_BGR[2],white_R-grade_R,255,cv2.THRESH_BINARY)
    ret,thresh_WR_1 = cv2.threshold(im_BGR[2],white_R+grade_R,255,cv2.THRESH_BINARY_INV)
    thresh_WR_0[thresh_WR_1==0] = 0
    ret,thresh_WG_0 = cv2.threshold(im_BGR[1],white_G-grade_G,255,cv2.THRESH_BINARY)
    ret,thresh_WG_1 = cv2.threshold(im_BGR[1],white_G+grade_G,255,cv2.THRESH_BINARY_INV)
    thresh_WG_0[thresh_WG_1==0] = 0
    ret,thresh_WB_0 = cv2.threshold(im_BGR[0],white_B-grade_B,255,cv2.THRESH_BINARY)
    ret,thresh_WB_1 = cv2.threshold(im_BGR[0],white_B+grade_B,255,cv2.THRESH_BINARY_INV)
    thresh_WB_0[thresh_WB_1==0] = 0
    # -----------------------------------------------
    thresh_WR_0[thresh_WG_0 == 0] = 0
    thresh_WR_0[thresh_WB_0 == 0] = 0
    Segmentation_out[thresh_WR_0==255,:] = (0,0,255)
    # -----------------------------------------------
    ret, contours, hierarchy = cv2.findContours(thresh_WR_0,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("thresh_WR_0",cv2.pyrDown(thresh_WR_0))
    cv2.imshow("Segmentation_out",cv2.pyrDown(Segmentation_out)) #紅色覆蓋
    return contours,thresh_WR_0
# -----------------------------------------------------------
#計算Foe距離
def CalFoe(im,lines,contours_buf,thresh_WR_0,mag) : #輸入RGB IMAGE
    im_test= im.copy() #複製IMAGE
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
    Fis = np.zeros_like(gray) #掩蓋
    # Foe_x = 1170 ; Foe_y = 33 #假設Foe (1080,30)
    Foe_x = 1080 ; Foe_y = 33 #假設Foe (1080,30)
    # -----------------------------------------------
    for (x1,y1),(x2,y2) in lines:
        if (mag[y1][x1] > 4) : #強度大於自訂範圍
            if (x1 != x2) and (y1 != y2) : #解聯立方程式(不能有無解和無窮多解)
                # ------------------------------------------
                # 解聯立方程式
                A = np.array([[x1,1],[x2,1]]) #係數
                B = np.array([y1,y2]).reshape(2, 1) #常數
                A_inv = np.linalg.inv(A)
                ans = A_inv.dot(B)
                a = ans[0]; b = ans[1]
                F1 = abs(a*Foe_x+b-Foe_y)
                # ------------------------------------------
                # 取範圍
                if (80 > F1): # 範圍內
                    Fis[y1][x1] = 255
                # elif (F1 == 0): # 等於
                #     Fis[y1][x1] = 255
                else: # 不在範圍裡
                    Fis[y1][x1] = 0
                # ------------------------------------------
            else: #有無解和無窮多解，算在範圍內
                Fis[y1][x1] = 255
        else: #mag濾掉的點
            Fis[y1][x1] = 100
    # -----------------------------------------------   
    for c in contours :
        if (cv2.contourArea(c) > 1000): #設定Contours面積範圍
            cnt_OK_range = 0 #設定計數器(FOE範圍內)
            cnt_NOT_range = 0 #設定計數器(FOE範圍外)
            (x, y, w, h) = cv2.boundingRect(c) # 檢測輪廓 
            for (x1,y1),(x2,y2) in lines :
                if (x+w > x1 > x) and (y+h > y1 > y) : #contours範圍
                    if (thresh_WR_0[y1][x1] == 255) : #有在白色範圍內
                        if (Fis[y1][x1] == 255) : #FOE範圍內 (藍)
                            cnt_OK_range += 1 #FOE範圍內計數器+1
                            cv2.circle(im_test,(x1,y1),1,(255,0,0), 2)
                        elif (Fis[y1][x1] == 0): #FOE範圍外 (紅)
                            cnt_NOT_range += 1 #FOE範圍外計數器+1
                            cv2.circle(im_test,(x1,y1),1,(0,0,255), 2)
                        else: #Mag濾掉的點 (黑)
                            cv2.circle(im_test,(x1,y1),1,(0,0,0), 2)
            if (cnt_OK_range != 0) or (cnt_NOT_range != 0): #分母或分子 不能等於0 
                cnt_range = cnt_OK_range + cnt_NOT_range #百分比分母
                Percentage = cnt_OK_range/cnt_range*100 #計算百分比
                if (Percentage > 10) : #大於40%就框起來
                    cv2.rectangle(im_test, (x,y), (x+w,y+h), (255,255,0), 3)
    cv2.imshow("im_test",cv2.pyrDown(im_test))
    return im_test
# -----------------------------------------------------------
#影片資訊
# cap = cv2.VideoCapture('ATN-1036_CH0220190622115005-R.avi')
path_test = 'D:\\Special_topic_project\\Project\\N04-TestReport\\EDIT_video\\Edit_video_4.mp4' #設定路徑
cap = cv2.VideoCapture(path_test)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #影片(寬)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #影片(高)
FPS = int(cap.get(cv2.CAP_PROP_FPS)) #FPS
print("Image Size: %d x %d , %d" % (width,height,FPS))
# -----------------------------------------------------------
ret,im = cap.read() #第一幀
old_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
old_gray = cv2.GaussianBlur(old_gray,(11,11),0) #高斯濾波(11x11)
# -----------------------------------------------------------
#存取影片
# fourcc = 0x00000021 
# cv2.VideoWriter_fourcc('H', '2', '6', '4')
# videoWriter = cv2.VideoWriter('D:\\Special_topic_project\\Project\\dist_Foe_6_CalFoe.mp4',fourcc,30,(width,height),False) #黑白
# videoWriter = cv2.VideoWriter('D:\\Special_topic_project\\Project\\dist_Foe_6_CalFoe.mp4',fourcc,30,(width,height)) #RGB
# -----------------------------------------------------------
#設定初始狀態
show_im=False; show_line=False; show_CalFoe=False; write_video=False; write_image=False
# -----------------------------------------------------------
while (cap.isOpened()):
    ret,im = cap.read() #第二幀
    if ret==True :
        new_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        im_draw = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        new_gray = cv2.GaussianBlur(new_gray,(11,11),0) #高斯濾波(11x11)
        flow = cv2.calcOpticalFlowFarneback(old_gray,new_gray,None,0.5,3,15,3,7,1.5,0) #位移矢量  
        mag, ang  = cv2.cartToPolar(flow[..., 0], flow[..., 1]) #得出 強度，角度
        old_gray = new_gray #第一幀=第二幀(圖像)  
        lines_buf = lines(im_draw,flow)
        # ---------------------------------------------------------------
        contours, thresh_WR_0 = Segmentation(im)
        CalFoe_buf = CalFoe(im,lines_buf,contours,thresh_WR_0,mag) #Foe判斷
        # ---------------------------------------------------------------
        if show_line:
            draw_flow(im_draw,lines_buf) #畫線
            # None
        if show_CalFoe:
            # contours, thresh_WR_0 = Segmentation(im)
            # CalFoe(im,lines_buf,contours,thresh_WR_0,mag) #Foe判斷
            None
        if write_image:
            # cv2.imwrite('CalFoe.jpg', CalFoe_buf) #輸出圖片
            None
        if write_video:
            # videoWriter.write(CalFoe_buf) #輸出影片
            None
        # ---------------------------------------------------------------
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_line = not show_line
            print('show_line is', ['off', 'on'][show_line])
        if ch == ord('2'):
            show_CalFoe = not show_CalFoe
            print('show_CalFoe is', ['off', 'on'][show_CalFoe])
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


