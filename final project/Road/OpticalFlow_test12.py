# OpenCV中的稠密光流 Farneback(動態)
# 找Foe奌並且做判斷找會動的物體
# 先利用短片來取眾數，然後取方程式
# !!!!問題
# 方法網址 : http://m.anoah.com/app/sltd/?c=main&a=detail&id=11728&fbclid=IwAR3xms8KHjz89cVUvwxm2IGLd0OSmVAN3OOK-RElrlWwTsTOHFXUBUSlfWU
import cv2
import numpy as np
from collections import Counter 
# -----------------------------------------------------------
# 繪畫光流在影片上
# def draw_flow(im,flow,step=8):
#     h,w = im.shape[:2] #切片長寬
#     y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1).astype(int)
#     fx,fy = flow[y,x].T
#     # create line endpoints
#     lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
#     lines = np.int32(lines)
#     # create image and draw
#     vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
#     for (x1,y1),(x2,y2) in lines:
#         cv2.line(vis,(x1,y1),(x2,y2),(255,0,0),1)
#         cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
#     return vis
# -----------------------------------------------------------
# 取眾樹模組(別人撰寫)
def mode(List):
    # list of elements to calculate mode 
    n_num = List
    n = len(n_num) 
    data = Counter(n_num) 
    get_mode = dict(data) 
    mode = [k for k, v in get_mode.items() if v == max(list(data.values()))] 

    if len(mode) == n: 
        # get_mode = "No mode found"
        mode = sum(List)/len(List)
        # print(mode)  
        return mode 
    elif len(mode) > 1 :
        mode = sum(mode)/len(mode)
        return mode
    elif len(mode) == 1:
        return mode
# -----------------------------------------------------------
cap = cv2.VideoCapture('Edit_video_1.mp4')
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = 720
height = 480

FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height, FPS))
# -----------------------------------------------------------
ang_buf = np.zeros((100, height, width), dtype=np.int) #[z][y][x] 建立三維陣列(整數)
mag_buf = np.zeros((100, height, width), dtype=np.int) #[z][y][x] 建立三維陣列(整數)
# -----------------------------------------------------------
ret,im = cap.read() #第一幀
old_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
# Gus_old_gray = cv2.GaussianBlur(old_gray,(11,11),0) #高斯濾波(11x11)
# -----------------------------------------------------------
# fourcc = 0x00000021 #存取影片
#cv2.VideoWriter_fourcc('H', '2', '6', '4')
#videoWriter = cv2.VideoWriter('./Farneback_output.mp4', fourcc , 30, (width, height)) # 建立 VideoWriter 物件，輸出影片至 output.avi
# -----------------------------------------------------------
# flag = False
z = 0
while (cap.isOpened()):
    ret,im = cap.read() #第二幀
    if ret==True :
        new_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #灰階化
        # Gus_new_gray = cv2.GaussianBlur(new_gray,(11,11),0) #高斯濾波(11x11)
        # ---------------------------------------------------------------
        flow = cv2.calcOpticalFlowFarneback(old_gray,new_gray,None,0.5,3,15,3,5,1.2,0) #位移矢量 d
        # mag, ang = cv2.cartToPolar(flow[:,:,0],flow[:,:,1]) #笛卡爾坐標轉換為極坐標(只會有正值)
        fx, fy = flow[:,:,0], flow[:,:,1] #dx,dy變化量
        # ang = np.arctan2(fy,fx)*180/np.pi/2 #角度(0~180)
        ang = np.arctan2(fy,fx) #弧度 0~6(0~2π) (tan(ang)=斜率)
        mag = np.sqrt(fx*fx+fy*fy) #得到運動的位移長度
        # ---------------------------------------------------------------
        if (z <= 99) : #取100幀
            for y in range(height) :
                for x in range(width) : 
                    ang_buf[z][y][x] = ang[y][x] #存取100所有資料
                    mag_buf[z][y][x] = mag[y][x] #存取100所有資料
            z += 1
        else :
            break
        # ---------------------------------------------------------------
        cv2.imshow("frame_im",im) #顯示原影像
        old_gray = new_gray #第一幀=第二幀(圖像)
        # ---------------------------------------------------------------  
        if cv2.waitKey(10) == 27:
            break   
        #videoWriter.write(draw_flow(new_gray,flow)) #輸出影片
    else :
        break
# ---------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
#videoWriter.release()
# ---------------------------------------------------------------
ang_list = list() #建立lsit存取資料
mag_list = list() #建立lsit存取資料
ang11 = np.zeros_like(new_gray) #建立二維陣列,存取眾數結果
mag11 = np.zeros_like(new_gray) #建立二維陣列,存取眾數結果
for y in range(height):
    for x in range(width):
        for z in range(99) :
            # ang_list.append(round(ang_buf[z][y][x])) #加入lsit中取眾數
            # mag_list.append(round(mag_buf[z][y][x])) #加入lsit中取眾數
            ang_list.append(ang_buf[z][y][x]) #加入lsit中取眾數
            mag_list.append(mag_buf[z][y][x]) #加入lsit中取眾數
        # ang11[y][x] = np.argmax(np.bincount(ang_list)) # 取眾數,但只能取正整數
        # mag11[y][x] = np.argmax(np.bincount(mag_list)) # 取眾數,但只能取正整數

        print(mode(ang_list))
        # ////////////////////////
        # 用來驗證是否對
        # if (y == 100) :
        #     if (x == 100) :
        #         print(ang_list)
        # ////////////////////////
        ang_list.clear() #清空list資料
        mag_list.clear() #清空list資料
# print(ang11[100][100])

# dx,dy = cv2.polarToCart(float(mag11),float(ang11),angleInDegrees=True)
# print(dx[100][100],dy[100][100])

