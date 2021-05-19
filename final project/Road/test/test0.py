# 剪取影片 (範例)
import cv2
import numpy as np
# ---------------------------------------------------------
cap = cv2.VideoCapture('ATN-1036_CH0220190622115005-R.avi')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height, FPS))
# ---------------------------------------------------------
# !! 測試前檢查檔名，以免覆蓋之前影片 !!
fourcc = 0x00000021
cv2.VideoWriter_fourcc('H', '2', '6', '4')
videoWriter = cv2.VideoWriter('Edit_video_5.mp4',fourcc ,30, (width,height))
# ---------------------------------------------------------
# 設定參數值
cnt = 0
video_max = 284
video_min = 282
# ---------------------------------------------------------
while (cap.isOpened()):   
    ret,im = cap.read()
    if ret==True :
        # ------------------------------------------------------------------------------
        #工作區
        if (video_max > cnt >= video_min) :
            videoWriter.write(im) #輸出影片 要等...
        elif (cnt > video_max) :
            break     
        cnt += 1
        # ------------------------------------------------------------------------------
    else :
        break
# ---------------------------------------------------------
cap.release()
videoWriter.release()
cv2.destroyAllWindows()