# 讀取影片做基本測試(範例)
import numpy as np
import cv2
# -------------------------------------------------------------------------------
cap = cv2.VideoCapture('Edit_video_5_yellowline.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print("Image Size: %d x %d , %d" % (width, height, FPS))
# -------------------------------------------------------------------------------
while (cap.isOpened()):
    ret,im = cap.read() #第二幀
    if ret==True :
        # ------------------------------------------------------------------------------
        # 工作區



        # ------------------------------------------------------------------------------
        cv2.namedWindow("im_frame",0)
        cv2.resizeWindow("im_frame", 1920, 1080)
        cv2.imshow("im_frame",im)
        if cv2.waitKey(10) == 27:
            break
    else :
        break
cap.release()
cv2.destroyAllWindows()