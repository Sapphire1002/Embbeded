# 讀取圖片做基本測試(範例)
import numpy as np
import cv2
# -------------------------------------------------------------------------------
img = cv2.imread('yellowline.jpg')
# -------------------------------------------------------------------------------
# 工作區






#-------------------------------------------------------------------------------
cv2.namedWindow("im_frame",0)
cv2.resizeWindow("im_frame", 1920, 1080)
cv2.imshow("im_frame",img)
#-------------------------------------------------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()