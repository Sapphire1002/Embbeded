import cv2
import numpy as np
from my_alg import ImgAlg as Alg

def nothing(no):
    pass


# 加速運算所以先等比例縮小圖片
path = "./count/Demo.JPG"
ori = cv2.imread(path)
# print(ori.shape)  # (3040, 4056, 3)
ori = cv2.resize(ori, (2028, 1570), interpolation=cv2.INTER_AREA)
ori_draw = ori.copy()
# cv2.imshow("ori", ori)

y, x, c = ori.shape
img = np.zeros(ori.shape, np.uint8)
for j in range(y):
    for i in range(x):
        b, g, r = ori[j, i]
        if b > g and b > r:
            img[j, i] = ori[j, i]
# cv2.imwrite("./count/Demo_img01.jpg", img)
# cv2.imshow("img", img)


# hsv
h = Alg()
blur = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
erode = cv2.erode(hsv, kernel, iterations=2)
dilate = cv2.dilate(hsv, kernel, iterations=2)
# cv2.imwrite("./count/Demo_hsv01.jpg", hsv)
# cv2.imshow("hsv", hsv)

B = h.colorRange(dilate, "blue")
Gr = h.colorRange(erode, "gray")
solar = Gr
# img = Gr + B
# cv2.imwrite("./count/Demo_B.jpg", solar)
# cv2.imshow("mask_img", solar)

contour, hierarchy = cv2.findContours(solar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
draw = cv2.drawContours(ori_draw, contour, -1, (0, 255, 255), 1)
# cv2.imshow("draw", draw)
cv2.imwrite("./count/Demo_draw02.jpg", draw)


# gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5, 5), 0)

# name = "adj thres"
# cv2.namedWindow(name)
# cv2.createTrackbar("thres1", name, 0, 255, h.nothing)
# cv2.createTrackbar("thres2", name, 0, 255, h.nothing)
#
# while True:
#     thres1 = cv2.getTrackbarPos("thres1", name)
#     thres2 = cv2.getTrackbarPos("thres2", name)
#     _, thres = cv2.threshold(blur, thres1, thres2, cv2.THRESH_BINARY)
#     cv2.imshow(name, thres)
#
#     if cv2.waitKey(1) == ord('q'):
#         break

# 描繪輪廓




cv2.waitKey(0)
cv2.destroyAllWindows()
