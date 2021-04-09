import cv2
import numpy as np


def nothing(no):
    pass


# 加速運算所以先等比例縮小圖片
path = "./count/Demo.JPG"
ori = cv2.imread(path)
re_ori = cv2.resize(ori, (1014, 760), interpolation=cv2.INTER_AREA)
# cv2.imwrite("./count/Demo_resize.jpg", re_ori)
# cv2.imshow("re_ori", re_ori)

# 根據顏色進行過濾
y, x, c = re_ori.shape
img = np.zeros(re_ori.shape, np.uint8)
for j in range(y):
    for i in range(x):
        b, g, r = re_ori[j, i]
        if b > g and b > r:
            img[j, i] = re_ori[j, i]
        else:
            img[j, i] = [0, 0, 0]
# cv2.imwrite("./count/Demo_img1.jpg", img)
# cv2.imshow("img", img)

# 選取 ROI(略過非目標的區域)
img[:, 0:180] = [0, 0, 0]
img[:, 785:] = [0, 0, 0]
img[680:, :] = [0, 0, 0]
# cv2.imwrite("./count/Demo_img2.jpg", img)
# cv2.imshow("img2", img)

# 灰階後 使用色調映射再處理一次
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
# cv2.imwrite("./count/Demo_heat.jpg", heat)
# cv2.imshow("heat", heat)
for j in range(y):
    for i in range(x):
        b, g, r = heat[j, i]
        M = max(b, g, r)
        if b > 150:
            img[j, i] = [0, 0, 0]
cv2.imwrite("./count/Demo_img3.jpg", img)
cv2.imshow("img3", img)

# 濾除雜訊(中值濾波)
md_blur = cv2.medianBlur(img, 3)
# cv2.imshow("median blur", md_blur)

# Sobel
blur_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("blur gray", blur_gray)
# sobel = cv2.Sobel(blur_gray, ddepth=-1, dx=1, dy=0, ksize=3)
# cv2.imshow("sobel", sobel)

# Threshold
# _, thres = cv2.threshold(blur_gray, 70, 210, cv2.THRESH_BINARY)
# cv2.imshow("thres", thres)

# 邊緣檢測(Canny)
# cv2.namedWindow("adj canny")
# cv2.createTrackbar("thres1", "adj canny", 0, 255, nothing)
# cv2.createTrackbar("thres2", "adj canny", 0, 255, nothing)
# cv2.setTrackbarPos("thres1", "adj canny", 90)
# cv2.setTrackbarPos("thres2", "adj canny", 255)
#
# while True:
#     thres1 = cv2.getTrackbarPos("thres1", "adj canny")
#     thres2 = cv2.getTrackbarPos("thres2", "adj canny")
#     canny = cv2.Canny(blur_gray, thres1, thres2)
#     cv2.imshow("adj canny", canny)
#
#     if cv2.waitKey(1) == ord('q'):
#         break

canny = cv2.Canny(blur_gray, 74, 131)
# cv2.imwrite("./count/Demo_canny.jpg", canny)
# cv2.imshow("canny", canny)

# 描繪輪廓
contour, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
draw = cv2.drawContours(re_ori, contour, -1, (0, 255, 255), 1)
cv2.imshow("draw", draw)
cv2.imwrite("./count/Demo_draw.jpg", draw)

# 輪廓面積(wait)
# l = len(contour)
# handle = list()
# for index in range(l):
#     area = cv2.contourArea(contour[index])
#     # print("the Area of", index, "= ", area)
#     if area > 0.0:
#         handle.append(contour[index])
# print(len(handle))
# draw = cv2.drawContours(re_ori, handle, -1, (0, 255, 255), 1)
# cv2.imshow("draw", draw)

cv2.waitKey(0)
cv2.destroyAllWindows()