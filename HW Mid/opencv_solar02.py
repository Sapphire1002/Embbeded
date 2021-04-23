import cv2
import numpy as np


def nothing(x):
    pass


path = './count/DJI_0002.JPG'
ori = cv2.imread(path)
# print(ori.shape)  # (2160, 3840, 3)

re_ori = cv2.resize(ori, (1280, 720), interpolation=cv2.INTER_AREA)
# cv2.imwrite("./count/DJI_0002_resize.jpg", re_ori)
# cv2.imshow("re_ori", re_ori)

# 顏色區分
# 用原圖的 BGR 處理
y, x, h = re_ori.shape
img = np.zeros(re_ori.shape, np.uint8)
for j in range(y):
    for i in range(x):
        b, g, r = re_ori[j, i]
        M = max(b, g, r)
        if M != b or b < 85:
            img[j, i] = [255, 255, 255]
        else:
            img[j, i] = re_ori[j, i]
# cv2.imshow("img", img)

# 用 heat 處理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
# cv2.imwrite("./count/DJI_0002_heat.jpg", heat)
# cv2.imshow("heat", heat)

for j in range(y):
    for i in range(x):
        b, g, r = heat[j, i]
        M = max(b, g, r)
        if M == r:
            img[j, i] = [255, 255, 255]
        elif M != g:
            img[j, i] = [255, 255, 255]
        else:
            img[j, i] = re_ori[j, i]
# cv2.imshow("img2", img)

# 選取 ROI
img[:, :350] = [255, 255, 255]
img[:, 1170:] = [255, 255, 255]
# cv2.imwrite("./count/DJI_0002_img.jpg", img)
# cv2.imshow("img3", img)

# 灰階背景差分
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# erode = cv2.erode(gray, kernel, iterations=1)
# dilate = cv2.dilate(gray, kernel, iterations=1)
# diff = cv2.absdiff(erode, dilate)
# cv2.imshow("diff", diff)

# cv2.namedWindow("adj canny")
# cv2.createTrackbar("thres1", "adj canny", 0, 255, nothing)
# cv2.createTrackbar("thres2", "adj canny", 0, 255, nothing)
# cv2.setTrackbarPos("thres1", "adj canny", 90)
# cv2.setTrackbarPos("thres2", "adj canny", 255)
#
# while True:
#     thres1 = cv2.getTrackbarPos("thres1", "adj canny")
#     thres2 = cv2.getTrackbarPos("thres2", "adj canny")
#     canny = cv2.Canny(gray, thres1, thres2)
#     cv2.imshow("adj canny", canny)
#
#     if cv2.waitKey(1) == ord('q'):
#         break

canny = cv2.Canny(gray, 75, 90)
# cv2.imwrite("./count/DJI_0002_canny.jpg", canny)
# cv2.imshow("canny", canny)

# 描繪輪廓
contour, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
draw = cv2.drawContours(re_ori, contour, -1, (0, 255, 255), 1)
# cv2.imwrite("./count/DJI_0002_draw.jpg", draw)
# cv2.imshow("draw", draw)

#
for i in range(len(contour)):
    print(i, ":", cv2.contourArea(contour[i]))

cv2.waitKey(0)
cv2.destroyAllWindows()
