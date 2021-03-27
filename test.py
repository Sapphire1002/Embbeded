import numpy as np
import cv2 as cv

img = cv.imread('C:/Users/iris2/Desktop/embbeded/road.jpg')
img2 = cv.imread("C:/Users/iris2/Desktop/embbeded/line.png")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
_, thresh2 = cv.threshold(gray2, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# cv.imshow("t2", thresh2)

# 噪聲去除
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# 確定背景區域
sure_bg = cv.dilate(opening, kernel, iterations=3)

# 尋找前景區域
# dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
# _, sure_fg = cv.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)

# 找到未知區域
sure_fg = np.uint8(gray2)
unknown = cv.subtract(sure_bg, sure_fg)
cv.imshow("bg", sure_bg)
cv.imshow("fg", sure_fg)
cv.imshow("un", unknown)

# 類別標記
_, markers = cv.connectedComponents(sure_fg)
print(markers.shape)

# 為所有的標記加1，保證背景是0而不是1
markers = markers + 1

# 現在讓所有的未知區域為0
markers[unknown == 255] = 0

markers = cv.watershed(img, markers)
img[markers == -1] = [0, 255, 255]

cv.namedWindow("Result", cv.WINDOW_NORMAL)
cv.imshow("Result", img)
cv.waitKey(0)
cv.destroyAllWindows()


