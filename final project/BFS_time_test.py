import time
import numpy as np
import cv2

# 輸入一張圖片
path = 'C:/Users/user/Desktop/Embedded/HW4 LBP+Watershed/road.jpg'
img = cv2.imread(path)
re_img = cv2.resize(img, (640, 480), cv2.INTER_AREA)


gray = cv2.cvtColor(re_img, cv2.COLOR_BGR2GRAY)
sobel = cv2.Sobel(gray, ddepth=-1, dx=1, dy=1, ksize=5)

# road coordinate
x, width = 400, 20
y, height = 400, 60
cv2.rectangle(re_img, (x, y), (x + width, y + height), (255, 255, 255), 2)  # road region

# find sample
road_sample = gray[y:y+height, x:x+width]
sky_sample = gray[0:height, :]
cv2.rectangle(re_img, (0, 0), (640, height), (0, 255, 255), 2)  # sky region

# create sign
sign = np.zeros(gray.shape, np.uint8)
# sky: 2, road: 1, searchable: 0
y_size, x_size, _ = re_img.shape
sign[0:height, :] = 2
sign[y:y+height, x:x+width] = 1

# watershed
# sobel = 32 為界線
# 方向: [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
# 起始點 (400, 400), (420, 400), (400, 460), (420, 460)
# 判斷: 只要4個列表階為 'N' 就換下的點
for y_coord in range(y_size):
    for x_coord in range(x_size):
        if sobel[y_coord, x_coord] <= 32:
            pass
        pass
    pass

cv2.imshow('re_img', re_img)
cv2.imshow('sobel', sobel)
cv2.imshow('sample', road_sample)
cv2.waitKey(0)
cv2.destroyAllWindows()
