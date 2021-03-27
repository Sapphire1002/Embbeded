from scipy import signal
import cv2
import numpy as np


def compare(img, sample, high=32, width=8):
    # 儲存指定區塊的sample
    y, x = sample
    s = img[y:y+high, x:x+width]

    # 統計 sample 的 LBP histogram feature vector


    pass


def bit_to_int(matrix):
    # 傳入 3x3  將二進制值轉成十進制, 傳回十進制值
    pos = np.array([[3, 4, 5], [2, 0, 6], [1, 0, 7]])
    weight = np.power(2, pos)
    lbp_value = np.sum(weight * matrix) - 1
    return lbp_value


def lbp(img):
    # 邊緣填充 0
    y, x = img.shape
    con_img = np.zeros((y+2, x+2), dtype=np.uint16)
    con_img[1:y+1, 1:x+1] = img

    # 儲存 lbp image
    lbp_img = np.zeros(img.shape, dtype=np.uint8)

    # 取值
    for j in range(0, y):
        for i in range(0, x):
            # 取 3x3 區塊
            target = con_img[j:j+3, i:i+3]
            bits = np.where(target >= target[1, 1], 1, 0)
            lbp_img[j, i] = bit_to_int(bits)
    return lbp_img


path = "C:/Users/iris2/Desktop/embbeded/HW3/road.jpg"
road = cv2.imread(path)

gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
sobel = cv2.Sobel(gray, ddepth=-1, dx=0, dy=1, ksize=3)
print(sobel)

# LBP 程式
# road_lbp = lbp(gray)
# compare(road_lbp, sample=(860, 60))

cv2.imshow("test", sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# region 32x8
