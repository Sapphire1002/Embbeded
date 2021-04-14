import cv2
import numpy as np


path = "./temp/5.JPG"
ori = cv2.imread(path)
ori_draw = ori.copy()

blur = cv2.GaussianBlur(ori, (5, 5), 0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)
# cv2.imwrite("./temp/5_hsv.jpg", hsv)

minR = np.array([0, 100, 100])
maxR = np.array([10, 255, 255])
mask_R = cv2.inRange(hsv, minR, maxR)
# cv2.imwrite("./temp/5_morph_mask_R.jpg", mask_R)
cv2.imshow("mask_R", mask_R)

contour, hierarchy = cv2.findContours(mask_R.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 用形狀比對
l = len(contour)
similar_counts = np.zeros(l)
for index in range(l):
    count = 0
    current = contour[index]
    for i in range(l):
        sim = cv2.matchShapes(current, contour[i], 1, 0.0)
        print(index, ":", sim)
        if sim <= 0.2:
            count += 1
    similar_counts[index] = count - 1
print(similar_counts)
compare_contour = list()
for i in range(len(similar_counts)):
    if similar_counts[i] in range(4, 10) and 160 > cv2.contourArea(contour[i]) > 55:
        compare_contour.append(contour[i])

draw = cv2.drawContours(ori_draw, compare_contour, -1, (255, 255, 0), 2)
cv2.imshow("draw", draw)

cv2.waitKey(0)
cv2.destroyAllWindows()
