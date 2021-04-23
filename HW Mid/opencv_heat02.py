import cv2
import numpy as np

path = './temp/15.JPG'
ori = cv2.imread(path)
ori_draw = ori.copy()

blur = cv2.GaussianBlur(ori, (5, 5), 0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
# cv2.imshow("hsv", hsv)
# cv2.imwrite("./temp/15_hsv.jpg", hsv)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
erode = cv2.erode(hsv, kernel, iterations=1)
minR = np.array([0, 100, 100])
maxR = np.array([10, 255, 255])
mask_R = cv2.inRange(erode, minR, maxR)
# cv2.imwrite("./temp/15_morph_mask_R.jpg", mask_R)
cv2.imshow("mask_R", mask_R)

contour, hierarchy = cv2.findContours(mask_R.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# draw = cv2.drawContours(ori_draw, contour, -1, (255, 255, 0), 2)
# cv2.imshow("draw", draw)
# cv2.imwrite("./temp/15_draw_R.jpg", draw)

l = len(contour)
similar_counts = np.zeros(l)
for index in range(l):
    count = 0
    current = contour[index]
    for i in range(l):
        sim = cv2.matchShapes(current, contour[i], 1, 0.0)
        print(index, ":", sim)
        if sim <= 0.15:
            count += 1
    similar_counts[index] = count - 1
print(similar_counts)
compare_contour = list()
for i in range(len(similar_counts)):
    if similar_counts[i] >= 3 and cv2.contourArea(contour[i]) > 50:
        compare_contour.append(contour[i])

draw = cv2.drawContours(ori_draw, compare_contour, -1, (255, 255, 0), 2)
cv2.imwrite("./temp/15_current_result.jpg", draw)
cv2.imshow("draw", draw)

cv2.waitKey(0)
cv2.destroyAllWindows()
