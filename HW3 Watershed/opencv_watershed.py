import cv2
import numpy as np


path_ori = "./road/road.jpg"
path = "./road/road_marker.jpg"
road_markers = cv2.imread(path)
road = cv2.imread(path_ori)

hsv = cv2.cvtColor(road_markers, cv2.COLOR_BGR2HSV)
minR = np.array([0, 100, 100])
maxR = np.array([10, 255, 255])
fg = cv2.inRange(hsv, minR, maxR)

gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mb = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel, iterations=2)
bg = cv2.dilate(mb, kernel, iterations=3)

unknown = cv2.subtract(bg, fg)
_, markers = cv2.connectedComponents(fg)
markers = cv2.watershed(road, markers=markers)
road[markers == -1] = [255, 0, 255]

cv2.imshow("road", road)
cv2.imwrite("./road/road_watershed.png", road)
cv2.waitKey(0)
cv2.destroyAllWindows()

