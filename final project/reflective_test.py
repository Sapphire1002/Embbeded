import numpy as np
import cv2


path = './HSV_HLS_TEST.png'

ori = cv2.imread(path)
ori = cv2.resize(ori, (640, 480), cv2.INTER_AREA)
b, g, r = cv2.split(ori)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

merge = cv2.merge([bH, gH, rH])

# blur = cv2.GaussianBlur(ori, (5, 5), 0)
# gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)

# 嘗試用差幀方式(明天)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# erode = cv2.erode(gray, kernel, iterations=1)
# dilate = cv2.dilate(gray, kernel, iterations=1)

# diff = cv2.subtract(erode, dilate)
# canny = cv2.Canny(diff, 90, 210)

cv2.imshow('b', b)
cv2.imshow('g', g)
cv2.imshow('r', r)
cv2.imshow('merge', merge)
cv2.imshow('ori', ori)
# cv2.imshow('gray', gray)
# cv2.imshow('erode', erode)
# cv2.imshow('dilate', dilate)
# cv2.imshow('diff', diff)

# cv2.imshow('canny', canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
