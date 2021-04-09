import cv2

path = "./road.jpg"
path_markers = "./road_draw.jpg"

road = cv2.imread(path)
road_draw = cv2.imread(path_markers)

road_gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", road_gray)
blur = cv2.GaussianBlur(road_gray, (5, 5), 0)
# cv2.imshow("blur", blur)

_, thres = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
# cv2.imshow("thres", thres)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mb = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel, iterations=2)
bg = cv2.dilate(mb, kernel, iterations=3)
cv2.imshow("bg", bg)

cv2.waitKey(0)
cv2.destroyAllWindows()
