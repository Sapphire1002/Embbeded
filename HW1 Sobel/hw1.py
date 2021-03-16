import cv2


# HW1 建環境，把結果截圖，並附上GitHub連結
def output_img(img, text):
    cv2.namedWindow(text, cv2.WINDOW_NORMAL)
    cv2.imshow(text, img)
    cv2.imwrite('%s.png' % text, img)


path = "./road/road.jpg"
road = cv2.imread(path)

road_gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
# output_img(road_gray, text='./road/gray')

road_sobel = cv2.Sobel(road_gray, ddepth=-1, dx=1, dy=0, ksize=3)
# output_img(road_sobel, text="./road/sobel_dx_3x3")

cv2.waitKey(0)
cv2.destroyAllWindows()
