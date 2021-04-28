from my_alg import *
import numpy as np
import cv2


def preprocess(img):
    myalg = ImgAlg()

    # 使用顏色區分 HSV
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.dilate(blur, kernel, iterations=5)
    morph = cv2.erode(dilate, kernel, iterations=5)

    hsv = cv2.cvtColor(morph, cv2.COLOR_BGR2HSV)

    mask_B = myalg.colorRange(hsv, 'blue')
    mask_Gr = myalg.colorRange(hsv, 'gray', optrange=(np.array([35, 20, 46]), np.array([180, 43, 220])))
    img_mask = mask_B + mask_Gr

    morph[img_mask != 255] = [0, 0, 0]
    gray = cv2.cvtColor(morph, cv2.COLOR_BGR2GRAY)

    # applyColorMap
    color_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    hsv = cv2.cvtColor(color_map, cv2.COLOR_BGR2HSV)

    mask_Y = myalg.colorRange(hsv, 'yellow')
    mask_Or = myalg.colorRange(hsv, 'orange')
    map_img_mask = mask_Y + mask_Or
    gray[map_img_mask != 255] = 0

    # cv2.imwrite('./Demo_img_mask_gray3.jpg', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return gray


def drawSolar(ori_img, img_gray):
    myalg = ImgAlg()

    blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    # thres1, thres2, method = myalg.adjust_threshold(
    #     img=blur,
    #     name='adj thres',
    #     param1='thres1',
    #     param2='thres2',
    #     param1_lim=(0, 255),
    #     param2_lim=(0, 255),
    #     method=cv2.THRESH_BINARY
    # )

    # threshold
    thres1, thres2 = 95, 255
    method = cv2.THRESH_BINARY
    _, thres = cv2.threshold(blur, thres1, thres2, method)

    # drawContour
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ori_img, contours, -1, (0, 255, 255), 2)

    # cv2.imwrite('./Demo_img_draw.jpg', draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return contours, ori_img


def counter(contours, draw_img):
    # 找已知是太陽能板的位置, 算出一塊太陽能板的面積
    x1, x2 = 805, 830
    y1, y2 = 465, 515
    find_solar = draw_img.copy()

    cv2.rectangle(find_solar, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imwrite("./Demo_find.jpg", find_solar)

    # 算輪廓面積除單塊的求數量
    target_area = (x2 - x1) * (y2 - y1)
    total_areas = 0
    length = len(contours)

    for index in range(0, length):
        area = cv2.contourArea(contours[index])
        total_areas += area

    nums = (total_areas // target_area) + 1
    return int(nums)


if __name__ == '__main__':
    file = "./Demo.JPG"
    ori = cv2.imread(file)
    handle_gray = preprocess(ori)
    contour, draw = drawSolar(ori, handle_gray)
    counts = counter(contour, draw)
    print(counts)

