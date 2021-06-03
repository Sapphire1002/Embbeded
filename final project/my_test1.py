import cv2
import numpy as np
import alg
from skimage.feature import local_binary_pattern as lbp


def handle(img_gray, lbp_img, block_size):
    y, x = block_size
    iy, ix = lbp_img.shape

    # 倒數第二列
    target = lbp_img[iy-2*y:iy-y, :]
    ori_target = img_gray[iy-2*y:iy-y, :]
    cv2.imshow('target', target)
    cv2.imshow('ori_target', ori_target)

    # 等分切成 block_size
    blocks = np.split(target, ix//x, axis=1)
    # blocks[0].astype(np.uint8)

    for index in range(0, len(blocks)):
        pass

    # print(np.unique(blocks, return_counts=True))
    cv2.imshow('blocks', blocks[0])

    # 計算全部 block 的直方圖
    # 先試用一個
    # hist = cv2.calcHist([blocks[0]], [0], None, [256], [0, 255])
    # cv2.imshow('block[0] hist', hist)


path = 'E:\\MyProgramming\\Python\\Project\\implement\\embedded final project\\video\\Edit_video_4.mp4'
cap = cv2.VideoCapture(path)
frame_cnt = 5
cnt = 0

while True:
    ret, frame = cap.read()
    if not ret:
        # cap = cv2.VideoCapture(path)
        # continue
        break

    cnt += 1
    frame = cv2.resize(frame, (640, 480), cv2.INTER_AREA)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    frame_sobel = cv2.Sobel(frame_blur, ddepth=-1, dx=1, dy=1, ksize=5)
    _, frame_thres = cv2.threshold(frame_sobel, 35, 255, cv2.THRESH_BINARY)
    print(np.unique(frame_sobel, return_counts=True))
    # _, frame_otsu = cv2.threshold(frame_blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mb = cv2.morphologyEx(frame_otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    # bg = cv2.dilate(mb, kernel, iterations=5)

    cv2.imshow('frame_gray', frame_blur)
    # cv2.imshow('frame_otsu', frame_otsu)
    cv2.imshow('frame_sobel', frame_sobel)
    cv2.imshow('frame_thres', frame_thres)
    # cv2.imshow('mb', mb)
    # cv2.imshow('bg', bg)

    if cnt % frame_cnt == 0:
        # frame_lbp2 = alg.lbp(frame_gray, 1)
        # frame_lbp2 = lbp(frame_thres, 8, 1, method='default')
        handle(frame, frame_thres, (60, 20))
        cnt = 0
        # cv2.imshow('frame_lbp2', frame_lbp2)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
