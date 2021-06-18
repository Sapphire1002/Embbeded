import cv2
import numpy as np
import time


def handle_optical_flow(ori_img):
    pass


path = './last.avi'
video = cv2.VideoCapture(path)
cnt = 0
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
kernel = np.ones((5, 5), np.uint8)
bg = None
frame_step = 1

while True:
    ret, frame = video.read()

    if not ret:
        break

    cnt += 1
    cv2.imshow('frame', frame)

    if cnt % frame_step == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)

        if bg is None:
            bg = blur
            continue

        # handle optical flow
        time.sleep(0.1)
        diff = cv2.absdiff(bg, blur)
        _, thres = cv2.threshold(diff, 25, 255,cv2.THRESH_BINARY)
        dilate = cv2.dilate(thres, es, iterations=2)
        bg = blur
        cv2.imshow('diff', diff)
        cv2.imshow('thres', thres)
        cv2.imshow('dilate', dilate)
        cnt = 0

    if cv2.waitKey(1) == ord('q'):
        break

    elif cv2.waitKey(1) == ord('p'):
        while cv2.waitKey(0) != ord(' '):
            pass

cv2.destroyAllWindows()

