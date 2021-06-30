import numpy as np
import cv2


def draw_flow(gray, flow, step=5):
    h, w = gray.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape((2, -1)).astype(np.int64)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2).astype(np.int32)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    Draw_img = np.zeros_like(gray)
    for (x1, y1), (x2, y2) in lines:
        if mag[y1, x1] > 5:
            theta = ang[y1, x1] * 180 / np.pi / 2
            print("theta: ", theta)

            if 80 < theta < 180:
                Draw_img[y1, x1] = 255

    cv2.imshow('Draw_img', Draw_img)
    return Draw_img


path = './last.avi'
video = cv2.VideoCapture(path)

width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
print("Image size: %d x %d, %d" % (width, height, fps))

ret, im = video.read()
old_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

while video.isOpened():
    ret, im = video.read()

    if not ret:
        break

    cv2.imshow('im', im)
    new_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    old_gray = new_gray

    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('p'):
        while cv2.waitKey(1) != ord(' '):
            pass
video.release()
cv2.destroyAllWindows()

