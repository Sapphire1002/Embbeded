import numpy as np
import cv2


def draw_flow(im, flows, step=15):
    h, w = im.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flows[y, x].T

    # create line endpoints
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    # create image and draw
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


path = './video/MOV00271_Trim.mp4'
cap = cv2.VideoCapture(cv2.samples.findFile(path))
_, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
prvs = cv2.GaussianBlur(prvs, (21, 21), 0)
all_frames = list()
print('handle...')

while True:
    ret, frame2 = cap.read()

    if not ret:
        break

    _next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    _next = cv2.GaussianBlur(_next, (21, 21), 0)

    flow = cv2.calcOpticalFlowFarneback(prvs, _next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    all_frames.append(draw_flow(_next, flow))
    # cv2.imshow('Optical flow', draw_flow(_next, flow))
    prvs = _next

    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.waitKey(1) == ord('p'):
        while cv2.waitKey(1) != ord(' '):
            pass

cap.release()
cv2.destroyAllWindows()

output_video = './video/MOV00271_Trim_flow_15_blur.avi'
y, x, _ = all_frames[0].shape
video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MJPG'), 30, (x, y))

for i in all_frames:
    video_writer.write(i)
video_writer.release()
