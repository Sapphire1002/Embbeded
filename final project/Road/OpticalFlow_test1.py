#OpenCV中的LK光流(範例)
import numpy as np
import cv2

cap = cv2.VideoCapture('ATN-1036_CH0220190622115005-R.avi')
########################################################################################################
# ShiTomasi corner detection的参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

#cv2.goodFeaturesToTrack(image,         #单通道
#                        maxCorners,    #角点数目最大值，若检测的角点超过此值，则只返回前maxCorners个强角点
#                        qualityLevel,  #角点的品质因子
#                        minDistance,   #如果在其周围minDistance范围内存在其他更强角点，则将此角点删除 
#                        corners        #存储所有角点
#                        mask,          #指定感兴趣区，若无指定，寻找全图
#                        blockSize,     #计算协方差矩阵时的窗口大小
#                        useHarrisDetector,  #bool 是否使用Harris角点检测，如不指定，则计算shi-tomasi角点
#                        k )           #Harris角点检测需要的k值
########################################################################################################
# 光流法参数
# maxLevel 未使用的图像金字塔层数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#nextPts,status,err = cv2.calcOpticalFlowPyrLK(prevImg,   #上一帧图片
#                                              nextImg,   #当前帧图片
#                                              prevPts,   #上一帧找到的特征点向量 
#                                              nextPts    #与返回值中的nextPtrs相同
#                                              [, status[, err[, winSize
#                                              [, maxLevel[, credxc iteria
#                                              [, flags[, minEigThreshold]]]]]]])
#返回值：
#nextPtrs 输出一个二维点的向量，这个向量可以是用来作为光流算法的输入特征点，也是光流算法在当前帧找到特征点的新位置（浮点数）
#status 标志，在当前帧当中发现的特征点标志status==1，否则为0
#err 向量中的每个特征对应的错误率
#其他输入值：
#status 与返回的status相同
#err 与返回的err相同
#winSize 在计算局部连续运动的窗口尺寸（在图像金字塔中）
#maxLevel 图像金字塔层数，0表示不使用金字塔
#criteria 寻找光流迭代终止的条件
#flags 有两个宏，表示两种计算方法，
#OPTFLOW_USE_INITIAL_FLOW表示使用估计值作为寻找到的初始光流，
#OPTFLOW_LK_GET_MIN_EIGENVALS表示使用最小特征值作为误差测量
#minEigThreshold 该算法计算光流方程的2×2规范化矩阵的最小特征值，除以窗口中的像素数; 如果此值小于minEigThreshold，则会过滤掉相应的功能并且不会处理该光流，因此它允许删除坏点并获得性能提升。
########################################################################################################
# 创建随机生成的颜色
color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()                             # 取出视频的第一帧

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # 灰度化

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)                         # 为绘制创建掩码图片

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 计算光流以获取点的新位置
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # 选择good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # 绘制跟踪框
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    k = cv2.waitKey(30)  # & 0xff
    if k == 27:
        break
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
