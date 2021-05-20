import numpy as np
import cv2


# step 1
def preprocess(curr_frame, size=(640, 480), thres_condi=0.32):
    """
    function:
        preprocess(curr_frame[, size=(640, 480)[, thres_condi=0.32]]): 設定 curr_frame 為 size 大小去做預處理

    parameter:
        curr_frame: 當前要處理的幀(一張圖片)
        size: 將圖片設定成 width * high 大小 (width, high), tuple
        thres_condi: 設定二值化的條件, 預設為 0.32, 範圍 [0, 1), float

    method:
        1. 先重新設定 curr_frame 大小
        2. 轉灰階
        3. 高斯濾波
        (kernel 大小設定 5*5, SD = 0)
        4. Sobel
        (dx, dy = 1, kernel 大小設定 5*5)
        5. 二值化
        (f(x) = (max(sobel) - min(sobel)) * thres_condi)
        (255 if sobel >= f(x) else 0)

    return:
        frame_ori: 傳回重塑過後的灰階圖片
        frame_pre: 傳回二值化的圖片
    """

    # 1.
    frame = cv2.resize(curr_frame, size, cv2.INTER_AREA)

    # 2.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4.
    sobel = cv2.Sobel(blur, ddepth=-1, dx=1, dy=1, ksize=5)
    frame_pre = sobel.copy()

    # 5.
    thres = ((np.max(sobel) - np.min(sobel)) * thres_condi).astype(np.uint8)

    frame_pre[sobel >= thres] = 255
    frame_pre[sobel < thres] = 0

    return gray, frame_pre


# step 2
def handle_sample(gray_img, pre_img, block_size=(20, 60)):
    """
    function:
        handle_sample(gray_img, pre_img[, block_size=(20, 60)]):

    parameter:
        gray_img: 輸入灰階圖像
        pre_img: 輸入經過預處理後的(二值化)圖像
        block_size: 輸入一個 block_size 的大小(width, high), tuple

    method:
        1. target -> 切分成 block 後默認處理圖片的倒數第二行
        2. 以 0 的值最多的當成 LBP 的 sample
        (經過二值化處理後, 馬路相對數值為 0 的部分最多)
        (取值的時候避免邊界會造成影響, 因此只計算 第二塊到倒數第二塊)

    return:
        sample: 輸出灰階圖像, 大小為 block_size 的值
        coord: sample 位於原始圖像的哪個位置, (y_coord, x_coord), 皆為左上角的座標, tuple
        (供 計算 LBP 時使用)
    """

    y, x = pre_img.shape
    width, high = block_size

    # 1.
    target = pre_img[y - 2 * high:y - high, :]
    blocks = np.split(target, x // width, axis=1)  # 會給一個列表, 存放所有的 block

    # 2.
    # 由於計算只有計算 第二塊 ~ 倒數第二塊, 所以原始圖片的 width 會去掉頭尾
    ori_target = gray_img[y - 2 * high:y - high, width:x - width]

    value_0 = list()
    for index in range(1, len(blocks) - 1):
        value_0.append(np.unique(blocks[index], return_counts=True)[1][0])

    max_index = value_0.index(max(value_0))

    sample = ori_target[:, width * max_index:width * (max_index+1)]

    # 由於 sample 的位置是已經裁減過的, 因此要傳回沒有裁減過的座標, x 軸要多一個 block 的寬度
    coord = (y - 2 * high, width * (max_index + 1))

    return sample, coord


# step 3
def calc_lbp(img):
    """
    function:
        calc_lbp(img): 輸入要計算的 img LBP 值

    parameter:
        img: 灰階圖像

    method:
        1. 方向採去正右為 最高位元, 順時針到右上為最低為元
        2. sample 的邊界區域由灰階值替代
        (目前計算 LBP 方式為 3*3)
        (相對於標線, 馬路的灰階值會較低, 所以使用灰階值替代)

    return:
        lbp_img: 存放 LBP 值的圖像, size 為 img 大小
    """

    y, x = img.shape

    # 1. 計算 LBP
    # 方向
    pos = np.array([
        [2, 1, 0],
        [3, 0, 7],
        [4, 5, 6]
    ])

    lbp_img = img.copy()
    for j in range(0, y - 2):
        for i in range(0, x - 2):
            target = img[j:j+3, i:i+3]
            bits = np.where(target > target[1, 1], 1, 0)
            weight = np.power(2, pos)
            lbp_img[j, i] = np.sum(weight * bits)

    return lbp_img


def handle_LBP(gray_img, sample, coord, similar_condi=0.85):
    """
    function:
        handle_LBP(gray_img, sample[, similar_condi=0.85]): 計算原始圖像和 sample 的 LBP 相似度

    parameter:
        gray_img: 調整大小後的灰階圖像
        sample: 做為比對 LBP 的樣本
        similar_condi: 相似度的門檻值

    method:
        1. 計算 sample LBP 值
        (單元測試, 被當成 sample 的區域畫在影片上)
    return:
        markers: 二值化的圖像, 用來當成 watershed 的 markers
    """

    sample_LBP = calc_lbp(sample)
    y, x = sample_LBP.shape
    cv2.rectangle(gray_img, (coord[1], coord[0]), (coord[1] + x, coord[0] + y), (255, 255, 255), 3)
    cv2.imshow('sample_lbp', sample_LBP)
    cv2.imshow('sample region', gray_img)
    pass


def main(path, frame_step=1):
    """
    function:
        main(path, frame_step=1): 讀取影片並且取每隔 frame_step 的影像做處理

    parameter:
        path: 影片路徑, str
        frame_step: 取第 frame_step 幀數, int(fps >= frame_step > 0)

    return: None
    """
    cnt = 0
    video = cv2.VideoCapture(path)

    while True:
        ret, frame = video.read()

        if not ret:
            # repeat
            video = cv2.VideoCapture(path)
            continue
            # break

        cnt += 1
        gray, frame_pre = preprocess(frame, (640, 480))
        cv2.imshow('gray', gray)
        cv2.imshow('preprocess', frame_pre)

        if cnt % frame_step == 0:
            sample, coord = handle_sample(gray, frame_pre, (20, 60))  # handle sample
            cv2.imshow('sample', sample)
            handle_LBP(gray, sample, coord)  # handle LBP
            cnt = 0

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    file = 'E:\\MyProgramming\\Python\\Project\\implement\\embedded final project\\video\\Edit_video_3.mp4'
    frame_cnt = 5

    main(file, frame_cnt)
