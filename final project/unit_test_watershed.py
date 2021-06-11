from skimage.feature import local_binary_pattern as lbp
import numpy as np
import cv2
import time


# step 1
def preprocess(curr_frame, size=(640, 480), thres_condi=0.2):
    """
    function:
        preprocess(curr_frame[, size=(640, 480)[, thres_condi=0.2]]): 設定 curr_frame 為 size 大小去做預處理

    parameter:
        curr_frame: 當前要處理的幀(一張圖片)
        size: 將圖片設定成 width * height 大小 (width, height), tuple
        thres_condi: 設定二值化的條件, 預設為 0.2, 範圍 [0, 1), float

    method:
        1. 先使用找出影片的 ROI 區域
        (邊界的灰階值如果在 10 以下則濾掉)
        2. 先重新設定 curr_frame 大小
        (判斷該影片為直的或橫的, 根據方向不同做出適合的縮放)
        3. 轉灰階
        4. 高斯濾波
        (kernel 大小設定 5*5, SD = 0)
        5. Sobel
        (dx, dy = 1, kernel 大小設定 5*5)
        6. 二值化
        (f(x) = (max(sobel) - min(sobel)) * thres_condi)
        (255 if sobel >= f(x) else 0)

    return:
        frame: 傳回重塑過後的 3 通道圖片
        frame_sobel: 傳回 sobel 後的圖片
        frame_pre: 傳回二值化的圖片
    """

    # 1.
    gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    gray[gray < 10] = 0

    contour, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_area = list()
    for cnt in contour:
        area = cv2.contourArea(cnt)
        all_area.append(area)

    index = all_area.index(max(all_area))
    del all_area

    x, y, w, h = cv2.boundingRect(contour[index])

    frame_roi = curr_frame[y:y+h, x:x+w]

    # 2.
    re_y, re_x, _ = frame_roi.shape
    size = (size[1], size[0]) if re_y > re_x else (size[0], size[1])

    frame = cv2.resize(frame_roi, size, cv2.INTER_AREA)

    # 3.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5.
    frame_sobel = cv2.Sobel(blur, ddepth=-1, dx=1, dy=1, ksize=5)
    frame_pre = frame_sobel.copy()

    # 6.
    thres = ((np.max(frame_sobel) - np.min(frame_sobel)) * thres_condi).astype(np.uint8)

    frame_pre[frame_sobel >= thres] = 255
    frame_pre[frame_sobel < thres] = 0

    return frame, frame_sobel, frame_pre


# step 2
def handle_sample(ori_img, pre_img, block_size=(20, 60)):
    """
    function:
        handle_sample(ori_img, pre_img[, block_size=(20, 60)]): 尋找符合條件的所有樣本

    parameter:
        ori_img: 輸入 3 通道圖像, 大小和 size 設定的相同
        pre_img: 輸入經過預處理後的(二值化)圖像, 大小和 size 設定的相同
        block_size: 輸入一個 block_size (width, height)的大小, 默認(20, 60), tuple

    method:
        1. target: blocks 的區域 以影片中心區域做上下兩行的 block
        2. 以 0 的值最多的當成 LBP 的 sample, 並把所有符合條件的都當成 sample(最後用 LBP 篩選掉不符合的)
        (經過二值化處理後, 馬路相對數值為 0 的部分最多)
        (取值的時候避免邊界會造成影響, 因此只計算 第二塊到倒數第二塊)

    return:
        gray: 傳回灰階圖像
        coord: 回傳所有 sample 位於原始圖像的位置, (y_coord, x_coord), 皆為左上角的座標, list
        block_size: 回傳一個 blocks 的大小 (width, height), tuple
    """

    y, x = pre_img.shape
    width, height = block_size

    center_y, center_x = y // 2, x // 2  # 中心位置

    # 1. 以影片中心為界線, 上下各取一排 blocks
    # 由於計算只有計算 第二塊 ~ 倒數第二塊, 所以原始圖片的 width 會去掉頭尾, 且切分成 blocks 大小
    target_up = pre_img[center_y - height: center_y, width:x - width]
    target_down = pre_img[center_y: center_y + height, width:x - width]

    blocks_up = np.split(target_up, (x - 2 * width) // width, axis=1)
    blocks_down = np.split(target_down, (x - 2 * width) // width, axis=1)

    # 2.
    # 先找出 0 的數量並且儲存成列表
    val_up_0 = list()
    val_down_0 = list()

    for index in range(0, len(blocks_up)):
        val_up_0.append(np.unique(blocks_up[index], return_counts=True)[1][0])
        val_down_0.append(np.unique(blocks_down[index], return_counts=True)[1][0])

    # 找出所有最大值索引(索引值代表著區塊位置)
    all_max_index_up = [index for index, val in enumerate(val_up_0) if val == max(val_up_0)]
    all_max_index_down = [index for index, val in enumerate(val_down_0) if val == max(val_down_0)]

    # 將索引值轉成各 block 左上角的座標
    coord_up = [(center_y - height, width * (i + 1)) for i in all_max_index_up]
    coord_down = [(center_y, width * (i + 1)) for i in all_max_index_down]
    coord = coord_up + coord_down

    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    return gray, coord, block_size


# step 3
def handle_LBP(gray_img, sample_coord, block_size=(20, 60), similar_condi=0.85):
    """
    function:
        handle_LBP(gray_img, sample_coord[, block_size=(20, 60)[, similar_condi=0.85]]):
        計算原始圖像和 sample 的 LBP 相似度

    parameter:
        gray_img: 調整大小後的灰階圖像
        sample_coord: 所有 sample 位於原始圖像的位置(左上角座標)
        block_size: 和 handle_sample 的 block_size 相同
        similar_condi: 相似度的門檻值, 默認 0.85, 範圍[0, 1), float

    method:
        1. 計算 sample LBP 值以及直方圖
        2. 相似度比較去除最不相似的 sample
        (採取逐一比較的方式, 去除相似度最低的, 其餘的都當成 markers)
        (相似度最低的判斷為 當前相似個數 - 最小相似個數 > 全距 // 2)
        3. 天空全部都標成 marker(最上面一行)
        4. 侵蝕一次

    return:
        markers: 二值化的圖像, 用來當成 watershed 的 markers
    """

    markers = np.zeros(gray_img.shape, np.uint8)
    width, height = block_size

    # 儲存 每張 LBP 值的 直方圖列表
    hist_list = list()

    for coord in sample_coord:
        y, x = coord
        target = gray_img[y:y + height, x:x+width]

        # 1. 計算 LBP 值(為了要使用 cv2.calcHist 函數, 要把圖像的資料類型轉成 np.uint8)
        lbp_img = lbp(target, 8, 1).astype(np.uint8)
        hist = cv2.calcHist([lbp_img], [0], None, [256], [0, 256])
        hist_list.append(hist)

        # 畫出所有 sample 的位置
        cv2.rectangle(gray_img, (x, y), (x + width, y + height), (255, 255, 255), 5)
        # cv2.imshow('sample region', gray_img)

    # print('hist 個數: ', len(hist_list))

    # 2.
    # 存放相似度結果的列表
    similar_list = list()
    for i in range(0, len(hist_list)):
        cnt = -1  # 計算相似個數(採逐一比較, 自己和自己比一定相似, 所以要減一)
        for j in range(0, len(hist_list)):
            sim = cv2.compareHist(hist_list[i], hist_list[j], cv2.HISTCMP_CORREL)
            if sim >= similar_condi:
                cnt += 1
        similar_list.append(cnt)
    # print("similar list: ", similar_list, "個數: ", len(similar_list))

    # 當前的個數 - 最小相似個數 <= sample 數量 // 2
    # markers_coord = list()  # 儲存最後被當成 markers 的座標
    for index, coord in enumerate(sample_coord):
        y, x = coord
        if similar_list[index] - min(similar_list) <= (max(similar_list) - min(similar_list)) // 2:
            # markers_coord.append((y, x))
            markers[y:y+height, x:x+width] = 255
            cv2.rectangle(gray_img, (x, y), (x + width, y + height), (0, 0, 0), 2)

    # 加上天空的座標
    for x in range(0, gray_img.shape[1], width):
        # markers_coord.append((0, x))
        markers[10:10+height, x+width:x-width] = 255

    # cv2.imshow('sample region', gray_img)
    # print(len(markers_coord))

    # 3.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    markers = cv2.erode(markers, kernel, iterations=1)
    # cv2.imshow('markers', markers)
    return markers


# step 4
def handle_watershed(ori_img, marker):
    """
    function(ori_img, markers): 根據 markers 做 watershed

    parameter:
        ori_img: 調整大小後的 3 通道圖像
        marker: 二值化圖像

    method:
        1. marker 做成 connectedComponents()
        (會得到每個區域的標記)
        2. watershed
        3. 將天空和其他標記區域區分顏色
        last: 切割背景區域傳給光流輸入

    return:
        res: Watershed 出來的結果
        ori_img: 裁減過後的原始圖像(對應光流的輸入)
    """

    # 1.
    _, markers = cv2.connectedComponents(marker)

    # 2.
    sign = cv2.watershed(ori_img, markers)
    # sign, count = np.unique(sign, return_counts=True)
    # print(sign, count)

    # 3.
    color_block = np.zeros(ori_img.shape, np.uint8)
    color_block[sign == -1] = [0, 0, 0]
    color_block[sign == 1] = [192, 62, 255]
    color_block[sign != 1] = [0, 255, 0]
    res = cv2.addWeighted(ori_img, 1, color_block, 0.4, 0)
    # cv2.imshow('res', res)

    # last: 切掉背景區域傳給光流輸入
    ori_img[sign == 1] = [0, 0, 0]
    # cv2.imshow('last', ori_img)
    return res, ori_img


def main(path, frame_step=1):
    """
    function:
        main(path, frame_step=1): 讀取影片並且取每隔 frame_step 的影像做處理

    parameter:
        path: 影片路徑, str
        frame_step: 取每隔frame_step 幀數, 默認1, 範圍 fps >= frame_step > 0, int

    return: None
    """
    cnt = 0
    video = cv2.VideoCapture(path)

    while True:
        st = time.time()
        ret, frame = video.read()

        if not ret:
            # repeat
            video = cv2.VideoCapture(path)
            continue
            # break

        cnt += 1
        frame, sobel, frame_pre = preprocess(frame)  # preprocess
        cv2.imshow('frame', frame)
        cv2.imshow('preprocess', frame_pre)

        if cnt % frame_step == 0:
            gray, coord, block_size = handle_sample(frame, frame_pre, (20, 60))  # handle sample
            markers = handle_LBP(gray, coord, block_size)  # handle LBP
            res, last = handle_watershed(frame, markers)  # handle watershed
            end = time.time()

            # 顯示輸出結果以及計算時間的文字
            handle_time = round(end - st, 3)
            handle_fps = int(1 // (end - st))
            str_handle_time = "handle time(s/frame): " + str(handle_time) + 's'
            str_handle_fps = 'handle(' + str(handle_fps) + 'frame/s)'
            cv2.putText(frame, str_handle_time, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)
            cv2.putText(
                frame, str_handle_fps, (10, 40),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 255), 1
            )

            # cv2.imshow('frame', frame)
            cv2.imshow('res', res)
            cv2.imshow('last', last)

            cnt = 0

        if cv2.waitKey(1) == ord('q'):
            break

        if cv2.waitKey(1) == ord('p'):
            while cv2.waitKey(0) != ord(' '):
                pass


if __name__ == '__main__':
    file = './video/2018-12-21_09-22-53_1811-Cam2.avi'
    frame_cnt = 1

    main(file, frame_cnt)

# 到時候寫成 class 形式(可以減少一堆座標及大小的參數傳遞, 也可以減少 list 存放的空間)
