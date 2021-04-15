import matplotlib.pyplot as plt
import cv2
import numpy as np
import time


# Target: Using LBP algorithm and take out the road
# LBP algorithm (做分類器)
# 1. 原始圖片轉灰階且設定 blocks 大小 (3x3) <- 改採用設定 r  OK
# 2. 取正中間點的值當成 threshold, 和周圍的區域進行比較(i if region > threshold else 0)  OK
# 3. 將周圍的區域按照順時針或逆時針旋轉並且以二進制(8bits)轉成 10 進制相加後, 計算後的值為該中心點的 LBP 值  OK
# 4. 做卷積運算, 得到這張圖片的所有的 LBP 值  OK
# 5. 繪製直方圖, x軸為 0~255, y軸為出現次數  OK
# 6. 將統計後的直方圖換成一個特徵向量(LBP 紋路特徵向量), 接著可用 SVM 等 ML 進行分類

# 單張圖片
# 1~4 都相同  OK
# 5. 設定 cells 大小 (目前 10x10)  OK
# 6. 根據 cells 大小迭代 LBP 值的圖片做出該 cells 的統計圖 (x, y 軸和分類器一樣)  OK
# 7. 拿一張道路的樣本和整張圖片比較, 相似則給1, 否則 0 ??
# Ex. 描繪邊界


def output_img(img, text):
    """輸出圖像並儲存到指定目錄"""
    cv2.namedWindow(text, cv2.WINDOW_NORMAL)
    cv2.imshow(text, img)
    cv2.imwrite('%s.png' % text, img)


class MyLBP(object):
    def __init__(self, img):
        """初始化數值
        self.img: 輸入灰階圖片, 類型 numpy.ndarray
        self.y, self.x: 圖片的大小 length * width, 類型 int
        self.size: 存放 block 的大小, 類型 int
        self.lbp_val: 存放整張圖片的且統計過的 LBP 值, 類型 numpy.ndarray
        self.cells: 存放 cells 大小, 類型 int
        self.cell_val: 存放某個特定區域的 LBP 統計值, 類型 numpy.ndarray
        self.cell_top_all: 存放全部 cells 區域內前五個最大的 LBP 統計值, 類型 numpy.ndarray
        self.lbp_img: 存放每個 pixel 的 LBP 值, 類型 numpy.ndarray
        self.start: 用來計算執行時間
        self.rotate_labels: 旋轉時當作一樣圖形, 類型 list, 元素 int
        """

        self.img = img
        self.y, self.x = img.shape
        self.size = None
        self.lbp_val = np.zeros(256, dtype=np.uint32)
        self.cells = None
        self.cell_val = np.zeros(256, dtype=np.uint8)
        self.cell_top_all = None
        self.lbp_img = np.zeros(img.shape, dtype=np.uint8)
        self.start = time.time()
        self.rotate_labels = [
                [0],
                [128, 64, 32, 16, 8, 4, 2, 1],
                [192, 96, 48, 24, 12, 6, 3, 129],
                [224, 112, 56, 28, 14, 7, 131, 193],
                [240, 120, 60, 30, 15, 135, 195, 225],
                [248, 124, 62, 31, 143, 199, 227, 241],
                [252, 126, 63, 159, 207, 231, 243, 249],
                [254, 128, 191, 223, 239, 247, 251, 253],
                [255]
            ]

    def compare(self, road_y=None, road_x=None):
        """拿一個 cells 當樣本來和圖的其他地方比對
        road_y: 第 y 個的 cells 位置
        road_x: 第 x 個的 cells 位置
        """
        y, x, _, _ = self.cell_top_all.shape
        sample = self.cell_top_all[road_y, road_x]  # 該區塊的 LBP 前五個最大值
        res_img = np.zeros((self.y, self.x), dtype=np.uint8)

        sample = np.sort(sample[0])[4]
        print(sample)

        # 用 sample 迭代 cell_top_all 寫相似演算法, 一樣給1, 反之0
        # 是馬路給 1, 不是馬路給 0
        for j in range(0, y):
            for i in range(0, x):
                color_error = abs(sample - np.sort(self.cell_top_all[j, i, 0])[4])

                y1 = j * self.cells
                y2 = y1 + self.cells
                x1 = i * self.cells
                x2 = x1 + self.cells

                print(color_error)
                if np.sum(color_error) == 0:
                    res_img[y1:y2, x1:x2] = self.img[y1:y2, x1:x2]
                else:
                    res_img[y1:y2, x1:x2] = 255

        return res_img

    def bit_to_int(self, bits_matrix, rotate):
        """處理 3x3 矩陣 二進制轉十進制
        以逆時針旋轉方式, 方向如下:
        右下, 右, 右上, 上, 左上, 左, 左下, 下

        考慮旋轉(10 進制表示):
        0: 0
        1: 128, 64, 32, 16, 8, 4, 2, 1
        2: 192, 96, 48, 24, 12, 6, 3, 129
        3: 224, 112, 56, 28, 14, 7, 131, 193
        4: 240, 120, 60, 30, 15, 135, 195, 225
        5: 248, 124, 62, 31, 143, 199, 227, 241
        6: 252, 126, 63, 159, 207, 231, 243, 249
        7: 254, 128, 191, 223, 239, 247, 251, 253
        8: 255

        不考慮旋轉:
        bits_matrix: 傳入 3x3 的二值矩陣
        return: lbp_value, 十進制整數
        """

        # 不考慮旋轉, 直接乘權重(目前是寫死的狀態)
        pos = np.array([[3, 4, 5], [2, 0, 6], [1, 0, 7]])
        weight = np.power(2, pos)
        lbp_value = np.sum(weight * bits_matrix) - 1

        # 考慮旋轉
        if rotate:
            # 找 1 的個數
            _, counts = np.unique(bits_matrix, return_counts=True)
            count = counts[-1] - 1  # threshold 判斷有用等於所以要減去中心的

            # 假如把符合條件的都設定為最大值
            if lbp_value in self.rotate_labels[count]:
                lbp_value = max(self.rotate_labels[count])

        self.lbp_val[lbp_value] += 1  # 統計整張圖的 LBP
        return lbp_value

    def lbp(self, cells=10, r=1, rotate=False):
        """step2 ~ step4, 單張圖片 step5, step6
        cells: 設定 cells 大小
        r: 設定 blocks 大小 (size = 2*r+1)
        """
        self.size = 2 * r + 1
        self.cells = cells
        # img.shape = (1000, 1000), cell_top_all.shape = (100, 100, 2, 5)
        self.cell_top_all = np.zeros((self.y // cells, self.x // cells, 2, 5), dtype=np.int16)

        con_img = np.zeros((self.y+2, self.x+2), dtype=np.uint16)
        con_img[1:self.y+1, 1:self.x+1] = self.img

        for j in range(0, self.y):
            for i in range(0, self.x):
                target_bits = con_img[j:j+self.size, i:i+self.size]
                # block center: r, r
                target_bits = np.where(target_bits >= target_bits[r, r], 1, 0)
                self.lbp_img[j, i] = self.bit_to_int(target_bits, rotate)  # 將每個 pixel 的 LBP 值存入

        for j in range(0, self.y // self.cells):
            for i in range(0, self.x // self.cells):
                target_cell = self.lbp_img[j:j+self.cells, i:i+self.cells]
                grayscale, counts = np.unique(target_cell, return_counts=True)  # 去重且回傳每個元素出現的個數

                # 取前 5 個出現次數最多的 LBP 值
                index = np.argsort(counts)[-5:]  # 傳回前五個最大值索引
                top_five = grayscale[index]  # 取 grayscale 前五個
                top_count = counts[index]

                if len(top_five) < 5:
                    # 元素不足 5 個則向後面填充 0
                    top_five = np.pad(top_five, (0, 5-len(top_five)), 'constant')
                    top_count = np.pad(top_count, (0, 5-len(top_count)), 'constant')
                self.cell_top_all[j, i] = np.array([top_five, top_count])

        return self.lbp_img

    def local_histogram(self, y_axis=None, x_axis=None):
        """繪製特定區域的 LBP 統計圖
        y_axis: 第 y 個的 cells 位置
        x_axis: 第 x 個的 cells 位置
        """
        if y_axis and x_axis:
            target = self.lbp_img[y_axis*self.cells:(y_axis+1)*self.cells, x_axis*self.cells:(x_axis+1)*self.cells]
            for y in range(0, self.cells):
                for x in range(0, self.cells):
                    self.cell_val[target[y][x]] += 1

            fig = plt.gcf()
            fig.set_size_inches(10, 6)
            x_scale = np.arange(0, 256)
            plt.title("Local LBP Chart")
            plt.xlabel("GrayScale")
            plt.ylabel("Times")
            plt.bar(x_scale, self.cell_val, width=2)
            plt.ylim(0, np.max(self.cell_val))
            plt.xticks(np.linspace(0, 255, 18))
            plt.show()

    def all_histogram(self):
        """繪製整張圖像的 LBP 統計圖"""
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        x_scale = np.arange(0, 256)
        plt.bar(x_scale, self.lbp_val, width=2)
        plt.title("LBP Chart")
        plt.xlabel("Gray Scale")
        plt.ylabel("Times")
        plt.ylim(0, np.max(self.lbp_val))
        plt.xticks(np.linspace(0, 255, 18))
        plt.show()

    def spend_time(self):
        """計算程式執行時間"""
        end = time.time()
        return end - self.start


path = "./road/road.jpg"
road = cv2.imread(path)

# use my lbp
# step1:
road_gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)

# step2 to 7:
handle = MyLBP(road_gray)
road_my_lbp = handle.lbp(rotate=False)
# handle.local_histogram(86, 5)
# handle.all_histogram()
result = handle.compare(86, 6)
# print("spend time: %.3f s" % handle.spend_time())
# spend time:  17.367 s

output_img(result, text="./road/road_my_lbp_compare_result")
# cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
