from scipy import signal
import numpy as np

__all__ = ['conv_to_gray', 'lbp', 'sobel']


def conv_to_gray(img):
    """
    conv_to_gray(img): 彩色轉成灰階
    img: 彩色圖片
    pixel_grayscale = (b + g + r) // 3
    return gray 灰階圖像
    """

    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = (b + g + r) // 3
    return gray


def bit_to_int(matrix, r):
    """(private)
    bit_to_int(matrix): 將輸入二進制矩陣轉成十進制的數值
    matrix: 二進制矩陣
    method: 計算方向為(高到低位元), 右, 右上, 上, 左上, 左, 左下, 下, 右下
    return lbp_value 十進制的 LBP 值
    """
    pos = np.array([
        [4, 5, 6],
        [3, 0, 7],
        [2, 1, 0]
    ])
    weight = np.power(2, pos)

    region = np.array([
        [matrix[0, 0], matrix[0, r], matrix[0, 2*r]],
        [matrix[r, 0], matrix[r, r], matrix[r, 2*r]],
        [matrix[2*r, 0], matrix[2*r, r], matrix[2*r, 2*r]]
    ])

    lbp_value = sum(weight * region)
    return lbp_value


def lbp(img, rad=1):
    """
    lbp(img[, rad=1]): 將灰階圖片轉成 LBP 值的圖片
    img: 輸入灰階圖像
    rad: 選擇 blocks 的半徑
    return lbp_img 存放 LBP 值的圖片
    """
    y, x = img.shape
    blocks = rad * 2 + 1

    reg_img = np.zeros((y+2*rad, x+2*rad), dtype=np.uint16)
    reg_img[rad: y+rad, rad:x+rad] = img

    lbp_img = np.zeros(img.shape, dtype=np.uint8)

    for j in range(y):
        for i in range(x):
            target = reg_img[j:j+blocks, i:i+blocks]
            bits = np.where(target > target[rad, rad], 1, 0)
            lbp_img[j, i] = bit_to_int(bits, rad)
    del reg_img
    return lbp_img


def sobel(img):
    """
    sobel(img): 將圖片做 Sobel 運算
    img: 輸入灰階圖像
    method:
    kernel of x ->
    [-1, 0, 1]
    [-2, 0, 2]
    [-1, 0, 1]
    kernel of y ->
    [-1, -2, -1]
    [0, 0, 0]
    [1, 2, 1]
    G = (gx * gx + gy * gy) ^ 0.5
    return G Sobel 運算後的結果(灰階圖片)
    """
    kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    ky = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    y, x = img.shape
    reg = np.zeros((y+2, x+2), np.int32)
    reg[1:y + 1, 1:x + 1] = img

    gx = signal.convolve(reg, kx, method='valid')
    gy = signal.convolve(reg, ky, method='valid')

    G = np.sqrt(gx * gx + gy * gy).astype(np.uint8)
    return G


def compare(img, sample, length, width):
    pass


def search_roadLBP(img):
    """
    search_roadLBP(img): 尋找馬路材質
    img:  輸入 lbp 轉換過的圖片
    method: 假設馬路在圖片最底端的位置, 找出一個 block 的 LBP 值, 迭代圖片去搜尋出現次數最多的
    return: block_road, 為最後比較結果時當成樣本
    """
    pass


def auto_marker(img):
    pass
