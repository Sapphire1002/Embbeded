from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import cv2
import numpy as np


path = "./road/road.jpg"
road = cv2.imread(path)
road_gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)

# use scikit-image module lbp
# use matplotlib imshow lbp image
radius = 1
n_points = 8 * radius

# lbp method returns the dtype and value of the image(current only the image)
# default: dtype float64, value 0 to 255
# ror: dtype float64, value 0 to 255
# nri_uniform: dtype float64, value 0 to 58
# uniform: dtype float64, value 0 to 9
# var: dtype float64, value has han

lbp = local_binary_pattern(road_gray, n_points, radius, method='var')
print(lbp[30:40, 790:800])
print(np.max(lbp), np.min(lbp))
plt.imshow(lbp, cmap='gray')
plt.show()
