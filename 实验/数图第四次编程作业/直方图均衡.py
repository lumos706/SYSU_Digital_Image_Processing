import cv2
import numpy as np
import matplotlib.pyplot as plt
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': 'larger'}
(plt.rc("font", family='MicroSoft YaHei', weight="bold"))
def equalize_hist_in_hsi(image):
    hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, i = cv2.split(hsi)
    i_equalized = cv2.equalizeHist(i)
    hsi_equalized = cv2.merge([h, s, i_equalized])
    result = cv2.cvtColor(hsi_equalized, cv2.COLOR_HSV2BGR)
    return result

def equalize_hist_in_rgb(image):
    b, g, r = cv2.split(image)
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    result = cv2.merge([b_eq, g_eq, r_eq])
    return result

# 读取图像
image_path = 'image.jpg'
image = cv2.imread(image_path)

# 检查图像是否加载成功
if image is None:
    raise FileNotFoundError(f"无法加载图像: {image_path}")

# 方法#1：HSI空间I分量直方图均衡
hsi_equalized = equalize_hist_in_hsi(image)

# 方法#2：RGB空间各通道直方图均衡
rgb_equalized = equalize_hist_in_rgb(image)

# 显示结果
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('原始图像')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(hsi_equalized, cv2.COLOR_BGR2RGB))
plt.title('仅在I通道上进行直方图均衡')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(rgb_equalized, cv2.COLOR_BGR2RGB))
plt.title('在RGB通道上进行直方图均衡')

plt.tight_layout()
plt.show()