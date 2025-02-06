import cv2
import numpy as np
import matplotlib.pyplot as plt
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': 'larger'}
(plt.rc("font", family='MicroSoft YaHei', weight="bold"))


def manual_otsu_thresholding(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 计算直方图
    hist, bins = np.histogram(image.ravel(), 256, [0, 256])

    # 总像素数
    total_pixels = image.size

    # 初始化变量
    current_max, threshold = 0, 0
    sum_total, sum_foreground = 0, 0
    weight_background, weight_foreground = 0, 0

    for i in range(256):
        sum_total += i * hist[i]

    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_foreground += i * hist[i]

        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground

        # 计算类间方差
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        # 找到最大类间方差对应的阈值
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i

    # 应用阈值
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return binary_image

# 使用手动实现的Otsu方法对image.png进行分割
binary_image = manual_otsu_thresholding('image.png')

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("原始图像")
plt.imshow(cv2.imread('image.png', cv2.IMREAD_GRAYSCALE), cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Otsu阈值化结果")
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()