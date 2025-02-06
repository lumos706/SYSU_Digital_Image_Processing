import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': 'larger'}
(plt.rc("font", family='MicroSoft YaHei', weight="bold"))


def compute_histogram(image):
    # 计算灰度直方图
    hist = np.zeros(256)
    for pixel in image.ravel():
        hist[pixel] += 1
    hist = hist / hist.sum()
    return hist


def histogram_equalization(image):
    # 直方图均衡化
    hist = compute_histogram(image)
    cdf = hist.cumsum()  # 累积分布函数
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype('uint8')

    equalized_image = cdf_normalized[image]
    return equalized_image, cdf_normalized


# 加载图像a
image_a = Image.open("a.jpg").convert("L")
image_a = np.array(image_a)

# 直方图均衡化
equalized_image, equalized_cdf = histogram_equalization(image_a)

# 绘制结果
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(image_a, cmap='gray')
axs[0, 0].set_title('原始图像')

original_hist = compute_histogram(image_a)
axs[0, 1].bar(range(256), original_hist)
axs[0, 1].set_title('图像直方图')
axs[0, 1].set_ylim(0, 0.6)
axs[0, 1].set_xticks([0, 63, 127, 191, 255])

axs[0, 2].plot(equalized_cdf, color='red')
axs[0, 2].set_title('直方图均衡变换')

equalized_hist = compute_histogram(equalized_image)
axs[1, 0].imshow(equalized_image, cmap='gray')
axs[1, 0].set_title('直方图均衡化后的图像')

axs[1, 1].bar(range(256), equalized_hist)
axs[1, 1].set_title('均衡化后的图像直方图')
axs[1, 1].set_xticks([0, 63, 127, 191, 255])
axs[1, 1].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

axs[1, 2].axis('off')

plt.show()
