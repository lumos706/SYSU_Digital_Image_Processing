import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': 'larger'}
(plt.rc("font", family='MicroSoft YaHei', weight="bold"))
def load_image(filepath):
    """加载图像并确保是二值化灰度图"""
    image = imread(filepath)
    if image.ndim == 3:  # RGB 图像
        image = rgb2gray(image)
    return image > 0.5

def binary_dilation_3x3(image):
    """使用3x3结构元素进行二进制膨胀"""
    structuring_element = np.ones((3, 3), dtype=bool)
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    result = np.zeros_like(image, dtype=bool)
    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            if np.any(padded_image[i-1:i+2, j-1:j+2] & structuring_element):
                result[i-1, j-1] = 1
    return result

def binary_erosion_3x3(image):
    """使用3x3结构元素进行二进制腐蚀"""
    structuring_element = np.ones((3, 3), dtype=bool)
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    result = np.zeros_like(image, dtype=bool)
    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            if np.all(padded_image[i-1:i+2, j-1:j+2] & structuring_element):
                result[i-1, j-1] = 1
    return result

def morphological_boundary_extraction(image):
    """形态边界提取"""
    eroded_image = binary_erosion_3x3(image)
    boundary = image & ~eroded_image
    return boundary

# 加载图像
image = load_image('image.jpg')

# 提取字符边界
boundary = morphological_boundary_extraction(image)

# 保存结果
imsave('boundary_result.png', img_as_ubyte(boundary))

# 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title("原始图像")
axes[0].axis('off')

axes[1].imshow(boundary, cmap='gray')
axes[1].set_title("形态边界提取结果")
axes[1].axis('off')

plt.tight_layout()
plt.show()