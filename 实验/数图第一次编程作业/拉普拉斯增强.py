import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': 'larger'}
(plt.rc("font", family='MicroSoft YaHei', weight="bold"))


def laplacian_enhancement(image, kernel, c=-1):
    # 拉普拉斯锐化
    padded_image = np.pad(image, 1, mode='edge')
    output_image = np.zeros_like(image)

    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            region = padded_image[i - 1:i + 2, j - 1:j + 2]
            laplace_value = np.sum(region * kernel)
            output_image[i - 1, j - 1] = np.clip(image[i - 1, j - 1] + c * laplace_value, 0, 255)

    return output_image


# 拉普拉斯核
laplacian_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# 加载图像b
image_b = Image.open("b.jpg").convert("L")
image_b = np.array(image_b)

# 应用拉普拉斯增强
enhanced_image = laplacian_enhancement(image_b, laplacian_kernel)

# 绘制结果
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image_b, cmap='gray')
axs[0].set_title('原始图像')

axs[1].imshow(enhanced_image, cmap='gray')
axs[1].set_title('拉普拉斯增强后的图像')

plt.show()
