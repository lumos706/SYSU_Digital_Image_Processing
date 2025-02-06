import cv2
import numpy as np
import matplotlib.pyplot as plt
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': 'larger'}
(plt.rc("font", family='MicroSoft YaHei', weight="bold"))
def add_salt_and_pepper_noise(image, pa, pb):
    noisy_image = image.copy()
    total_pixels = image.size
    num_salt = int(pa * total_pixels)
    num_pepper = int(pb * total_pixels)

    # 添加盐噪声
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    # 添加椒噪声
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

# 加载图像 (a)
image = cv2.imread('a.jpg', cv2.IMREAD_GRAYSCALE)
image_b = cv2.imread('5_10(b).png', cv2.IMREAD_GRAYSCALE)
# 添加椒盐噪声
pa = pb = 0.2
noisy_image = add_salt_and_pepper_noise(image, pa, pb)

# 应用中值滤波
median_filtered_1 = cv2.medianBlur(noisy_image, 3)
median_filtered_2 = cv2.medianBlur(median_filtered_1, 3)
median_filtered_3 = cv2.medianBlur(median_filtered_2, 3)
median_filtered_4 = cv2.medianBlur(median_filtered_3, 3)
median_filtered_5 = cv2.medianBlur(median_filtered_4, 3)
# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(2, 4, 1)
plt.title("原始图像")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title("加入椒盐噪声")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title("中值滤波 (1 次)")
plt.imshow(median_filtered_1, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title("中值滤波 (2 次)")
plt.imshow(median_filtered_1, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.title("中值滤波 (3 次)")
plt.imshow(median_filtered_2, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.title("中值滤波 (4 次)")
plt.imshow(median_filtered_3, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.title("中值滤波 (5 次)")
plt.imshow(median_filtered_4, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.title("5.10(b)")
plt.imshow(image_b, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
