import numpy as np
import cv2
import matplotlib.pyplot as plt
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': 'larger'}
(plt.rc("font", family='MicroSoft YaHei', weight="bold"))
# 生成滤波器 H(u,v) (式 5.77)
def generate_filter(shape, T):
    M, N = shape
    u = np.arange(-M//2, M//2)
    v = np.arange(-N//2, N//2)
    U, V = np.meshgrid(u, v)
    a = b = 0.1
    uv_product = U * a + V * b
    H = (T / (np.pi * uv_product)) * np.sin(np.pi * uv_product) * np.exp(-1j * np.pi * uv_product)
    H[uv_product == 0] = T  # 对于 uv_product = 0 的点，H 为 T
    return np.fft.fftshift(H)

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, var=10):
    noise = np.random.normal(mean, np.sqrt(var), image.shape)
    noisy_image = np.clip(image + noise, 0, 255)  # 保证值在[0,255]
    return noisy_image.astype(np.uint8)

# 维纳滤波器恢复 (式 5.85)
def wiener_filter(blurred_img, H, K=0.01):
    H_conj = np.conj(H)
    H_abs2 = np.abs(H)**2
    G = np.fft.fft2(blurred_img)
    F_hat = (H_conj / (H_abs2 + K)) * G
    restored_img = np.fft.ifft2(F_hat)
    return np.abs(restored_img)

# 加载图像 (b)
image_b = cv2.imread('b.jpg', cv2.IMREAD_GRAYSCALE)

# 滤波器参数
T = 1
H = generate_filter(image_b.shape, T)

# 对图像 (b) 模糊处理
F = np.fft.fft2(image_b)
blurred_freq = F * H
blurred_image = np.abs(np.fft.ifft2(blurred_freq))

# 添加高斯噪声
noisy_image = add_gaussian_noise(blurred_image)

# 应用维纳滤波器
restored_image = wiener_filter(noisy_image, H, K=0.001)

# 显示结果
plt.figure(figsize=(15, 8))
plt.subplot(1, 4, 1)
plt.title("原始图像")
plt.imshow(image_b, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("模糊图像")
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("加入高斯噪声")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("维纳滤波器恢复")
plt.imshow(restored_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
plt.subplot(1, 3, 1)
plt.title("K=0.001")
plt.imshow(restored_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("K=0.003")
restored_image_1 = wiener_filter(noisy_image, H, K=0.003)
plt.imshow(restored_image_1, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
restored_image_2 = wiener_filter(noisy_image, H, K=0.005)
plt.title("K=0.005")
plt.imshow(restored_image_2, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
