import math
import FFT
import matplotlib.pyplot as plt
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': 'larger'}
(plt.rc("font", family='MicroSoft YaHei', weight="bold"))


# 生成高斯低通滤波器
def gaussian_low_pass_filter(shape, cutoff_ratio=0.95):
    M, N = shape
    center_x, center_y = M // 2, N // 2
    sigma = min(M, N) / 4  # 高斯滤波器的标准差
    filter_matrix = [[0] * N for _ in range(M)]
    total_energy, current_energy = 0, 0

    for u in range(M):
        for v in range(N):
            distance = ((u - center_x) ** 2 + (v - center_y) ** 2) / (2 * sigma ** 2)
            filter_matrix[u][v] = math.exp(-distance)
            total_energy += filter_matrix[u][v]

    # 归一化滤波器以确保总能量比例达到cutoff_ratio
    sorted_values = sorted(filter_matrix[u][v] for u in range(M) for v in range(N))
    threshold_energy = cutoff_ratio * total_energy

    for value in sorted_values:
        current_energy += value
        if current_energy >= threshold_energy:
            threshold = value
            break

    for u in range(M):
        for v in range(N):
            if filter_matrix[u][v] < threshold:
                filter_matrix[u][v] = 0

    return filter_matrix

# 在频域中应用二维滤波器
def apply_frequency_filter(image, filter_matrix):
    fft_image = FFT.fft_2d(image)
    M, N = len(image), len(image[0])
    filtered_fft = [[fft_image[u][v] * filter_matrix[u][v] for v in range(N)] for u in range(M)]
    return FFT.ifft_2d(filtered_fft)

# 将二维矩阵乘以 (-22336216_陶宇卓_数图第三次编程作业)^(x+y)
def multiply_by_neg1_pow_xy(matrix):
    M, N = len(matrix), len(matrix[0])
    return [[matrix[x][y] * ((-1) ** (x + y)) for y in range(N)] for x in range(M)]

# 主函数
def main():
    # 加载一个 512x512 的灰度图像（可以从文件加载或生成合成数据）
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # 加载输入图像
    image = Image.open("第四章编程作业图.jpg").convert("L")  # 转换为灰度图像
    image = image.resize((512, 512))
    image_array = np.array(image, dtype=float)

    # (a) 原始图像
    original_image = image_array

    # (b) 零填充后的图像
    padded_image = np.pad(original_image, ((0, 512), (0, 512)), mode="constant")

    # (c) 乘以 (-22336216_陶宇卓_数图第三次编程作业)^(x+y) 后的图像
    neg1_image = multiply_by_neg1_pow_xy(padded_image)

    # (d) 图像的傅里叶变换（频谱）
    fft_image = FFT.fft_2d(neg1_image)
    magnitude_spectrum = np.log(np.abs(fft_image) + 1)

    # (e) 高斯低通滤波器
    gaussian_filter = gaussian_low_pass_filter((1024, 1024), cutoff_ratio=0.95)

    # (f) 滤波器与频谱的乘积
    filtered_fft = [[fft_image[u][v] * gaussian_filter[u][v] for v in range(1024)] for u in range(1024)]
    filtered_magnitude_spectrum = np.log(np.abs(filtered_fft) + 1)

    # (g) 逆傅里叶变换后的图像并乘以 (-22336216_陶宇卓_数图第三次编程作业)^(x+y)
    filtered_image = FFT.ifft_2d(filtered_fft)
    g_p = multiply_by_neg1_pow_xy(np.real(filtered_image))

    # (h) 提取最终的 MxN 结果
    # 将 g_p 转换为 NumPy 数组
    g_p_array = np.array(g_p)
    final_image = g_p_array[:512, :512]
    # 可视化结果
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 4, 1)
    plt.title("(a) 原始图像")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.title("(b) 零填充后的图像")
    plt.imshow(padded_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.title("(c) 乘以 (-22336216_陶宇卓_数图第三次编程作业)^(x+y)")
    plt.imshow(neg1_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.title("(d) 频谱图")
    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.title("(e) 高斯滤波器")
    plt.imshow(gaussian_filter, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.title("(f) 滤波后的频谱")
    plt.imshow(filtered_magnitude_spectrum, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.title("(g) g_p 图像")
    plt.imshow(g_p, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.title("(h) 最终结果")
    plt.imshow(final_image, cmap="gray")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()

