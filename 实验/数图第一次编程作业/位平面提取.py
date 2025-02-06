import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': 'larger'}
(plt.rc("font", family='MicroSoft YaHei', weight="bold"))


def bit_plane_extraction(image, bit_plane):
    # 提取位平面
    bit_mask = 1 << bit_plane
    return (image & bit_mask) >> bit_plane


def modify_bit_plane(image, bit_plane, new_bit_plane_values):
    bit_mask = 1 << bit_plane
    modified_image = image & ~bit_mask
    modified_image |= (new_bit_plane_values & 1) << bit_plane
    return modified_image

# 加载图像
image_c = Image.open("lena-512-gray.png").convert("L")
# image_c = Image.open("第四章编程作业图.jpg").convert("L")
image_c = np.array(image_c)
random_bit_plane_values = np.random.randint(0, 2, size=image_c.shape)
# random_bit_plane_values = np.zeros(image_c.shape, dtype=int)
print(random_bit_plane_values)
# 提取和展示8个位平面
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    bit_plane_image = bit_plane_extraction(image_c, i)
    axs[i//4, i%4].imshow(bit_plane_image, cmap='gray')
    axs[i//4, i%4].set_title(f'Bit Plane {i+1}')
plt.show()

# 修改最低位平面并展示
modified_image_array = modify_bit_plane(image_c, 0, random_bit_plane_values)
modified_image = Image.fromarray(modified_image_array)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image_c, cmap='gray')
axs[0].set_title('Original Image')

axs[1].imshow(modified_image, cmap='gray')
axs[1].set_title('Modified Image (Lowest Bit Plane Changed)')

plt.show()
