# create_sample_images.py
import numpy as np
from PIL import Image
import os

os.makedirs('datasets/sample_images', exist_ok=True)

# Tạo vài ảnh mẫu
# 1. Ảnh đỏ
red_img = np.zeros((200, 200, 3), dtype=np.uint8)
red_img[:, :, 0] = 255  # Red channel
Image.fromarray(red_img).save('datasets/sample_images/red_image.png')

# 2. Ảnh xanh lá
green_img = np.zeros((200, 200, 3), dtype=np.uint8)
green_img[:, :, 1] = 255  # Green channel
Image.fromarray(green_img).save('datasets/sample_images/green_image.png')

# 3. Ảnh có pattern
pattern_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
# Tạo stripes
pattern_img[::20, :, :] = [255, 255, 255]
Image.fromarray(pattern_img).save('datasets/sample_images/pattern_image.png')

# 4. Gradient
gradient = np.zeros((200, 200, 3), dtype=np.uint8)
for i in range(200):
    gradient[:, i] = [i*255//200, i*255//200, i*255//200]
Image.fromarray(gradient).save('datasets/sample_images/gradient_image.png')

print("✅ Đã tạo 4 ảnh mẫu!")
