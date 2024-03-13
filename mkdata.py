import cv2
import numpy as np


bg_path = "examples/ride/scene.jpg"
bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)

mask = np.zeros_like(bg)[:, :, -1]
H, W = mask.shape
print(H, W)
ratio_x, ratio_y = 2 / 3, 1 / 10
ratio_h, ratio_w = 1 / 6, 1 / 4

y, x = int(H * ratio_x), int(W * ratio_y)
h, w = int(H * ratio_h), int(W * ratio_w)

mask[y: y+h, x: x+w] = 255
cv2.imwrite("examples/ride/tar01.png", mask)