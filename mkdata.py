import cv2
import numpy as np


bg_path = "examples/ride/scene.jpg"
bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)

mask = np.zeros_like(bg)[:, :, -1]
H, W = mask.shape
print(H, W)

def draw_bbox(bg_path):
    def draw_one_bbox(ratio_x, ratio_w, ratio_y, ratio_h, i):
        y, x = int(H * ratio_x), int(W * ratio_y)
        h, w = int(H * ratio_h), int(W * ratio_w)
        mask = np.zeros_like(bg)[:, :, -1]
        cv2.imwrite("examples/ride/tar01.png", mask)
    
    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    mask = np.zeros_like(bg)[:, :, -1]
    H, W = mask.shape
    