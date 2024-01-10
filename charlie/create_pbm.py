import numpy as np
import cv2

grid = np.zeros([40,40])
grid[:3, 0] = 1
grid[:, 39] = 20
grid[:3, :] = 99
grid[39, :] = 255 

cv2.imwrite("resource/map.png", grid)
