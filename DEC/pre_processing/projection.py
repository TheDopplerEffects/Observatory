import cv2 as cv
import math as m
import numpy as np
import os
import timeit
from tqdm import tqdm

path = ""
wrpath = ""

image_width_mm = 55.74
pxl_per_mm = 1440 / 55.74
IMG_DIMS = (1440, 1080)

radius_inch = (17.5) / 2
radius_mm = radius_inch * 25.4
radius_pxl = round(radius_mm * pxl_per_mm)

center = (865, 5984)

r1x = center[0]
r1y = center[1]
r1 = (r1x**2 + r1y**2) ** 0.5
theta1 = m.atan(r1x / r1y)

r2x = abs(IMG_DIMS[0] - center[0])
r2y = center[1]
r2 = (r2x**2 + r2y**2) ** 0.5
theta2 = m.atan(r2x / r2y)

h1 = (r1 - r1 * m.cos(theta1)) / m.cos(theta1)
dx1 = m.ceil((h1 * m.sin(theta1)))
dy1 = m.ceil((h1 * m.cos(theta1)))

h2 = (r2 - r2 * m.cos(theta2)) / m.cos(theta2)

dx2 = m.ceil((h2 * m.sin(theta2)))
dy2 = m.ceil((h2 * m.cos(theta2)))

dy_max = m.ceil(max(dy1, dy2))

x_ind = np.expand_dims(np.array([x for x in range(IMG_DIMS[0])]), axis=1)
y_ind = np.expand_dims(np.array([x for x in range(IMG_DIMS[1])]), axis=1)

ry = np.transpose(np.abs(center[1] - y_ind))
rx = np.abs(center[0] - x_ind)

r = np.sqrt(np.power(ry, 2) + np.power(rx, 2))
theta = np.divide(rx, ry)
cos = np.cos(theta)
sin = np.sin(theta)
h = np.divide(np.subtract(r, np.multiply(r, cos)), cos)

dx = np.multiply(h, sin)
dy = np.multiply(h, cos)

dx[center[0] :, :] = np.negative(dx[center[0] :, :])

dx = np.transpose(np.ceil(dx))
dy = np.transpose(np.ceil(dy))

dy = y_ind - dy
dx = np.transpose(x_ind) + dx

dy = dy.astype(int)
dx = dx.astype(int)

fds = os.listdir(path)

blank_image = np.zeros((IMG_DIMS[1] + dy_max, IMG_DIMS[0] + (dx1 + dx2)))
for fd in tqdm(fds):
    image = cv.imread(path + fd, cv.IMREAD_GRAYSCALE)
    new_image = np.copy(blank_image)
    new_image[dy, dx] = image
    new_image = new_image[
        : new_image.shape[0] - 2 * dy_max - 1, dx1 + 12 : new_image.shape[1] - 12
    ]
    cv.imwrite(wrpath + fd, new_image)
