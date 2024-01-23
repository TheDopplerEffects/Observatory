import cv2 as cv
import math as m
import numpy as np
import os
import timeit

path = "/home/tim/Documents/Datasets/VernierImages/DEC/DecV1/flipped/images/"
wrpath = "/home/tim/Documents/Datasets/VernierImages/DEC/DecV1/corrected/images/"

image_width_mm = 55.74
pxl_per_mm = 1440/55.74
IMG_DIMS = (1440,1080)

radius_inch = (17.5)/2
radius_mm = radius_inch * 25.4
radius_pxl = round(radius_mm*pxl_per_mm)

center = (865,5984)

r1x = center[0]
r1y = center[1]
r1 = (r1x**2 + r1y**2)**0.5
theta1 = m.atan(r1x/r1y)

r2x = abs(IMG_DIMS[0] - center[0])
r2y = center[1]
r2 = (r2x**2 + r2y**2)**0.5
theta2 = m.atan(r2x/r2y)

h1 = (r1 - r1*m.cos(theta1))/m.cos(theta1)
dx1 = m.ceil((h1 * m.sin(theta1)))
dy1 = m.ceil((h1 * m.cos(theta1)))

h2 = (r2 - r2*m.cos(theta2))/m.cos(theta2)
dx2 = m.ceil((h2 * m.sin(theta2)))
dy2 = m.ceil((h2 * m.cos(theta2)))

dy_max = m.ceil(max(dy1,dy2))

fds = os.listdir(path)
start = timeit.default_timer()
for fd in fds:
    image_gray = cv.imread(path + fd,cv.IMREAD_GRAYSCALE)
    new_image = np.zeros((IMG_DIMS[1] + dy_max,IMG_DIMS[0] + (dx1 + dx2)))
    for i in range(IMG_DIMS[0]):
        for j in range(IMG_DIMS[1]):
            rx = abs(center[0] - i)
            ry = abs(center[1] - j)
            r = ((rx)**2 + (ry)**2)**0.5
            theta = m.atan(rx/ry)
            cos = m.cos(theta)
            sin = m.sin(theta)
            h = (r - r*cos)/cos
            
            if i < center[0]:
                dx = m.ceil(h*sin - dx1)
                dy = m.ceil(h*cos - dy_max)
                new_image[j - dy,i + dx] = image_gray[j,i]
            elif i >= center[0]:
                dx = -m.ceil(h*sin - dx2 + dx1 + 2)
                dy = m.ceil(h*cos - dy_max)
                new_image[j - dy, i + dx] = image_gray[j,i]
    new_image = new_image[dy_max + 1 : new_image.shape[0] - dy_max,dx1:new_image.shape[1] - 24]
    
    # cv.imwrite(wrpath + fd,new_image)
    # print(image_gray.shape,new_image.shape)
    # print(timeit.default_timer() - start)
    cv.imwrite('new_image.png',new_image)
    break
