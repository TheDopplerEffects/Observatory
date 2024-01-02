import os
import cv2 as cv
import numpy as np


path = '/home/tim/Documents/Datasets/VernierImages/DEC/DecV2/images/'

fds = os.listdir(path)

for fd in fds:
    image = cv.imread(path + fd, cv.IMREAD_GRAYSCALE)
    rows = image.shape[0]
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, rows / 8, 200,50,minRadius=1,maxRadius=500)
    # print(circles)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(image, center, radius, (255, 0, 255), 3)
    
    cv.imwrite('image.png',image)   
    break