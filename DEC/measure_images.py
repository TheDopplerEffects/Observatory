import cv2 as cv
import torch as t
from post_processing.post_processing import *
from post_processing.utils import *

from config.config import *
impath = '/home/tim/Documents/Datasets/VernierImages/DEC/DecV1/corrected/images/400.png'
lblpath = '/home/tim/Documents/Datasets/VernierImages/DEC/DecV1/corrected/labels/400.txt'

image = cv.imread(impath,cv.IMREAD_GRAYSCALE)
class_names = ['20', '40', '60', 'A', 'Arrow', 'B', 'I', 'II', 'III', 'IIII', 'IX', 'V',
               'VI', 'VII', 'VIII', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XIX', 'XV', 'XVI', 
               'XVII', 'XVIII', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', '28', '46', '64', 
               '82', '100', '0', '5', '10', '15', '25', '30', '35', '45', '50', 
               '55', '65', '70', '75', '80', '85', '90']

with open(lblpath,'r') as f:
    lines = f.readlines()
    dets = []
    for line in lines:
        line = line.split()
        line = [float(x) for x in line]
        line[1] = line[1]*image.shape[1]
        line[2] = line[2]*image.shape[0]
        line[3] = line[3]*image.shape[1]
        line[4] = line[4]*image.shape[0]
        line = [int(x) for x in line]

        dets.append([line[1] - line[3]//2,line[2] - line[4]//2,line[1] + line[3]//2,line[2] + line[4]//2,1,line[0]])

dets = t.tensor(dets)

measure,image = infer_measure(image,dets)
print(measure)