import cv2 as cv
import torch as t
from post_processing.post_processing import *
from post_processing.utils import *

from config.config import *
impath = '/home/tim/Documents/Datasets/VernierImages/DEC/DecV1/corrected/images/400.png'
lblpath = '/home/tim/Documents/Datasets/VernierImages/DEC/DecV1/corrected/labels/400.txt'

image = cv.imread(impath)
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
fine_boxes,coarse_boxes,numeral_positions = post_process_boxes(dets,image.shape[0]//2)

coarse_segment,fine_segment,boundary_line = split_image(cv.cvtColor(image,cv.COLOR_BGR2GRAY),numeral_positions,True,'split_image.png')

coarse_line_positions,coarse_line_labels = get_coarse_line_positions(coarse_boxes)

coarse_lines = get_coarse_lines(coarse_segment,coarse_line_positions,coarse_line_labels,numeral_positions[1],False,'0.png')
coarse_points_tmp = get_coarse_points(coarse_lines,boundary_line)
coarse_points = label_coarse_points(coarse_points_tmp,True,True)

fine_line_positions,fine_line_labels = get_fine_line_positions(fine_boxes)
fine_lines = get_fine_lines(fine_segment,fine_line_positions,fine_line_labels,numeral_positions[0],False,'0.png')
fine_points_tmp = get_fine_points(fine_lines,boundary_line)
fine_points = label_fine_points(fine_points_tmp)

degrees,coarse_mins = measure_degree_mins(coarse_points,fine_points)

fine_mins,secs = measure_seconds(coarse_points,fine_points)

pt1 = (0,int(boundary_line[1]))
pt2 = (image.shape[1],int(image.shape[1] * boundary_line[0] + boundary_line[1]))
cv.line(image,pt1,pt2,(0,255,0),1)


# for point in coarse_points:
#     x = round(point[0].item())
#     y = round(point[1].item())
#     cv.circle(image,(x,y),1,(255,0,0),1)

# for point in fine_points:
#     x = round(point[0].item())
#     y = round(point[1].item())
#     cv.circle(image,(x,y),1,(0,0,255),1)

# for line in fine_lines:
#     m = line[0]
#     b = line[1]
#     pt1 = (0,round(b.item()))
#     pt2 = (image.shape[1],round(m.item()*image.shape[1] + b.item()))
#     cv.line(image,pt1,pt2,(0,0,255),1)

# for line in coarse_lines:
#     m = line[0]
#     b = line[1]
#     pt1 = (0,round(b.item()))
#     pt2 = (image.shape[1],round(m.item()*image.shape[1] + b.item()))
#     cv.line(image,pt1,pt2,(0,255,0),1)
# for i in range(len(coarse_line_positions)):
#     x = int(coarse_line_positions[i].item())
#     y = coarse_segment.shape[0]
#     cv.circle(coarse_segment,(x,y),3,(0,255,0),1)
# cv.imwrite('coarse_segment.png',coarse_segment)

image = draw_coarse_points(image,coarse_points)
image = draw_fine_points(image,fine_points)
cv.imwrite('image.png',image)
# fine_line_positions,fine_line_labels = get_fine_line_positions(fine_boxes,image.shape)
