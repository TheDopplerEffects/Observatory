import cv2
import numpy as np
import matplotlib.pyplot as plt
import pprint
import math as m
from config.config import *

_COLORS = np.array([[1.        , 0.523602  , 0.        ],
       [0.5913978 , 1.        , 0.37634408],
       [0.8030303 , 0.        , 0.        ],
       [0.        , 0.3       , 1.        ],
       [0.8064516 , 1.        , 0.16129032],
       [0.        , 0.        , 0.5       ],
       [1.        , 0.7705156 , 0.        ],
       [0.        , 0.        , 0.8030303 ],
       [0.        , 0.56666666, 1.        ],
       [0.        , 0.15882353, 1.        ],
       [0.        , 0.8333333 , 1.        ],
       [0.2624921 , 1.        , 0.70524985],
       [0.16129032, 1.        , 0.8064516 ],
       [0.9203036 , 1.        , 0.04743833],
       [0.        , 0.        , 0.6426025 ],
       [1.        , 0.16049382, 0.        ],
       [1.        , 0.90123457, 0.        ],
       [1.        , 0.27668846, 0.        ],
       [0.49019608, 1.        , 0.47754586],
       [0.37634408, 1.        , 0.5913978 ],
       [0.        , 0.69215685, 1.        ],
       [0.70524985, 1.        , 0.2624921 ],
       [0.        , 0.4254902 , 1.        ],
       [0.        , 0.03333334, 1.        ],
       [0.        , 0.        , 0.9456328 ],
       [0.6426025 , 0.        , 0.        ],
       [1.        , 0.654321  , 0.        ],
       [1.        , 0.4074074 , 0.        ],
       [0.04743833, 0.9588235 , 0.9203036 ],
        [0.5913978 , 1.        , 0.37634408],
       [0.8030303 , 0.        , 0.        ],
       [0.        , 0.3       , 1.        ],
       [0.8064516 , 1.        , 0.16129032],
       [0.        , 0.        , 0.5       ],
       [1.        , 0.7705156 , 0.        ],
       [0.        , 0.        , 0.8030303 ],
       [0.        , 0.56666666, 1.        ],
       [0.        , 0.15882353, 1.        ],
       [0.        , 0.8333333 , 1.        ],
       [0.2624921 , 1.        , 0.70524985],
       [0.16129032, 1.        , 0.8064516 ],
       [0.9203036 , 1.        , 0.04743833],
       [0.        , 0.        , 0.6426025 ],
       [1.        , 0.16049382, 0.        ],
       [1.        , 0.90123457, 0.        ],
       [1.        , 0.27668846, 0.        ],
       [0.49019608, 1.        , 0.47754586],
       [0.37634408, 1.        , 0.5913978 ],
       [0.        , 0.69215685, 1.        ],
       [0.70524985, 1.        , 0.2624921 ],
       [0.        , 0.4254902 , 1.        ],
       [0.        , 0.03333334, 1.        ],
       [0.        , 0.        , 0.9456328 ],
       [0.6426025 , 0.        , 0.        ],
       [1.        , 0.654321  , 0.        ],
       [1.        , 0.4074074 , 0.        ],
       [0.04743833, 0.9588235 , 0.9203036 ],
       [0.9456328 , 0.02977487, 0.        ]])

def draw_vert_lines(image,lines,numeral_positions,col=(0,0,255)):
    xt = 5000
    for pt in lines:
        m = pt[0].item()
        b = pt[1].item()
        pt1 = (round(-xt),round(m * -xt + b))
        pt2 = (round(xt),round(m * xt + b))
        cv2.line(image,pt1,pt2,col,1,1)
    return image

def draw_boundary_line(image,boundary_line):        
    m = boundary_line[0]
    b = boundary_line[1]

    pt1 = (0,int(m*0 + b))
    pt2 = (int(XDIM),int(m*XDIM + b))

    cv2.line(image,pt1,pt2,(0,255,0),1) 
    
    return image
def draw_patch_boxes(image,coarse_boxes,fine_boxes,numeral_positions,boundary_line):
    for box in coarse_boxes:
        x1,y1,x2,y2,score,cls_id = box
        pt1 = (round(x1.item()),round(numeral_positions[1].item()))
        pt2 = (round(x2.item()),round(boundary_line))
        cv2.rectangle(image,pt1,pt2,(255,0,0),2,1)

    for box in fine_boxes:
        x1,y1,x2,y2,score,cls_id = box
        pt1 = (round(x1.item()),round(boundary_line))
        pt2 = (round(x2.item()),round(numeral_positions[0].item()))
        cv2.rectangle(image,pt1,pt2,(0,0,255),2,1)
    return image

def draw_coarse_points(image, points):
    for i,point in enumerate(points):
        x0 = round(point[0].item())
        y0 = round(point[1].item())
        val = point[2]
        degree = m.floor(val)
        min = (val - degree) * 60
        y_offset = 15 + 10*(i%2)
        if min == 0 or min==30:
        # if i%5 == 0:
        # if True:
            text = f'{degree}:{round(min.item())}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            txt_size = cv2.getTextSize(text, font, font_scale, 1)[0]
            cv2.rectangle(
                image,
                (x0, y0 - y_offset),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1]) - y_offset),
                (0,0,0),
                -1
                )
            cv2.putText(image, text, (x0 , y0 + txt_size[1] - y_offset), font, font_scale, (255,255,255), thickness=1)
        cv2.circle(image,(x0,y0 - 1),0,(255,0,0),2)
    return image

def draw_fine_points(image, points):
    for i,point in enumerate(points):
        x0 = round(point[0].item())
        y0 = round(point[1].item())
        val = point[2]
        min = m.floor(val)
        sec = (val - min) * 60
        y_offset = 10 + 10*(i%2)
        # if sec == 0 or sec == 30:
        if True:
            text = f'{min}:{round(sec.item())}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            txt_size = cv2.getTextSize(text, font, font_scale, 1)[0]
            cv2.rectangle(
                image,
                (x0, y0 + y_offset),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1]) + y_offset),
                (0,0,0),
                -1
                )
            cv2.putText(image, text, (x0 , y0 + txt_size[1] + y_offset), font, font_scale, (255,255,255), thickness=1)
        cv2.circle(image,(x0,y0 + 1),0,(0,0,255),2)
    return image

def draw_boxes(img, boxes):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(boxes[i][5])
        score = boxes[i][4]
        x0 = int(box[0].item())
        y0 = int(box[1].item())
        x1 = int(box[2].item())
        y1 = int(box[3].item())
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    return img

def draw_measure(img,hrs,mins,secs):
    text = '{}hrs:{}mins:{}secs'.format(hrs,mins,secs)
    txt_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (50,50), font, 2, txt_color, thickness=3)
    return img