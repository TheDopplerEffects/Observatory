import numpy as np
import torch as t
class_names = ['20','40','60','A','Arrow','B','I','II','III','IIII','IX','V',
            'VI','VII','VIII','X','XI','XII','XIII','XIV','XIX',
            'XV','XVI','XVII','XVIII','XX','XXI','XXII','XXIII','XXIV']

NUMERAL_LUT = {'I': 1, 'II': 2, 'III': 3, 'IIII': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10, 
                'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 
                'XX': 20, 'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 0,'20':20,'40':40,'60':60,'Arrow':0}

ROMAN_LUT = ['I','II','III','IIII','IX','V',
            'VI','VII','VIII','X','XI','XII','XIII','XIV','XIX',
            'XV','XVI','XVII','XVIII','XX','XXI','XXII','XXIII','XXIV']

## PARAMS
XDIM = 1440
YDIM = 1080

# Sobel Kernel
VS_KERNEL = 3
HS_KERNEL = 1

# Threshold Gaussian Kernel
TH_XKERNEL = (9,5)
TH_YKERNEL = (5,9)

# Tick Detection Patch Size
PATCH_WIDTH = 20

# Hough Line Theta Tolerance and Theta Resolution
XTOL = 0.005
YTOL = 0.02

SEGMENT_THETA_RES = np.pi/1080
SEGMENT_RHO_RES = 1
SEGMENT_VOTE_RES = 1000

COARSE_THETA_RES = np.pi/(360 * 2**2)
COARSE_RHO_RES = 1
COARSE_VOTE_THRESH = 130

FINE_THETA_RES = np.pi/ (360 * 2**2)
FINE_RHO_RES = 1
FINE_VOTE_THRESH = 115

Y_MINLINEDIST = 115

# Yolo Model Params
CONF_THRESH = 0.70

MIN_BOX_SIZE = 1000
NMS_CONF_THRESH = 0.25
NMS_IOU_THRESH = 0.0

DIST_FROM_EDGE = 0.005 # Percentage

# Error Checking Params
MIN_COARSE_POINT_DISTANCE = 5
EXPECTED_SECOND_SEQUENCE = t.tensor([61, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10,  8, 6,  4,  2,  0, -2])
