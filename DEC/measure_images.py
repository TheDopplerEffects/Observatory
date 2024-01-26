import cv2 as cv
import torch as t
import os
from tqdm import tqdm

from post_processing.post_processing import *
from post_processing.utils import *

from config.config import *
path = '/home/tim/Documents/Datasets/VernierImages/DEC/DecSplits/test/'
impath = '/home/tim/Documents/Datasets/VernierImages/DEC/DecSplits/test/images/'
lblpath = '/home/tim/Documents/Datasets/VernierImages/DEC/DecSplits/test/labels/'

class_names = ['20', '40', '60', 'A', 'Arrow', 'B', 'I', 'II', 'III', 'IIII', 'IX', 'V',
               'VI', 'VII', 'VIII', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XIX', 'XV', 'XVI', 
               'XVII', 'XVIII', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', '28', '46', '64', 
               '82', '100', '0', '5', '10', '15', '25', '30', '35', '45', '50', 
               '55', '65', '70', '75', '80', '85', '90']

lbls = {}
with open(path + 'measure_labels.csv','r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip('\n')
        line = line.split(',')
        lbls[line[0]] = [int(x) if x != 'a' else x for x in line[1:]]


fds = os.listdir(impath)
hr_sum = 0
count = 0
coarse_min_sum = 0
fine_min_sum = 0
sec_sum = 0
sec_within_error_sum = 0
sec_within_error_sum_f = 0
correct = 0
correct_two = 0
correct_four = 0
no_detects = 0
count = len(fds)
# for fd in tqdm(fds):
for fd in fds:
# for fd in [0]:
    # fd = '2d33ba49ac67439b.png'
    image = cv.imread(impath + fd,cv.IMREAD_GRAYSCALE)
    img_id = fd[:-4]
    with open(lblpath + img_id + '.txt','r') as f:
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
    try:
        measure,image = infer_measure(image,dets,save_img=False,fd=fd,save_patches=False,save_thresh=False)

        if measure is not None:
            hours,coarse_mins,fine_mins,seconds = measure
            hrs = round(hours.item())
            coarse_mins = round(coarse_mins.item())
            fine_mins = round(fine_mins.item())
            seconds = round(seconds.item())
            gt = lbls[img_id]

            if gt[0] == hrs:
                hr_sum += 1

            if gt[1] == coarse_mins:
                coarse_min_sum += 1

            if gt[2] == fine_mins:
                fine_min_sum += 1    

            if gt[3] == seconds:
                sec_sum += 1

            if abs(gt[2] - seconds) <= 10:
                sec_within_error_sum += 1
            if abs(gt[2] - seconds) <= 20:
                sec_within_error_sum_f += 1

            if gt[0] == hrs and gt[1] == coarse_mins and gt[2] == fine_mins and gt[3] == seconds:
                correct += 1
            if gt[0] == hrs and gt[1] == coarse_mins and gt[2] == fine_mins and abs(gt[3] - seconds) <= 2:
                correct_two += 1
            if gt[0] == hrs and gt[1] == coarse_mins and gt[2] == fine_mins and abs(gt[3] - seconds) <= 4:
                correct_four += 1
            
            if abs(gt[2] - fine_mins) > 1 and abs(gt[2] - fine_mins) < 5:
                print(img_id,gt,[hrs,coarse_mins,fine_mins,seconds])
                cv.imwrite('images/' + fd,image)
                            
                
                # s1 = f'{int(gt[0])}:{int(gt[1])}:{int(gt[2])}'
                # s2 = f'{int(hours)}:{int(mins)}:{int(seconds)}'
                # FMT = '%H:%M:%S'
                # t1 = datetime.strptime(s2,FMT)
                # t2 = datetime.strptime(s1,FMT)
                # if t1 > t2:
                #     td = t1 - t2
                # else:
                #     td = t2 - t1
                # # print(img,s1,s2,td.total_seconds())
                # error += td.total_seconds()**2
    except Exception as e:
        # print(fd)
        print(fd,e)
        no_detects += 1  
    # break
        
result = ''
result += f'Total Images: {count}\n'
result += f'Coarse Params: {COARSE_THETA_RES:.6f}, {COARSE_RHO_RES}, {COARSE_VOTE_THRESH}\n'
result += f'Fine Params: {FINE_THETA_RES:.6f}, {FINE_RHO_RES}, {FINE_VOTE_THRESH}\n'
result += f'Hr Accuracy: {hr_sum/count}\n'
result += f'Coarse Min Accuracy: {coarse_min_sum/count}\n'
result += f'Fine Min Accuracy: {fine_min_sum/count}\n'
result += f'Sec Accuracy: {sec_sum/count}\n'
result += f'Sec <=10 Accuracy: {sec_within_error_sum/count}\n'
result += f'Sec <=20 Accuracy: {sec_within_error_sum_f/count}\n'
result += f'No Detections: {no_detects}\n'
# result += f'Overall Accuracy: {correct/count}\n' 
# result += f'Overall Accuracy <= 2: {correct_two/count}\n' 
# result += f'Overall Accuracy <= 4: {correct_four/count}\n' 
# result += f'MSE: {error/count}'
print(result)
# cv.imwrite('image.png',image)