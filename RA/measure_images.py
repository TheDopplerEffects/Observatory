import cv2
import os
import torch as t
import csv
from datetime import datetime
from post_processing.post_processing import *

np.set_printoptions(suppress=True)
t.set_printoptions(sci_mode=False)

def evaluate(detections,ground_truth):
    correct = 0
    correct_two = 0
    correct_four = 0
    error = 0
    hr_sum = 0
    min_sum = 0
    sec_sum = 0
    no_detects = 0
    sec_within_error_sum = 0
    sec_within_error_sum_f = 0

    for img in detections.keys():
        measurement = detections[img]['measurement']
        image = detections[img]['image']
        if measurement is not None:
            hours,mins,seconds = measurement
            hrs = int(hours.item())
            mins = int(mins.item())
            seconds = int(seconds.item())
            gt = ground_truth[img]
            min_val = 1e6
            # if len(gt) == 4:
            #     print(img,(hours,mins,seconds),gt)
            for sec in gt[2:]:
                diff = abs(seconds - sec)
                if diff < min_val:
                    min_val = diff
                    gt_secs = sec

            gt[2] = gt_secs
            
            if gt[0] == hrs:
                hr_sum += 1

            if gt[1] == mins:
                min_sum += 1
                    
            if gt[2] == seconds:
                sec_sum += 1

            if abs(gt[2] - seconds) <= 2:
                sec_within_error_sum += 1
            if abs(gt[2] - seconds) <= 4:
                sec_within_error_sum_f += 1

            if gt[0] == hrs and gt[1] == mins and gt[2] == seconds:
                correct += 1
            if gt[0] == hrs and gt[1] == mins and abs(gt[2] - seconds) <= 2:
                correct_two += 1
            if gt[0] == hrs and gt[1] == mins and abs(gt[2] - seconds) <= 4:
                correct_four += 1
            
            # if gt[1] != mins:
            #     print(img)            
            
            s1 = f'{int(gt[0])}:{int(gt[1])}:{int(gt[2])}'
            s2 = f'{int(hours)}:{int(mins)}:{int(seconds)}'
            FMT = '%H:%M:%S'
            t1 = datetime.strptime(s2,FMT)
            t2 = datetime.strptime(s1,FMT)
            if t1 > t2:
                td = t1 - t2
            else:
                td = t2 - t1
            # print(img,s1,s2,td.total_seconds())
            error += td.total_seconds()**2
        else:
            no_detects += 1  
    count = len(detections.keys())
    result = ''
    result += f'Total Images: {count}\n'
    result += f'Coarse Params: {COARSE_THETA_RES:.6f}, {COARSE_RHO_RES}, {COARSE_VOTE_THRESH}\n'
    result += f'Fine Params: {FINE_THETA_RES:.6f}, {FINE_RHO_RES}, {FINE_VOTE_THRESH}\n'
    result += f'Hr Accuracy: {hr_sum/count}\n'
    result += f'Min Accuracy: {min_sum/count}\n'
    result += f'Sec Accuracy: {sec_sum/count}\n'
    result += f'Sec <=2 Accuracy: {sec_within_error_sum/count}\n'
    result += f'Sec <=4 Accuracy: {sec_within_error_sum_f/count}\n'
    result += f'No Detections: {no_detects}\n'
    result += f'Overall Accuracy: {correct/count}\n' 
    result += f'Overall Accuracy <= 2: {correct_two/count}\n' 
    result += f'Overall Accuracy <= 4: {correct_four/count}\n' 
    result += f'MSE: {error/count}'
    print(result)

if __name__ == '__main__':
    path = '/home/tim/Documents/Datasets/VernierImages/RA/yolo/RAScaleMeasure/'
    # lblpath = '/home/tim/Documents/Datasets/VernierImages/RA/yolo/RAScaleMeasure/labels/'
    lblpath = '/home/tim/Documents/Projects/VernierBaseline/results/'
    fds = os.listdir(path + 'images/')
    # fds = ['25.png']
    # fds = ['204.png','159.png','95.png','60.png']
    detections = {}
    gt = {}
    labels = []
    with open(path+'labels.csv',newline='') as csvfile:
        writer = csv.reader(csvfile, delimiter=',')
        for row in writer:
            labels.append(list(map(int,row)))
            
    for i,measure in enumerate(labels):
        gt[f'{i}.png'] = measure

    for fd in fds:
        dets = []
        image = cv.imread(path + 'images/' + fd,cv.IMREAD_GRAYSCALE)
        with open(f'{lblpath}{int(fd[:-4])}.txt','r') as f:
            lines = f.readlines()
            for line in lines:
                dets.append([float(x) for x in line.split(' ')])

        dets = t.Tensor(dets)
        try:
            measurement,image = infer_measure(image,dets,save_img=True,fd=fd,save_patches=False)
            detections[fd] = {}
            detections[fd]['measurement'] = measurement
            detections[fd]['image'] = image
        except AssertionError:
            detections[fd] = {}
            detections[fd]['measurement'] = None
            detections[fd]['image'] = image
        cv.imwrite(f'images/{fd}',image)

    evaluate(detections,gt)