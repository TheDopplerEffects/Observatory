import cv2
import os
import torch as t
import csv
from datetime import datetime
from time import perf_counter
from post_processing.post_processing import *

np.set_printoptions(suppress=True)
t.set_printoptions(sci_mode=False)

def evaluate(detections,ground_truth):
    correct = 0
    correct_two = 0
    correct_four = 0
    error = 0
    coarse_sum = 0
    fine_sum = 0

    no_detects = 0

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

            for sec in gt[2:]:
                diff = abs(seconds - sec)
                if diff < min_val:
                    min_val = diff
                    gt_secs = sec

            gt[2] = gt_secs
                
            if gt[0] == hrs and gt[1] == mins:
                coarse_sum += 1 
            else:
                cv.imwrite(f'images/{img}.png',image)

            if gt[2] == seconds:
                fine_sum += 1
            
            s1 = f'{int(gt[0])}:{int(gt[1])}:{int(gt[2])}'
            s2 = f'{int(hours)}:{int(mins)}:{int(seconds)}'
            FMT = '%H:%M:%S'
            t1 = datetime.strptime(s2,FMT)
            t2 = datetime.strptime(s1,FMT)
            if t1 > t2:
                td = t1 - t2
            else:
                td = t2 - t1
            error += abs(td.total_seconds())

            if abs(td.total_seconds()) == 0:
                correct += 1

            if abs(td.total_seconds()) <= 2:
                correct_two += 1
            
            if abs(td.total_seconds()) <= 4:
                correct_four += 1
        else:
            print(img)
            no_detects += 1  

    count = len(detections.keys())
    result = ''
    result += f'Total Images: {count}\n'
    result += f'Coarse Params: {COARSE_THETA_RES:.6f}, {COARSE_RHO_RES}, {COARSE_VOTE_THRESH}\n'
    result += f'Fine Params: {FINE_THETA_RES:.6f}, {FINE_RHO_RES}, {FINE_VOTE_THRESH}\n'
    result += f'Upper Scale Accuracy: {coarse_sum/count}\n'
    result += f'Lower Scale Accuracy: {fine_sum/count}\n'
    result += f'No Detections: {no_detects}\n'
    result += f'Overall Accuracy: {correct/count}\n' 
    result += f'Overall Accuracy <= 2: {correct_two/count}\n' 
    result += f'Overall Accuracy <= 4: {correct_four/count}\n' 
    result += f'MSE: {error/count}'
    print(result)

if __name__ == '__main__':
    path = '/home/tim/Documents/Datasets/VernierImages/RA/yolo/RAScaleMeasure/'
    # lblpath = '/home/tim/Documents/Datasets/VernierImages/RA/yolo/RAScaleMeasure/labels/'
    lbl_src = 'base'
    # lblpath = '/ssd/Models/YOLOv6/runs/inference/yolov6m_RA_V3/labels/'
    lblpath = '/ssd/Projects/VernierBaseline/RA/results_80/'
    fds = os.listdir(path + 'images/')
    detections = {}
    gt = {}
    labels = []
    with open(path+'labels.csv',newline='') as csvfile:
        writer = csv.reader(csvfile, delimiter=',')
        for row in writer:
            labels.append(list(map(int,row)))
            
    for i,measure in enumerate(labels):
        gt[f'{i}.png'] = measure
    intervals = []
    # fds = ['0.png']
    for fd in fds:
        dets = []
        image = cv.imread(path + 'images/' + fd,cv.IMREAD_GRAYSCALE)
        with open(f'{lblpath}{int(fd[:-4])}.txt','r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip().split(' ')
                if lbl_src == 'gt':
                    id = float(line[5])
                    conf = float(line[4])
                    x1 = float(line[0])
                    y1 = float(line[1])
                    x2 = float(line[2])
                    y2 = float(line[3])
                elif lbl_src == 'dets':
                    id = float(line[0])
                    xc = float(line[1]) * image.shape[1]
                    yc = float(line[2]) * image.shape[0]
                    w = float(line[3]) * image.shape[1]
                    h = float(line[4]) * image.shape[0]
                    conf = float(line[5])

                    x1 = xc - w/2
                    y1 = yc - h/2
                    x2 = xc + w/2
                    y2 = yc + h/2
                else:
                    id = float(line[5])
                    conf = float(line[4])
                    x1 = float(line[0])
                    y1 = float(line[1])
                    x2 = float(line[2])
                    y2 = float(line[3])
 
                dets.append([x1,y1,x2,y2,conf,id])
        dets = t.Tensor(dets)
        try:
            t1 = perf_counter()
            measurement,image = infer_measure(image,dets,save_img=True,fd=fd,save_patches=True,save_thresh=False)
            intervals.append(perf_counter() - t1)
            detections[fd] = {}
            detections[fd]['measurement'] = measurement
            detections[fd]['image'] = image
        except AssertionError as e:
            detections[fd] = {}
            detections[fd]['measurement'] = None
            detections[fd]['image'] = image
            detections[fd]['error'] = e
            
    evaluate(detections,gt)