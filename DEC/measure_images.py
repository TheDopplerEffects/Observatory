import os
import glob
import logging
from turtle import fd
import torch as t
from torchvision.transforms.functional import pil_to_tensor

from PIL import Image
from tqdm import tqdm
from time import perf_counter
from post_processing.post_processing import *
from post_processing.utils import *

from config.config import *

def get_gt_measurements(path):
    lbls = {}
    with open(path + 'measure_labels.csv','r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n')
            line = line.split(',')
            lbls[line[0]] = np.array([int(x) if x != 'a' else None for x in line[1:]])
    return lbls

def get_det_labels(path,images):
    """
    Reads in Annotations from Model Results
    """
    labels = {}
    for label_fd in glob.glob(os.path.join(path,'*.txt')):
        file,ext = os.path.splitext(os.path.split(label_fd)[-1])
        with open(label_fd,'r') as f:
            lines = f.readlines()
            dets = []
            for line in lines:
                line = line.split()
                id = float(line[0])
                xc = float(line[1])
                yc = float(line[2])
                w = float(line[3])
                h = float(line[4])
                conf = float(line[5])*100

                x1 = xc - w/2.0
                y1 = yc - h/2.0
                x2 = xc + w/2.0
                y2 = yc + h/2.0
                dets.append([x1,y1,x2,y2,conf,id])
            image_dims = images[file].size
            dets = scale_bbox(t.tensor(dets),image_dims)
            labels[file] = dets
    return labels

def get_gt_labels(path,images):
    """
    Reads in Annotations from Model Results
    """
    labels = {}
    for label_fd in glob.glob(os.path.join(path,'*.txt')):
        file,ext = os.path.splitext(os.path.split(label_fd)[-1])
        with open(label_fd,'r') as f:
            lines = f.readlines()
            dets = []
            for line in lines:
                line = line.split()
                id = float(line[0])
                conf = float(100)
                xc = float(line[1])
                yc = float(line[2])
                w = float(line[3])
                h = float(line[4])

                x1 = xc - w / 2.0
                x2 = xc + w / 2.0

                y1 = yc - h / 2.0
                y2 = yc + h / 2.0
                dets.append([x1,y1,x2,y2,conf,id])
            image_dims = images[file].size
            dets = scale_bbox(t.tensor(dets),image_dims)
            labels[file] = dets
    return labels

def get_baseline_labels(path,images):
    """
    Reads in Annotations from Model Results
    """
    labels = {}
    for label_fd in glob.glob(os.path.join(path,'*.txt')):
        file,ext = os.path.splitext(os.path.split(label_fd)[-1])
        with open(label_fd,'r') as f:
            lines = f.readlines()
            dets = []
            for line in lines:
                line = line.split()
                id = float(line[5])
                conf = float(line[4])*100
                xc = float(line[0])
                yc = float(line[1])
                w = float(line[2])
                h = float(line[3])
                x1 = xc - w / 2
                x2 = xc + w / 2
                y1 = yc - h / 2
                y2 = yc + h / 2
                dets.append([x1,y1,x2,y2,conf,id])
            labels[file] = t.tensor(dets).int()
    return labels

def get_images(path):
    images = {}
    for image_fd in glob.glob(os.path.join(path,'*.png')):
        file,ext = os.path.splitext(os.path.split(image_fd)[-1])
        image = Image.open(image_fd).convert('L')
        images[file] = image
    return images

def scale_bbox(bboxs,image_dims):
    scale_matrix = t.tensor([image_dims[0],image_dims[1],image_dims[0],image_dims[1],1.0,1.0])
    bboxs = bboxs*scale_matrix
    return bboxs.int()

def measure_images(images,labels):
    measurements = {}
    for image_id in images:
        image = images[image_id]
        bboxs = labels[image_id]
        image = np.array(image).astype(np.uint8)
        measurements[image_id] = {}
        try:
            measurement,image = infer_measure(image,bboxs)
            measurements[image_id]['measurement'] = measurement
            measurements[image_id]['image'] = image
        except AssertionError as e:
            print(e,image_id)

            measurements[image_id]['measurement'] = None
            measurements[image_id]['error'] = e
    return measurements

def evaluate(measurements,ground_truth):
    count = 0
    coarse_sum = 0
    fine_sum = 0
    correct = 0
    correct_10 = 0
    correct_20 = 0
    no_detects = 0
    count = 0
    mean_absolute_error = 0
    for img in measurements.keys():
        measurement = measurements[img]['measurement']
        if measurement is not None:
            image = measurements[img]['image']
            hours,coarse_mins,fine_mins,seconds = measurement
            gt = ground_truth[img]
            hrs = round(hours.item())
            coarse_mins = round(coarse_mins.item())
            fine_mins = round(fine_mins.item())
            seconds = round(seconds.item())

            if gt[0] == hours and gt[1] == coarse_mins:
                coarse_sum += 1
            # else:
            #     print(img,gt[0],gt[1],hours,coarse_mins)
            if gt[2] == fine_mins and gt[3] == seconds:
                fine_sum += 1

            combined_mins = coarse_mins + fine_mins
            combined_hrs = hrs
            if combined_mins >= 60:
                if combined_hrs <= 0:
                    combined_hrs -= 1
                else:
                    combined_hrs += 1
                combined_mins -= 60
            
            gt_combined_mins = gt[1] + gt[2]
            gt_hours = gt[0]
            if gt_combined_mins >= 60:
                if gt_hours <= 0:
                    gt_hours -= 1
                else:
                    gt_hours += 1
                gt_combined_mins -= 60
                
            final_measurement = combined_hrs*60*60 + combined_mins*60 + seconds
            gt_final_measurement = gt_hours*60*60 + gt_combined_mins*60 + gt[3]
            if abs(gt_final_measurement - final_measurement) == 0:
                correct += 1

            if abs(gt_final_measurement - final_measurement) <= 10:
                correct_10 += 1
                
            if abs(gt_final_measurement - final_measurement) <= 20:
                correct_20 += 1
            
            mean_absolute_error += abs(gt_final_measurement - final_measurement)

            # if abs(gt_final_measurement - final_measurement) >= 3600:
            #     cv.imwrite('images/' + img + '.png',image)  
            #     print(img,gt,[hrs,coarse_mins,fine_mins,seconds])
            #     print(img,[gt_hours,gt_combined_mins,gt[3]],[combined_hrs,combined_mins,seconds])
        else:
            print(img)
            no_detects += 1  
        
        count += 1
    result = ''
    result += f'Total Images: {count}\n'
    result += f'Coarse Params: {COARSE_THETA_RES:.6f}, {COARSE_RHO_RES}, {COARSE_VOTE_THRESH}\n'
    result += f'Fine Params: {FINE_THETA_RES:.6f}, {FINE_RHO_RES}, {FINE_VOTE_THRESH}\n'
    result += f'Upper Accuracy: {coarse_sum/count}\n'
    result += f'Lower Accuracy: {fine_sum/count}\n'
    result += f'No Detections: {no_detects}\n'
    result += f'Overall Accuracy: {correct/count}\n' 
    result += f'Overall Accuracy <= 10: {correct_10/count}\n' 
    result += f'Overall Accuracy <= 20: {correct_20/count}\n' 
    result += f'MAE: {mean_absolute_error/count}'
    print(result)

if __name__ == '__main__':
    path = '/home/tim/Documents/Datasets/VernierImages/DEC/DecSplits/test/'
    impath = '/home/tim/Documents/Datasets/VernierImages/DEC/DecSplits/test/images/'
    lblpath = '/home/tim/ssd/Projects/VernierBaseline/DEC/results_ext/'
    lbl_src = 'base'

    gt = get_gt_measurements(path)
    images = get_images(impath)
    if lbl_src == 'gt':
        labels = get_gt_labels(lblpath,images)
    elif lbl_src == 'base':
        labels = get_baseline_labels(lblpath,images)
    else:
        labels = get_det_labels(lblpath,images)
    
    measurements = measure_images(images,labels)
    evaluate(measurements,gt)