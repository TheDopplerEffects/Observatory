import numpy as np
import torch as t
import cv2 as cv
import math
import timeit

from post_processing.error_checking import *
from config.config import *
from post_processing.utils import *
from pprint import pprint

def post_process_boxes(boxes,boundary,conf=0.6):
    """
    Post Processing for Detected BBoxes
    :param boxes: Bounding Boxes -> tensor([x1,y1,x2,y2,score,cls_id])
    :param boundary: Detected Region Boundary
    :returns: Bounding Boxes Grouped Based on Region Boundary and Sorted in Raster Order and Average X and Y Positions of the Groups
    """
    assert boxes is not None, 'Box Post Processing Error: No Boxes Detected'
    fine_boxes = None
    coarse_boxes = None
    
    # Filter Based on Region Boundary and Remove A or B Label
    for i,box in enumerate(boxes):
        if box[-1] != 5 and box[-1] != 3 and box[4].item() >= conf and (box[2] - box[0]) * (box[3] - box[1]) > MIN_BOX_SIZE: # Remove A and B Labels, Detections Below Confidence Threshold and Small Bounding Boxes
            if box[0] > XDIM*(DIST_FROM_EDGE) and box[2] < XDIM*(1-DIST_FROM_EDGE): # Removes Boxes too close to edge which are likely to be misdetected
                if box[1] > boundary:
                    if fine_boxes is None:
                        fine_boxes = box.expand(1,6)
                    else:
                        fine_boxes = t.cat((fine_boxes,box.expand(1,6)))

                if box[1] < boundary:
                    if coarse_boxes is None:
                        coarse_boxes = box.expand(1,6)
                    else:
                        coarse_boxes = t.cat((coarse_boxes,box.expand(1,6)))
    
    # Sort L to R
    assert fine_boxes is not None, 'No Fine Boxes Detected'
    assert coarse_boxes is not None, 'No Coarse Boxes Detected'

    sort = t.sort(fine_boxes[:,0],descending=False)
    fine_boxes_sorted = fine_boxes[sort.indices]
    sort = t.sort(coarse_boxes[:,0],descending=False)
    coarse_boxes_sorted = coarse_boxes[sort.indices]

    # Get Appox Numeral Lines
    fine_y_position = t.sum(fine_boxes_sorted[:,1])/len(fine_boxes_sorted[:,1])
    coarse_y_position = t.sum(coarse_boxes_sorted[:,3])/len(coarse_boxes_sorted[:,3])

    assert fine_boxes_sorted is not None, 'Dectection Grouping Error: Sorted Fine Boxes Emtpy'
    assert coarse_boxes_sorted is not None, 'Dectection Grouping Error: Sorted Coarse Boxes Emtpy'
    
    return fine_boxes_sorted,coarse_boxes_sorted, (fine_y_position,coarse_y_position)

def split_image(img,numeral_positions,save_thresh=False,fd=None):
    """
    Detecteds image center line using HoughLine Algorithm
    :param img: Grayscale Image of Scale
    :param numeral_positions: Average locations of numerals
    :returns: Bounding Boxes Grouped Based on Region Boundary and Sorted in Raster Order and Average X and Y Positions of the Groups
    """
    # Crop Image Between Detected Objects to Reduce size of HoughLine image
    fine_pos = int(numeral_positions[0])
    coarse_pos = int(numeral_positions[1])
    cropped_img = img[coarse_pos:fine_pos,:]

    # Filter, Threshold and Houghlines
    edge_map = cv.Sobel(cropped_img,cv.CV_8U,0,1,ksize=VS_KERNEL)
    _ , yThresh = cv.threshold(cv.GaussianBlur(edge_map,TH_YKERNEL,0),0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

    if save_thresh:
        cv.imwrite(fd,yThresh)
        
    yLines = cv.HoughLines(yThresh, SEGMENT_RHO_RES, SEGMENT_THETA_RES, SEGMENT_VOTE_RES, None, 0, 0, np.pi/2 - YTOL, np.pi/2 + YTOL)

    assert yLines is not None, 'Image Split Error: No Boundary Lines Detected'

    theta_sum = 0
    rho_sum = 0
    
    # Accumulate Multiple Line Detections
    for k in range(0, len(yLines)):
        rho = yLines[k][0][0]
        theta = yLines[k][0][1]
        theta_sum += theta
        rho_sum += rho
    
    theta_offset = theta_sum/len(yLines)
    rho_offset = rho_sum/len(yLines)
    
    # Convert Line to Slope-Intercept Form
    m = -(math.cos(theta_offset)/math.sin(theta_offset))
    b = (rho_offset/math.sin(theta_offset))
    center = round(img.shape[1]//2 * m + b)
    boundary_edge = round(center + coarse_pos)
    boundary_line = [m,b + coarse_pos]
    
    return img[coarse_pos:boundary_edge],img[boundary_edge:fine_pos],boundary_line

def is_roman(cls_id):
    return class_names[cls_id] in ROMAN_LUT

def is_minute(cls_id):
    return cls_id in [0,1,2]

def get_minutes(cls_ids,i):
    if is_roman(cls_ids[i]):
        return 0
    elif is_minute(cls_ids[i]):
        return NUMERAL_LUT[class_names[cls_ids[i]]]
    elif cls_ids[i] == 4:
        return 0

def get_coarse_line_positions(boxes):
    """
    Determines Quarter Tick Line Positions and Labels using Coarse Bounding Boxes
    :param boxes: Coarse Scale Bounding Boxes, [x1,y1,x2,y2,score,cls_id]
    :param image_dims: Original Grayscale Image Dimensions
    :returns: Approximate Quarter Tick Positions and Labels -> tensor([x]), tensor([hours,mins])
    """
    line_positions = None
    line_labels = None
    for i in range(len(boxes[:-1])):
        curr_box = boxes[i]
        curr_value = NUMERAL_LUT[class_names[curr_box[5]]]

        next_box = boxes[i + 1]
        next_value = NUMERAL_LUT[class_names[next_box[5]]]

        position_delta = (next_box[0] + (next_box[2] - next_box[0])/2) - (curr_box[0] + (curr_box[2] - curr_box[0])/2)
        value_delta = next_value - curr_value
        x = (curr_box[0] + (curr_box[2] - curr_box[0])/2)
        
        dX = position_delta/5
        dV = value_delta/5
        if i == 0:
            for j in range(1,EXTEND_DEGREES + 1):
                position = t.tensor([x - j*dX])
                label = t.tensor([curr_value - j*dV])
                if position > 0:
                    if line_positions is None:
                        line_positions = position
                    else:
                        line_positions = t.cat((position,line_positions))
                    if line_labels is None:
                        line_labels = label
                    else:
                        line_labels = t.cat((label,line_labels))

        for j in range(5):
            position = t.tensor([x + j*dX])
            label = t.tensor([curr_value + j*dV])
            if line_positions is None:
                line_positions = position
            else:
                line_positions = t.cat((line_positions,position))
            if line_labels is None:
                line_labels = label
            else:
                line_labels = t.cat((line_labels,label))
    last_box = boxes[-1]

    last_val = t.tensor([NUMERAL_LUT[class_names[last_box[5]]]])
    last_position = t.tensor([(last_box[0] + (last_box[2] - last_box[0])/2)])

    line_positions = t.cat((line_positions,last_position))
    line_labels = t.cat((line_labels,last_val))

    if last_val == 0 or last_val == 90:
        dV = -dV

    for j in range(1,EXTEND_DEGREES + 1):
        position = t.tensor([last_position + j*dX])
        label = t.tensor([last_val + j*dV])
        if line_positions is None:
            line_positions = position
        else:
            line_positions = t.cat((line_positions,position))
        if line_labels is None:
            line_labels = label
        else:
            line_labels = t.cat((line_labels,label))

    return line_positions,line_labels
        
def get_coarse_lines(coarse_segment,line_positions,line_labels,position,save_patches=False,fd=None):
    # Use Approx Line Positions to Slice Image and Use Hough to Detect

    coarse_lines = None
    vert_edges = cv.Sobel(coarse_segment,cv.CV_8U,1,0,ksize=VS_KERNEL)
    _ , threshold = cv.threshold(vert_edges,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
    for i,line in enumerate(line_positions):
        patch = threshold[:,int(line) - PATCH_WIDTH:int(line) + PATCH_WIDTH]
        # Rotate Patch 90deg to avoid limit as angle approaches zero
        patch_rotated = cv.rotate(patch,cv.ROTATE_90_CLOCKWISE)
        linesX = cv.HoughLines(patch_rotated,COARSE_RHO_RES,COARSE_THETA_RES,COARSE_VOTE_THRESH,None,0,0,np.pi/2 - COARSE_XTOL, np.pi/2 + COARSE_XTOL)
        if linesX is not None:
            theta_sum = 0
            rho_sum = 0
            # Average Detected Lines by rho and theta
            for k in range(0, len(linesX)):
                rho = linesX[k][0][0]
                theta = linesX[k][0][1]
                theta_sum += theta
                rho_sum += rho
            theta_offset = theta_sum/len(linesX)
            rho_offset = rho_sum/len(linesX)
            
            m = -np.cos(theta_offset)/np.sin(theta_offset)
            b = rho_offset/np.sin(theta_offset)

            # Rotate Line by 90 CC (x,y) -> (-y,x)     y = -(cos(t)/sin(t)) x + (rho/sin(t)) -> y = (sin(t)/cos(t)) x - (rho/cos(t))
            # y = m x + b -> 
            m2 = np.sin(theta_offset)/np.cos(theta_offset)
            b2 = -rho_offset/np.cos(theta_offset)

            intercept_offset = patch.shape[0]
            b2 += intercept_offset
            # Offset of new origin
            dX = -(int(line.item()) - PATCH_WIDTH)
            dY = -position
            
            # New intercept after translation
            b3 = b2 - dY + m2*dX

            if coarse_lines is None:
                coarse_lines = t.tensor([[m2,b3,line_labels[i]]])
            else:
                coarse_lines = t.cat((coarse_lines,t.tensor([[m2,b3,line_labels[i]]])))

        if save_patches and linesX is not None:
            patch = cv.cvtColor(patch,cv.COLOR_GRAY2BGR)
            patch_rotated = cv.cvtColor(patch_rotated,cv.COLOR_GRAY2BGR)

            dim = patch.shape
            xt = 500
            pt1 = (-xt,round(m2 * -xt + b2))
            pt2 = (round(xt),round(m2*xt + b2))
            cv.line(patch,pt1,pt2,(0,0,255),1)
            cv.imwrite(f'images/coarse_patch_{line_labels[i]}_{fd}',patch)

            dim = patch_rotated.shape
            xt = 5000
            pt1 = (0,round(m * 0 + b))
            pt2 = (round(dim[1]),round(m*dim[1] + b))
            cv.line(patch_rotated,pt1,pt2,(0,0,255),1)
            cv.imwrite(f'images/coarse_patch_rot_{line_labels[i]}_{fd}',patch_rotated)
        elif save_patches:
            cv.imwrite(f'images/coarse_patch_{line_labels[i]}_{fd}',patch)

                
    assert coarse_lines is not None, 'Coarse Line Detection Error: No Lines Detected'
    
    return coarse_lines

def get_fine_line_positions(boxes):
    ## Gets Line Positions of Fine Scale Using Hough Transform
    # Determine X position of quarter ticks for Hough Transform
    
    line_labels = None
    line_positions = None

    for i in range(len(boxes[:-1])):
        curr_box = boxes[i]
        curr_value = NUMERAL_LUT[class_names[curr_box[5]]]

        next_box = boxes[i + 1]
        next_value = NUMERAL_LUT[class_names[next_box[5]]]

        position_delta = (next_box[0] + (next_box[2] - next_box[0])/2) - (curr_box[0] + (curr_box[2] - curr_box[0])/2)
        value_delta = next_value - curr_value
        x = (curr_box[0] + (curr_box[2] - curr_box[0])/2)
        
        dX = position_delta/2
        dV = value_delta/2

        for j in range(2):
            position = t.tensor([x + j*dX])
            label = t.tensor([curr_value + j*dV])
            if line_positions is None:
                line_positions = position
            else:
                line_positions = t.cat((line_positions,position))
            if line_labels is None:
                line_labels = label
            else:
                line_labels = t.cat((line_labels,label))
    last_box = boxes[-1]

    last_val = t.tensor([NUMERAL_LUT[class_names[last_box[5]]]])
    last_position = t.tensor([(last_box[0] + (last_box[2] - last_box[0])/2)])

    line_positions = t.cat((line_positions,last_position))
    line_labels = t.cat((line_labels,last_val))
    
    assert line_positions is not None, 'Fine Line Position Error: No Line Position'
    assert line_labels is not None, 'Fine Line Position Error: No Line Labels'
    
    return line_positions,line_labels

def get_fine_lines(fine_segment,line_positions,line_labels,position,save_patches=False,fd=None):
    fine_lines = None
    vert_edges = cv.Sobel(fine_segment,cv.CV_8U,1,0,ksize=VS_KERNEL)
    _ , threshold = cv.threshold(vert_edges,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
    for i,line in enumerate(line_positions):
        patch = threshold[:,int(line) - PATCH_WIDTH:int(line) + PATCH_WIDTH]

        # Rotate Patch 90deg to avoid limit as angle approaches zero
        patch_rotated = cv.rotate(patch,cv.ROTATE_90_CLOCKWISE)
        
        linesX = cv.HoughLines(patch_rotated,FINE_RHO_RES,FINE_THETA_RES,FINE_VOTE_THRESH,None,0,0,np.pi/2 - FINE_XTOL, np.pi/2 + FINE_XTOL)

        if linesX is not None:
            theta_sum = 0
            rho_sum = 0
            for k in range(0, len(linesX)):
                rho = linesX[k][0][0]
                theta = linesX[k][0][1]
                theta_sum += theta
                rho_sum += rho
            
            theta_offset = theta_sum/len(linesX)
            rho_offset = rho_sum/len(linesX)

            m = -np.cos(theta_offset)/np.sin(theta_offset)
            b = rho_offset/np.sin(theta_offset)

            # Rotate Line by 90 CC (x,y) -> (-y,x)     y = -(cos(t)/sin(t)) x + (rho/sin(t)) -> y = (sin(t)/cos(t)) x - (rho/cos(t))
            m2 = np.sin(theta_offset)/np.cos(theta_offset)
            b2 = -rho_offset/np.cos(theta_offset)

            # Offset of new origin
            dX = -(int(line.item()) - PATCH_WIDTH)
            dY = -position
            
            intercept_offset = patch.shape[0]
            b2 += intercept_offset
            # New intercept after translation
            b3 = b2 - dY + m2*dX
            
            if fine_lines is None:
                fine_lines = t.tensor([[m2,b3,line_labels[i]]])
            else:
                fine_lines = t.cat((fine_lines,t.tensor([[m2,b3,line_labels[i]]])))
                
            if save_patches:
                patch = cv.cvtColor(patch,cv.COLOR_GRAY2BGR)
                patch_rotated = cv.cvtColor(patch_rotated,cv.COLOR_GRAY2BGR)

                dim = patch.shape
                xt = 5000
                pt1 = (-xt,round(m2 * -xt + b2))
                pt2 = (round(xt),round(m2*xt + b2))
                cv.line(patch,pt1,pt2,(0,0,255),1)
                cv.imwrite(f'images/fine_patch_{i}_{fd}',patch)

                dim = patch_rotated.shape
                xt = 5000
                pt1 = (-xt,round(m * -xt + b))
                pt2 = (round(xt),round(m*xt + b))
                cv.line(patch_rotated,pt1,pt2,(0,0,255),1)
                cv.imwrite(f'images/fine_patch_rot_{i}_{fd}',patch_rotated)
                
    assert fine_lines is not None, 'Fine Line Detection Error: No Lines Detected'

    return fine_lines

def get_coarse_points(lines,boundary_line):
    m1 = boundary_line[0]
    b1 = boundary_line[1]
    coarse_points = None
    for pt in lines:
        m2 = pt[0].item()
        b2 = pt[1].item()

        x0 = (b2 - b1)/(m1 - m2)
        y0 = -(b1*m2 - b2*m1)/(m1 - m2)
        
        if coarse_points is None:
            coarse_points = t.tensor([[x0,y0,pt[2]]])
        else:
            coarse_points = t.cat((coarse_points,t.tensor([[x0,y0,pt[2]]])))
    return coarse_points

def get_fine_points(lines,boundary_line):
    m1 = boundary_line[0]
    b1 = boundary_line[1]
    fine_points = None
    for pt in lines:
        m2 = pt[0].item()
        b2 = pt[1].item()

        x0 = (b2 - b1)/(m1 - m2)
        y0 = -(b1*m2 - b2*m1)/(m1 - m2)

        if fine_points is None:
            fine_points = t.tensor([[x0,y0,pt[2]]])
        else:
            fine_points = t.cat((fine_points,t.tensor([[x0,y0,pt[2]]])))
    return fine_points

def label_coarse_points(points):
    points_out = None
    for i,point in enumerate(points[:-1]):
        current_val = point[2]
        next_val = points[i + 1][2]
        degree_gap = next_val - current_val
        steps = abs(6*degree_gap)
        iX = point[0]
        iY = point[1]
        dX = (points[i + 1][0] - iX)/steps
        dY = (points[i + 1][1] - iY)/steps
        for j in range(int(steps.item())):
            if points_out is None:
                points_out = t.tensor([[iX + dX*j,iY + dY*j,current_val + j*degree_gap/6]])
            else:
                points_out = t.cat((points_out,t.tensor([[iX + dX*j,iY + dY*j,current_val + j*degree_gap/6]])))
    last_point = points[-1]
    points_out = t.cat((points_out,last_point.unsqueeze(0)))
    current_val = last_point[2]
    iX = last_point[0]
    iY = last_point[1]
    if current_val == 0 or current_val == 90:
        degree_gap = -degree_gap
    # if extend_right:
    #     for j in range(1,EXTEND_DEGREES*6):
    #         if points_out is None:
    #             points_out = t.tensor([[iX + dX*j,iY + dY*j,current_val + j*degree_gap/6]])
    #         else:
    #             points_out = t.cat((points_out,t.tensor([[iX + dX*j,iY + dY*j,current_val + j*degree_gap/6]])))
    return points_out

def label_fine_points(points):
    points_out = None
    for i,point in enumerate(points[:-1]):
        current_val = point[2]
        next_val = points[i + 1][2]
        degree_gap = next_val - current_val
        steps = abs(6*degree_gap)
        iX = point[0]
        iY = point[1]
        dX = (points[i + 1][0] - iX)/steps
        dY = (points[i + 1][1] - iY)/steps
        for j in range(int(steps.item())):
            if points_out is None:
                points_out = t.tensor([[iX + dX*j,iY + dY*j,current_val + j*degree_gap/6]])
            else:
                points_out = t.cat((points_out,t.tensor([[iX + dX*j,iY + dY*j,current_val + j*degree_gap/6]])))

    points_out = t.cat((points_out,t.tensor([[points[-1][0],points[-1][1],0]])))
    return points_out

def measure_degree_mins(coarse_points,fine_points):
    arrow_point = fine_points[-2][0]
    k = 0
    if len(coarse_points) > 1:
        while round(arrow_point.item() - coarse_points[k][0].item()) > 0 and k < len(coarse_points) - 1:
            k += 1
            point = coarse_points[k]
        degrees = t.floor(point[2])
        mins = (point[2] - degrees)*60
    else:
        degrees,mins = (-1,-1)
    return degrees,mins

def measure_seconds(coarse_points,fine_points):
    y_low = fine_points[0][0]
    y_high = fine_points[-1][0]

    coarse_low = 0
    coarse_high = 0
    for i,point in enumerate(coarse_points):
        if point[0] < y_low:
            coarse_low = i
        
        if point[0] > y_high:
            coarse_high = i
            break

    coarse_slice = coarse_points[coarse_low:coarse_high]

    coarse_x_positions = coarse_points[:,1]
    fine_x_positions = fine_points[:,1]

    diff_matrix = t.abs(coarse_x_positions.unsqueeze(1) - fine_x_positions)
    min_0,index_0 = t.min(diff_matrix,1)
    assert min_0.shape[0] != 0, 'Measure Error: No Min Distance'
    
    min_1,index_1 = t.min(min_0,0)
    min_index = index_1.item()
    hour_index = index_0[min_index].item()
    value = fine_points[hour_index][-1]
    mins = t.floor(value)
    seconds = (value - mins)*60
    return mins,seconds


def infer_measure(image,dets,save_img=False,fd=None,save_patches=False,save_thresh=False):
    fine_boxes,coarse_boxes,numeral_positions = post_process_boxes(dets,YDIM/2,CONF_THRESH)
    # extend_left,extend_right = verify_boxes(fine_boxes,coarse_boxes)
    coarse_segment, fine_segment, boundary_line = split_image(image,numeral_positions,save_thresh=save_thresh,fd=fd)
    
    coarse_line_positions,coarse_line_labels = get_coarse_line_positions(coarse_boxes)
    fine_line_positions,fine_line_labels = get_fine_line_positions(fine_boxes)

    coarse_lines = get_coarse_lines(coarse_segment,coarse_line_positions,coarse_line_labels,numeral_positions[1],save_patches=True,fd=fd)
    fine_lines = get_fine_lines(fine_segment,fine_line_positions,fine_line_labels,boundary_line[1],save_patches=save_patches,fd=fd)

    coarse_points_tmp = get_coarse_points(coarse_lines,boundary_line)
    fine_points_tmp = get_fine_points(fine_lines,boundary_line)

    coarse_points = label_coarse_points(coarse_points_tmp)
    fine_points = label_fine_points(fine_points_tmp)

    degrees,coarse_mins = measure_degree_mins(coarse_points,fine_points)
    fine_mins,seconds = measure_seconds(coarse_points,fine_points)
    print(coarse_line_positions,coarse_line_labels)
    if save_img:
        image = cv.cvtColor(image,cv.COLOR_GRAY2BGR)
        
        image = draw_boundary_line(image,boundary_line)    
        # image = draw_patch_boxes(image,coarse_boxes,fine_boxes,numeral_positions,boundary_line[1])     
        # image = draw_boxes(image,coarse_boxes)
        # image = draw_boxes(image,fine_boxes)
        # image = draw_vert_lines(image,coarse_lines,numeral_positions,col=(255,0,0))
        # image = draw_vert_lines(image,fine_lines,numeral_positions,col=(0,0,255))
        # image = draw_fine_points(image,fine_points)
        image = draw_coarse_points(image,coarse_points)
        # verify_points(t.clone(coarse_points),t.clone(fine_points))
        return [degrees,coarse_mins + fine_mins,seconds],image
    else:
        return [degrees,coarse_mins + fine_mins,seconds],image