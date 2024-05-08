import torch as t
from config.config import *

# def verify_dets(boxes)
def verify_points(coarse_points,fine_points):
    hours = coarse_points[:,2]
    mins = coarse_points[:,3]
    seconds = fine_points[:,2]
    
    hour_diff = hours[:-1] - hours[1:]
    min_diff = mins[:-1] - mins[1:]
    second_diff = seconds[:-1] - seconds[1:]
    
    coarse_x_positions = coarse_points[:,0]
    coarse_y_positions = coarse_points[:,1]
    
    fine_x_positions = fine_points[:,0]
    fine_y_positions = fine_points[:,1]
    
    coarse_x_diff = coarse_x_positions[1:] - coarse_x_positions[:-1]
    fine_x_diff = fine_x_positions[1:] - fine_x_positions[:-1]
    
    last_point = coarse_x_positions[-1]
    arrow_position = fine_x_positions[t.where(fine_points[:,2] == 0)[0]]
    
    # assert arrow_position < last_point, "Measurement Error: Arrow Beyond Labeled Points"
    # assert sum(coarse_x_diff < 0) == 0, f"Coarse Points Sequence Error: Coarse X Position Sequence Incorrect (Less)" 
    # assert sum(coarse_x_diff < MIN_COARSE_POINT_DISTANCE) == 0, f"Coarse Points Sequence Error: Coarse X Position Sequence Incorrect (Greater) {coarse_x_diff}" 
    
    # assert sum(sum(hour_diff == i for i in [1,0])) == len(hour_diff), f"Coarse Points Sequence Error: Hour Label Sequence Incorrect {hour_diff}"     
    # assert sum(sum(min_diff == i for i in [1,-59])) == len(min_diff), f"Coarse Points Sequence Error: Min Label Sequence Incorrect {mins}"
    
    # assert t.equal(seconds,EXPECTED_SECOND_SEQUENCE), f"Fine Points Sequence Error: Second Labels Do Not Match Expected Sequence {seconds}"
    
    
def verify_boxes(fine_boxes,coarse_boxes):
    
    arrow_ind = t.where(fine_boxes[:,5] == 4)[0]
    assert len(arrow_ind) == 1, 'Detection Error: Incorrect Number of Arrows Detected'
    
    sixty_ind = t.where(fine_boxes[:,5] == 2)[0]
    assert len(sixty_ind) == 1, 'Detection Error: Incorrect Number of Sixties Detected'
    img = np.zeros((XDIM,YDIM))


    
    for box_set in [fine_boxes,coarse_boxes]:
        for box in box_set:
            x1 = round(box[0].item())
            y1 = round(box[1].item())
            x2 = round(box[2].item())
            y2 = round(box[3].item())
            img[y1:y2,x1:x2] += 1

    assert np.max(img) == 1, 'Detection Error: Overlapping Boxes Detected'
    
    first_coarse_box = coarse_boxes[0]
    first_coarse_box_center = first_coarse_box[0] + (first_coarse_box[2] - first_coarse_box[0])/2

    last_coarse_box = coarse_boxes[-1]
    coarse_box_center = last_coarse_box[0] + (last_coarse_box[2] - last_coarse_box[0])/2

    arrow_box = fine_boxes[arrow_ind][0]
    arrow_center = arrow_box[0] + (arrow_box[2] - arrow_box[0])/2

    sixty_box = fine_boxes[sixty_ind][0]
    sixty_center = sixty_box[0] + (sixty_box[2] - sixty_box[0])/2
    
    if arrow_center > coarse_box_center:
        extend_right = True
    else:
        extend_right = False
        
    if sixty_center < first_coarse_box_center:
        extend_left = True
    else:
        extend_left = False    
        
    return extend_left,extend_right