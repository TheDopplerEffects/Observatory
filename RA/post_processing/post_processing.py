import numpy as np
import torch as t
import cv2 as cv
import math

from post_processing.error_checking import *
from config.config import *
from post_processing.utils import *


def post_process_boxes(boxes, boundary, conf=0.6):
    """
    Post processing for detected bounding boxes
    :param boxes: Bounding Boxes -> tensor([x1,y1,x2,y2,score,cls_id])
    :param boundary: Detected Region Boundary
    :returns: Bounding boxes grouped based on region boundary and sorted in raster order, min and max y coordinate of fine and coarse boxes
    """
    assert boxes is not None, "Box Post Processing Error: No Boxes Detected"
    fine_boxes = None
    coarse_boxes = None
    # Filter Based on Region Boundary and Remove A or B Label
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i, box in enumerate(boxes):
        if (
            box[-1] not in [3, 5]
            and box[4].item() >= conf
            and box_areas[i] > MIN_BOX_SIZE
        ):  # Remove A and B Labels, Detections Below Confidence Threshold and Small Bounding Boxes
            if box[0] > XDIM * (DIST_FROM_EDGE) and box[2] < XDIM * (
                1 - DIST_FROM_EDGE
            ):  # Removes Boxes too close to edge which are likely to be missdetected
                if box[1] > boundary:
                    if fine_boxes is None:
                        fine_boxes = box.expand(1, 6)
                    else:
                        fine_boxes = t.cat((fine_boxes, box.expand(1, 6)))

                if box[1] <= boundary:
                    if coarse_boxes is None:
                        coarse_boxes = box.expand(1, 6)
                    else:
                        coarse_boxes = t.cat((coarse_boxes, box.expand(1, 6)))

    assert fine_boxes is not None, "No Fine Boxes Detected"
    assert coarse_boxes is not None, "No Coarse Boxes Detected"

    # Sort L to R
    sort = t.sort(fine_boxes[:, 0], descending=False)
    fine_boxes_sorted = fine_boxes[sort.indices]
    sort = t.sort(coarse_boxes[:, 0], descending=False)
    coarse_boxes_sorted = coarse_boxes[sort.indices]

    # Get Numeral Lines Locations
    fine_y_position = t.min(fine_boxes_sorted[:, 1])
    coarse_y_position = t.max(coarse_boxes_sorted[:, 3])

    assert (
        fine_boxes_sorted is not None
    ), "Dectection Grouping Error: Sorted Fine Boxes Emtpy"
    assert (
        coarse_boxes_sorted is not None
    ), "Dectection Grouping Error: Sorted Coarse Boxes Emtpy"

    return fine_boxes_sorted, coarse_boxes_sorted, (fine_y_position, coarse_y_position)


def split_image(img, numeral_positions, save_thresh=False, fd=None):
    """
    Detecteds image center line using HoughLine Algorithm
    :param img: Grayscale image of scale
    :param numeral_positions: Average locations of numerals
    :returns: Bounding Boxes Grouped Based on Region Boundary and Sorted in Raster Order and Average X and Y Positions of the Groups
    """
    # Crop Image Between Detected Objects to Reduce size of HoughLine image
    fine_pos = int(numeral_positions[0])
    coarse_pos = int(numeral_positions[1])
    cropped_img = img[coarse_pos:fine_pos, :]

    # Filter, Threshold and Houghlines
    edge_map = cv.Sobel(cropped_img, cv.CV_8U, 0, 1, ksize=VS_KERNEL)
    retval, yThresh = cv.threshold(edge_map, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    lines = cv.HoughLines(
        yThresh,
        SEGMENT_RHO_RES,
        SEGMENT_THETA_RES,
        SEGMENT_VOTE_RES,
        None,
        0,
        0,
        np.pi / 2 - YTOL,
        np.pi / 2 + YTOL,
    )

    # Optional Image Saving for Debugging
    if save_thresh:
        cv.imwrite(f"images/tresh_{fd}", yThresh)

    # Average Detected Lines Together
    theta_offset = np.mean(lines[:, 0, 1])
    rho_offset = np.mean(lines[:, 0, 0])

    # Convert Line to Slope-Intercept Form
    m = -(math.cos(theta_offset) / math.sin(theta_offset))
    b = rho_offset / math.sin(theta_offset)
    center = round(img.shape[1] // 2 * m + b)

    boundary_edge = round(center + coarse_pos)
    boundary_line = [m, b + coarse_pos]

    return img[coarse_pos:boundary_edge], img[boundary_edge:fine_pos], boundary_line


def is_roman(cls_id):
    return class_names[cls_id] in ROMAN_LUT


def get_coarse_line_positions(boxes, image_dims):
    """
    Determines tick line positions and labels using coarse bounding boxes
    :param boxes: coarse scale bounding boxes, [x1,y1,x2,y2,score,cls_id]
    :param image_dims: original grayscale image dimensions
    :returns: approximate tick positions and labels -> tensor([x]), tensor([hours,mins])
    """
    center = (round(image_dims[1] / 2), round(image_dims[0] / 2))
    roman = None
    # Find Roman Numeral Closest to center of image
    for box in boxes:
        if is_roman(int(box[5].item())):
            if roman is None:
                roman = box
            else:
                distance_1 = abs(center[0] - (roman[0] + (roman[2] - roman[0]) / 2))
                distance_2 = abs(center[0] - (box[0] + (box[2] - box[0]) / 2))
                if distance_2 > distance_1:
                    roman = box

    assert roman is not None, "Detection Error: No Roman Numeral Found"

    hour_label = NUMERAL_LUT[class_names[int(roman[5].item())]]
    hour_position = roman[0] + (roman[2] - roman[0]) / 2

    line_labels = None
    line_positions = None

    # Calculate and label potential tick locations given bouding boxes
    for i in range(len(boxes)):
        if i != len(boxes) - 1:
            center = boxes[i, 0] + (boxes[i, 2] - boxes[i, 0]) / 2
            next_center = boxes[i + 1, 0] + (boxes[i + 1, 2] - boxes[i + 1, 0]) / 2
            difference = next_center - center

            if center > hour_position:
                if hour_label == 0:
                    hour = 23
                else:
                    hour = hour_label - 1
                hour_calc = hour
            elif center == hour_position:
                hour = hour_label
                if hour_label == 0:
                    hour_calc = 23
                else:
                    hour_calc = hour_label - 1
            else:
                hour = hour_label
                hour_calc = hour

            if is_roman(int(boxes[i][5].item())):
                min = 0
                min_calc = 60
            else:
                min = NUMERAL_LUT[class_names[int(boxes[i][5].item())]]
                min_calc = min

            if line_positions is None:
                line_positions = t.tensor([center])
            else:
                line_positions = t.cat((line_positions, t.tensor([center])), axis=0)

            if line_labels is None:
                line_labels = t.tensor([[hour, min]])
            else:
                line_labels = t.cat((line_labels, t.tensor([[hour, min]])), axis=0)

            for j in range(1, 4):
                line_positions = t.cat(
                    (line_positions, t.tensor([center + j * difference / 4])), axis=0
                )
                line_labels = t.cat(
                    (line_labels, t.tensor([[hour_calc, min_calc - 5 * j]])), axis=0
                )
        else:
            center = boxes[i, 0] + (boxes[i, 2] - boxes[i, 0]) / 2
            if center > hour_position:
                if hour_label == 0:
                    hour = 23
                else:
                    hour = hour_label - 1
            else:
                hour = hour_label

            if is_roman(int(boxes[i][5].item())):
                min = 0
            else:
                min = NUMERAL_LUT[class_names[int(boxes[i][5].item())]]

            line_positions = t.cat((line_positions, t.tensor([center])), axis=0)
            line_labels = t.cat((line_labels, t.tensor([[hour, min]])), axis=0)

    assert line_positions is not None, "Coarse Line Position Error: No Line Position"
    assert line_labels is not None, "Coarse Line Position Error: No Line Labels"

    return line_positions, line_labels


def get_coarse_lines(
    coarse_segment, line_positions, line_labels, position, save_patches=False, fd=None
):
    """
    Detects tick lines on cropped image patches using Hough Transform
    :param coarse_segment: cropped image region of coarse Scale
    :param line_positions: X positions of ticks
    :param line_labels: hour and minute label of ticks
    :position: Y axis coordinate of coarse segment for intercept correct
    :returns: Detected tick locations in slope-intercept form -> tensor([m,b,hour,min])
    """
    # Use Approx Line Positions to Slice Image and Use Hough to Detect
    coarse_lines = None
    vert_edges = cv.Sobel(coarse_segment, cv.CV_8U, 1, 0, ksize=VS_KERNEL)
    _, threshold = cv.threshold(vert_edges, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    for i, line in enumerate(line_positions):
        patch = threshold[:, int(line) - PATCH_WIDTH : int(line) + PATCH_WIDTH]

        # Rotate Patch 90deg to avoid limit as angle approaches zero
        patch_rotated = cv.rotate(patch, cv.ROTATE_90_CLOCKWISE)
        lines = cv.HoughLines(
            patch_rotated,
            COARSE_RHO_RES,
            COARSE_THETA_RES,
            COARSE_VOTE_THRESH,
            None,
            0,
            0,
            np.pi / 2 - XTOL,
            np.pi / 2 + XTOL,
        )
        if lines is not None:
            # Average Detected Lines Together
            theta_offset = np.mean(lines[:, 0, 1])
            rho_offset = np.mean(lines[:, 0, 0])

            # Rotate Line by 90 CC (x,y) -> (-y,x)     y = -(cos(t)/sin(t)) x + (rho/sin(t)) -> y = (sin(t)/cos(t)) x - (rho/cos(t))
            m2 = np.sin(theta_offset) / np.cos(theta_offset)
            b2 = -rho_offset / np.cos(theta_offset)

            # Offset intercept by height of patch
            b2 += patch.shape[0]

            # Offset of new origin
            dX = -(int(line.item()) - PATCH_WIDTH)
            dY = -position

            # New intercept after translation
            b3 = b2 - dY + m2 * dX

            if coarse_lines is None:
                coarse_lines = t.tensor(
                    [[m2, b3, line_labels[i][0], line_labels[i][1]]]
                )
            else:
                coarse_lines = t.cat(
                    (
                        coarse_lines,
                        t.tensor([[m2, b3, line_labels[i][0], line_labels[i][1]]]),
                    )
                )

            # Optional Patch saving for Debugging
            if save_patches:
                patch = cv.cvtColor(patch, cv.COLOR_GRAY2BGR)
                patch_rotated = cv.cvtColor(patch_rotated, cv.COLOR_GRAY2BGR)
                xt = 500
                pt1 = (-xt, round(m2 * -xt + b2))
                pt2 = (round(xt), round(m2 * xt + b2))
                cv.line(patch, pt1, pt2, (0, 0, 255), 1)
                cv.imwrite(f"images/coarse_patch_{line_labels[i][0]}_{fd}", patch)

    assert coarse_lines is not None, "Coarse Line Detection Error: No Lines Detected"

    return coarse_lines


def get_fine_line_positions(boxes):
    """
    Determines tick line positions and labels using fine bounding Boxes
    :param boxes: fine scale bounding boxes, tensor([x1,y1,x2,y2,score,cls_id])
    :returns: Approximate quarter tick positions and labels -> tensor([x]), tensor([seconds])
    """
    box_ids = boxes[:, 5].type(t.int)

    line_labels = None
    line_positions = None

    # Calculate and label potential tick locations given bouding boxes
    for i in range(len(boxes)):
        center = boxes[i][0] + (boxes[i][2] - boxes[i][0]) / 2
        second = NUMERAL_LUT[class_names[box_ids[i]]]
        for j in range(2):
            if j == 0:
                if line_positions is None:
                    line_positions = t.tensor([center])
                else:
                    line_positions = t.cat((line_positions, t.tensor([center])))

                if line_labels is None:
                    line_labels = t.tensor([second])
                else:
                    line_labels = t.cat((line_labels, t.tensor([second])), axis=0)
            else:
                if i < len(boxes) - 1:
                    next_box = boxes[i + 1]
                    next_center = next_box[0] + (next_box[2] - next_box[0]) / 2

                    midpoint = center + (next_center - center) / 2
                    if line_positions is None:
                        line_positions = t.tensor([midpoint])
                    else:
                        line_positions = t.cat((line_positions, t.tensor([midpoint])))

                    if line_labels is None:
                        line_labels = t.tensor([second - 10])
                    else:
                        line_labels = t.cat(
                            (line_labels, t.tensor([second - 10])), axis=0
                        )

    assert line_positions is not None, "Fine Line Position Error: No Line Position"
    assert line_labels is not None, "Fine Line Position Error: No Line Labels"

    return line_positions, line_labels


def get_fine_lines(
    fine_segment, line_positions, line_labels, position, save_patches=False, fd=None
):
    """
    Detects quarter tick lines on cropped image patches using Hough Transform
    :param coarse_segment: Cropped Image Region of Fine Scale
    :param line_positions: X positions of half ticks
    :param line_labels: Hour and Minute label of quarter ticks
    :position: Y axis coordinate of coarse segment for intercept correct
    :returns: Detected quarter tick locations in slope-intercept form -> tensor([m,b,hour,min])
    """
    fine_lines = None
    vert_edges = cv.Sobel(fine_segment, cv.CV_8U, 1, 0, ksize=VS_KERNEL)
    _, threshold = cv.threshold(vert_edges, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    for i, line in enumerate(line_positions):
        patch = threshold[:, int(line) - PATCH_WIDTH : int(line) + PATCH_WIDTH]

        # Rotate Patch 90deg to avoid limit as angle approaches zero
        patch_rotated = cv.rotate(patch, cv.ROTATE_90_CLOCKWISE)

        lines = cv.HoughLines(
            patch_rotated,
            FINE_RHO_RES,
            FINE_THETA_RES,
            FINE_VOTE_THRESH,
            None,
            0,
            0,
            np.pi / 2 - XTOL,
            np.pi / 2 + XTOL,
        )

        if lines is not None:
            theta_offset = np.mean(lines[:, :, 1])
            rho_offset = np.mean(lines[:, :, 0])
            # Rotate Line by 90 CC (x,y) -> (-y,x),  y = -(cos(t)/sin(t)) x + (rho/sin(t)) -> y = (sin(t)/cos(t)) x - (rho/cos(t))
            m2 = np.sin(theta_offset) / np.cos(theta_offset)
            b2 = -rho_offset / np.cos(theta_offset)

            # Offset intercept by height of patch
            b2 += patch.shape[0]

            # Offset of new origin
            dX = -(int(line.item()) - PATCH_WIDTH)
            dY = -position

            # New intercept after translation
            b3 = b2 - dY + m2 * dX

            if fine_lines is None:
                fine_lines = t.tensor([[m2, b3, line_labels[i]]])
            else:
                fine_lines = t.cat((fine_lines, t.tensor([[m2, b3, line_labels[i]]])))

            # Optional Patch saving for Debugging
            if save_patches:
                patch = cv.cvtColor(patch, cv.COLOR_GRAY2BGR)
                patch_rotated = cv.cvtColor(patch_rotated, cv.COLOR_GRAY2BGR)

                dim = patch.shape
                xt = 5000
                pt1 = (-xt, round(m2 * -xt + b2))
                pt2 = (round(xt), round(m2 * xt + b2))
                cv.line(patch, pt1, pt2, (0, 0, 255), 1)
                cv.imwrite(f"images/fine_patch_{line_labels[i]}_{fd}", patch)

    assert fine_lines is not None, "Fine Line Detection Error: No Lines Detected"

    return fine_lines


def get_coarse_points(lines, boundary_line):
    """
    Calculate intersection of detected coarse lines and image boundary line
    A_1*x + y + C1 = 0, A_1 = m1, C1 = b1
    A_2*x + y + C1 = 0, A_2 = m2, C2 = b2
    (x,y) = ((C2 -C1)/(A1 - A2),-(C1*A2 - C2*A1)/(A1 - A2))
    :param lines: Detected coarse lines in slope-intercept form
    :param boundary_line: Image boundary line in slope-intercept form
    :returns: Intersection coordinates of each coarse line with boundary -> tensor([x,y,hour,minute])
    """
    m1 = boundary_line[0]
    b1 = boundary_line[1]
    coarse_points = None
    for pt in lines:
        m2 = pt[0].item()
        b2 = pt[1].item()

        x0 = (b2 - b1) / (m1 - m2)
        y0 = -(b1 * m2 - b2 * m1) / (m1 - m2)

        if coarse_points is None:
            coarse_points = t.tensor([[x0, y0, pt[2], pt[3]]])
        else:
            coarse_points = t.cat((coarse_points, t.tensor([[x0, y0, pt[2], pt[3]]])))
    return coarse_points


def get_fine_points(lines, boundary_line):
    """
    Calculate intersection of detected fine lines and image boundary line
    A_1*x + y + C1 = 0, A_1 = m, C1 = b
    A_2*x + y + C1 = 0, A_2 = m, C2 = b
    (x_i,y_i) = ((C2 -C1)/(A1 - A2),-(C1*A2 - C2*A1)/(A1 - A2))
    :param lines: Detected fine lines in slope-intercept form
    :param boundary_line: Image boundary line in slope-intercept form
    :returns: Intersection coordinates of each coarse line with boundary -> tensor([x,y,seconds])
    """
    m1 = boundary_line[0]
    b1 = boundary_line[1]
    fine_points = None
    for pt in lines:
        m2 = pt[0].item()
        b2 = pt[1].item()

        x0 = (b2 - b1) / (m1 - m2)
        y0 = -(b1 * m2 - b2 * m1) / (m1 - m2)

        if fine_points is None:
            fine_points = t.tensor([[x0, y0, pt[2]]])
        else:
            fine_points = t.cat((fine_points, t.tensor([[x0, y0, pt[2]]])))
    return fine_points


def label_coarse_points(lines, extend_left=False, extend_right=False):
    """
    Calculates intermediate tick locations based on spacing of detected coarse line
    :params line: Detected tick intersection locations
    :params optional extend_left/right: extends intermediate point locations outside detected region when first/last detection within fine scale locations
    :returns: Intermediate tick locations -> tensor([x,y,hour,minute])
    """
    points_out = None
    for i, line in enumerate(lines[:-1]):
        current_hour = lines[i][2]
        current_min = lines[i][3]

        next_hour = lines[i + 1][2]
        next_min = lines[i + 1][3]
        x = lines[i][0].item()
        y = lines[i][1].item()

        x2 = lines[i + 1][0].item()
        y2 = lines[i + 1][1].item()

        hour = current_hour.item()
        minute = current_min.item()

        # No Hour change between current and next point
        if current_hour == next_hour:
            difference = current_min - next_min
            n = int(difference.item() - 1)

            dX = (x2 - x) / difference
            dY = (y2 - y) / difference

            if points_out is None:
                points_out = t.tensor([[x, y, hour, minute]])
            else:
                points_out = t.cat((points_out, t.tensor([[x, y, hour, minute]])))

            for j in range(n):
                y += dY
                x += dX
                points_out = t.cat(
                    (points_out, t.tensor([[x, y, hour, minute - j - 1]]))
                )

        # Current location at Roman Numeral
        elif current_min == 0:
            difference = 60 - next_min
            n = int(difference.item() - 1)
            dX = (x2 - x) / difference
            dY = (y2 - y) / difference

            if points_out is None:
                points_out = t.tensor([[x, y, hour, minute]])
            else:
                points_out = t.cat((points_out, t.tensor([[x, y, hour, minute]])))

            # Handle Hour Rollover between points
            if hour == 0:
                hour = 23
            else:
                hour -= 1

            minute = 60

            for j in range(n):
                y += dY
                x += dX
                points_out = t.cat(
                    (points_out, t.tensor([[x, y, hour, minute - j - 1]]))
                )
        else:
            difference = int(current_min.item())
            div1 = difference
            n1 = difference - 1

            difference = 60 - int(next_min.item())
            div2 = difference

            n2 = difference - 1
            dX = (x2 - x) / (div1 + div2)
            dY = (y2 - y) / (div1 + div2)

            if points_out is None:
                points_out = t.tensor([[x, y, hour, minute]])
            else:
                points_out = t.cat((points_out, t.tensor([[x, y, hour, minute]])))

            for i in range(n1):
                y += dY
                x += dX
                t.cat((points_out, t.tensor([[x, y, hour, minute - i - 1]])))

            if points_out is None:
                points_out = t.tensor([[x, y, hour, minute]])
            else:
                points_out = t.cat((points_out, t.tensor([[x, y, hour, minute]])))

            # Handle Hour Rollover between points
            if hour == 1:
                hour = 24
            else:
                hour -= 1
            minute = 60
            for j in range(n2):
                y += dY
                x += dX
                points_out = t.cat(
                    (points_out, t.tensor([[x, y, hour, minute - j - 1]]))
                )

        # Point Extrapolation using first box line spacing
        if extend_left and i == 0:

            x = lines[0][0]
            y = lines[0][1]

            hour = current_hour.item()
            minute = current_min.item()
            if minute == 55:
                for j in range(1, 5):
                    y -= dY
                    x -= dX
                    points_out = t.cat(
                        (t.tensor([[x, y, hour, minute + j]]), points_out)
                    )
                y -= dY
                x -= dX
                points_out = t.cat((t.tensor([[x, y, hour, minute + j]]), points_out))
            else:
                for j in range(1, 6):
                    y -= dY
                    x -= dX
                    points_out = t.cat(
                        (t.tensor([[x, y, hour, minute + j]]), points_out)
                    )

    x = lines[-1][0]
    y = lines[-1][1]
    hour = lines[-1][2]
    minute = lines[-1][3]
    points_out = t.cat((points_out, t.tensor([[x, y, hour, minute]])))

    # Point Extrapolation using last box line spacing
    if extend_right:
        if minute == 0:
            minute = 60
            if hour == 0:
                hour = 23
            else:
                hour -= 1
        for j in range(1, 6):
            x += dX
            y += dY
            points_out = t.cat((points_out, t.tensor([[x, y, hour, minute - j]])))

    return points_out


def label_fine_points(points):
    """
    Calculates intermediate tick locations based on spacing of detected fine line
    :params line: Detected tick intersection locations
    :returns: Intermediate tick locations -> tensor([x,y,seconds])
    """
    points_out = None
    for i in range(len(points) - 1):
        diff = int((points[i][2] - points[i + 1][2]).item())
        n = diff // 2 - 1
        x = points[i][0]
        x_1 = points[i + 1][0]
        y = points[i][1]
        y_1 = points[i + 1][1]

        dY = (y_1 - y) / (diff / 2)
        dX = (x_1 - x) / (diff / 2)
        # Calculate extra fine tick location
        if points[i][2] == 60:
            x_tra = x - dX
            y_tra = y - dY
            if points_out is None:
                points_out = t.tensor([[x_tra, y_tra, 61]])
            else:
                points_out = t.cat((points_out, t.tensor([[x_tra, y_tra, 61]])))

        if points_out is None:
            points_out = t.tensor([[x, y, points[i][2]]])
        else:
            points_out = t.cat((points_out, t.tensor([[x, y, points[i][2]]])))
        for j in range(n):
            y += dY
            x += dX
            points_out = t.cat(
                (points_out, t.tensor([[x, y, points[i][2] - 2 * j - 2]]))
            )

    # Calculate extra fine tick location
    points_out = t.cat((points_out, t.tensor([[points[-1][0], points[-1][1], 0]])))
    points_out = t.cat(
        (points_out, t.tensor([[points[-1][0] + dX, points[-1][1] + dY, -2]]))
    )
    return points_out


def measure_hours_mins(coarse_points, fine_points):
    """
    Determines coarse scale measurement, finds closest point to arrow
    :param coarse_points: Detected coarse points
    :param fine_points: Detected fine points
    :returns: Measurment of coarse scale -> hours,mins
    """
    arrow_point = fine_points[-2][0]
    k = 0
    if len(coarse_points) > 1:
        while (
            round(arrow_point.item() - coarse_points[k][0].item()) > 0
            and k < len(coarse_points) - 1
        ):
            k += 1
            point = coarse_points[k]
        hrs, mins = point[2], point[3]
    else:
        hrs, mins = (-1, -1)
    return hrs, mins


def measure_seconds(coarse_points, fine_points):
    """
    Determines fine scale measurement, calculates difference matrix between all coarse and fine points
    :param coarse_points: Detected coarse points
    :param fine_points: Detected fine points
    :returns: Measurment of fine scale -> seconds
    """
    y_low = fine_points[0][0]
    y_high = fine_points[-1][0]

    coarse_low = 0
    coarse_high = 0
    # Filter out coarse points past 60 and Arrow positions on fine scale
    for i, point in enumerate(coarse_points):
        if point[0] < y_low:
            coarse_low = i

        if point[0] > y_high:
            coarse_high = i
            break
    coarse_slice = coarse_points[coarse_low:coarse_high]

    coarse_x_positions = coarse_points[:, 1]
    fine_x_positions = fine_points[:, 1]

    # Calculate Difference Matrix for coarse and fine x positions
    diff_matrix = t.abs(coarse_x_positions.unsqueeze(1) - fine_x_positions)
    min_0, index_0 = t.min(diff_matrix, 1)
    assert min_0.shape[0] != 0, "Measure Error: No Min Distance"

    # Take Minimum of difference matrix to get seconds measurement
    min_1, index_1 = t.min(min_0, 0)
    min_index = index_1.item()
    hour_index = index_0[min_index].item()
    return fine_points[hour_index][-1]


def infer_measure(
    image, dets, save_img=False, fd=None, save_patches=False, save_thresh=False
):
    """
    Determines Measurement from input image
    :param dets: Detections output from model -> tensor([x1,y1,x2,y2,score,cls_id])
    :param optional save_img: Debugging option to save image
    :param optional fd: Filename of current image
    :param save_patches: Debugging option to save image patches
    :param save_thresh: Debeggin option to save thresholded image
    :returns: Detected image measurement and annotated image -> [hours,minutes,seconds],image
    """
    fine_boxes, coarse_boxes, numeral_positions = post_process_boxes(
        dets, YDIM / 2, CONF_THRESH
    )

    extend_left, extend_right = verify_boxes(fine_boxes, coarse_boxes)

    coarse_segment, fine_segment, boundary_line = split_image(
        image, numeral_positions, save_thresh=save_thresh, fd=fd
    )

    coarse_line_positions, coarse_line_labels = get_coarse_line_positions(
        coarse_boxes, image.shape
    )
    fine_line_positions, fine_line_labels = get_fine_line_positions(fine_boxes)

    coarse_lines = get_coarse_lines(
        coarse_segment,
        coarse_line_positions,
        coarse_line_labels,
        numeral_positions[1],
        save_patches=save_patches,
        fd=fd,
    )
    fine_lines = get_fine_lines(
        fine_segment,
        fine_line_positions,
        fine_line_labels,
        boundary_line[1],
        save_patches=save_patches,
        fd=fd,
    )

    coarse_points_tmp = get_coarse_points(coarse_lines, boundary_line)
    fine_points_tmp = get_fine_points(fine_lines, boundary_line)

    coarse_points = label_coarse_points(coarse_points_tmp, True, True)
    fine_points = label_fine_points(fine_points_tmp)

    hours, mins = measure_hours_mins(coarse_points, fine_points)
    seconds = measure_seconds(coarse_points, fine_points)

    if seconds == 61:
        seconds = t.tensor(0)
    elif seconds == -2:
        seconds = t.tensor(0)
    elif seconds == 60:
        seconds = t.tensor(0)

    # Optional param for debuggin
    if save_img:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        image = draw_boundary_line(image, boundary_line)
        # image = draw_patch_boxes(image,coarse_boxes,fine_boxes,numeral_positions,boundary_line[1])
        image = draw_boxes(image, coarse_boxes)
        image = draw_boxes(image, fine_boxes)
        # image = draw_vert_lines(image,coarse_lines,numeral_positions,col=(255,0,0))
        # image = draw_vert_lines(image,fine_lines,numeral_positions,col=(0,0,255))
        image = draw_fine_points(image, fine_points)
        image = draw_coarse_points(image, coarse_points)
        return [hours, mins, seconds], image
    else:
        return [hours, mins, seconds], image
