import cv2
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import average_precision_score
<<<<<<< HEAD
=======

box_offset = 384
grid_cells = 7**2

>>>>>>> 336c5b43eb35ff46bf1b5b2c44e489a5201b979c

# BoundBox class
class BoundBox:
    def __init__(self, pc, xmin, ymin, xmax, ymax, c1, c2, c3):
        self.pc = pc
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

# Utilities to determine metrics and performance

# Calculate the output of testing the neural network
def calculate_metrics(truth, predicted):
    # Evaluation metric from A2
    m = average_precision_score(truth, predicted)
    return m
<<<<<<< HEAD
    #pass
=======
>>>>>>> 336c5b43eb35ff46bf1b5b2c44e489a5201b979c

# Interpret output of neural network
def interpret_output(response):
    # TODO
    pass

def check_box(y_vals, box_thresh):
    pass

# interval_a: box 1 (min, max)
# interval_b: box 2 (mix, max)
def overlap(interval_a, interval_b):
    amin, amax = interval_a
    bmin, bmax = interval_b

    if bmin < amin:
        # bmin < bmax < amin< amax, no overlap, return 0
        if bmax < amin:
            return 0
        else:
            # bmin < amin < amax or bmax, has overlap, return overlap value
            return min(amax,bmax) - amin
    else:
        # No overlap
        if amax < bmin:
            return 0
        else:
            return min(amax,bmax) - bmin

# Calculate intersection over union confidence scores for bounding boxes,
# TODO: Check threshold
# box1, box2: (7,7,8)
def iou(box1, box2):
    # TODO
    intersection_w = overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersection_h = overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect_area = intersection_w * intersection_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect_area
    iou_val = float(intersect_area) / union
    return iou_val

# y_vals: (1500,7,7,8)
#   pc,bx,by,bw,bh,c1,c2,c3
def draw_box(x, y_vals):
    for imgx,imgy in zip(x,y_vals):
        box_val = []
<<<<<<< HEAD
        for cell in img:
            for box_values in cell:
                xmin = int(box_values[1] - box_values[3] // 2)
                xmax = int(box_values[1] + box_values[3] // 2)
                ymin = int(box_values[2] - box_values[4] // 2)
                ymax = int(box_values[2] + box_values[4] // 2)
                c1 = box_values[5]
                c2 = box_values[6]
                c3 = box_values[7]
                box = BoundBox(box_values[0], xmin, ymin, xmax, ymax, c1, c2, c3)
                box_val.append(box)
                if box.pc != 0:
                    start_p = (box.xmin, box.ymin)
                    end_p = (box.xmax, box.ymax)

                    # Red color, Line thickness of 3 px , font scale is 1
                    color = (0,0,255)
                    thickness = 3
                    fontScale = 1

                    # Lable the box
                    label_str = ''
                    if box.c1 == 1:
                        label_str = 'circle'
                    if box.c2 == 1:
                        label_str = 'square'
                    if box.c3 == 1:
                        label_str = 'hexagon'
                    cv2.rectangle(cell, start_p, end_p, color, thickness)
                    cv2.putText(cell, label_str, (box.xmin, box.ymin - 13), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)
=======
        label_str = ''
        start_x = start_y = end_x = end_y = 0

        for i in range(grid_cells):
            xmin = int(imgy[i*8+1] - imgy[i*8+3] // 2)
            xmax = int(imgy[i*8+1] + imgy[i*8+3] // 2)
            ymin = int(imgy[i*8+2] - imgy[i*8+4] // 2)
            ymax = int(imgy[i*8+2] + imgy[i*8+4] // 2)
            c1 = imgy[i*8+5]
            c2 = imgy[i*8+6]
            c3 = imgy[i*8+7]
            box = BoundBox(imgy[i*8], xmin, ymin, xmax, ymax, c1, c2, c3)
            box_val.append(box)
            if box.pc != 0:
                # Lable the box
                if box.c1 == 1:
                    label_str = 'circle'
                if box.c2 == 1:
                    label_str = 'square'
                if box.c3 == 1:
                    label_str = 'hexagon'
                    
        for i in range(0,len(box_val)): 
            start_x += box_val[i].xmin
            start_y += box_val[i].ymin
            end_x += box_val[i].xmax
            end_y += box_val[i].ymax

        start_x = start_x // grid_cells
        start_y = start_y // grid_cells
        end_x = end_x // grid_cells
        end_y = end_y // grid_cells

        start = (start_x, start_y)
        end = (end_x+box_offset, end_y+box_offset)

        color = (0,0,255)
        thickness = 2
        fontScale = 0.7
        cv2.rectangle(imgx, start, end, color, thickness)
        cv2.putText(imgx, label_str, (start_x+15, start_y+30), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)
        cv2.imshow('image', imgx)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
>>>>>>> 336c5b43eb35ff46bf1b5b2c44e489a5201b979c
