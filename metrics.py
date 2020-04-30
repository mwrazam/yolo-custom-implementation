import cv2
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
# from sklearn.metrics import average_precision_score

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
    # m = average_precision_score(truth, predicted)
    # return m
    pass

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
        for cell in imgy:
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

