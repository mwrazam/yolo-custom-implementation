import cv2
import math
import numpy as np
import tensorflow as tf
from keras import backend as K

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
    # TODO
    pass

# Interpret output of neural network
def interpret_output(response):
    # TODO
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

# Calculate intersection over union confidence scores for bounding boxes
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
    print("IOU Values is:")
    print(iou_val)
    return iou_val

# y_vals: (1500,7,7,8)
#   pc,bx,by,bw,bh,c1,c2,c3
def draw_box(y_vals):
    for img in y_vals:
        box_val = []
        for cell in img:
            # for box in cell:
            print(cell)
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
                    # cv2.rectangle(cell, start_p, end_p, color, thickness)
                    # cv2.putText(cell, label_str, (box.xmin, box.ymin - 13), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)

            ###################
            # Test iou function
            # iou(box_val[0], box_val[1])

# y_true, y_pred: (1500,7,7,8)
# 8: pc,bx,by,bw,bh,c1,c2,c3
def yolo_loss(y_true, y_pred):
    gridcells = 7
    box_per_cell = 1

    # Each bounding box consists of 5 predictions: x, y, w, h, and confidence
    box_pred_per_cell = 5

    # B * 5 + C
    total_pred_per_cell = box_per_cell * box_pred_per_cell
    final_pred_cell = total_pred_per_cell + 3

    lamda_coord = 5
    lamda_noobj = 0.5
    truth_num = final_pred_cell

    # truth table format is [[confid,x,y,w,h]..,classes] for one cell
    totloss =0
    for i in range(y_true.shape[0]):
        
        yt = y_true[i,:,:,:].flatten()
        yp = y_pred[i,:,:,:].flatten()
        # print(yt)
        for cell in range(gridcells**2):
            cell_loss = 0
            pcloss = 0
            xyloss = 0
            whloss = 0
            closs = 0

            # pc loss
            pcloss += (yt[cell*truth_num] - yp[cell*truth_num])**2

            # bx and by loss: sum [(x_t - x_p)^2 + (y_t - y_p)^2]
            xyloss += (yt[cell*truth_num+1] - yp[cell*truth_num+1]) ** 2 + (yt[cell*truth_num+2] - yp[cell*truth_num+2]) ** 2

            # width and height loss: sum [(root(w_t) - root(w_p))^2 + (root(h_t) - root(h_p))^2]
            whloss += (math.sqrt(yt[cell*truth_num+3]) - math.sqrt(yp[cell*truth_num+3]))**2 + (math.sqrt(yt[cell*truth_num+4]) - math.sqrt(yp[cell*truth_num+4]))**2
            
            # Class loss
            closs += (yt[cell*truth_num+5] - yp[cell*truth_num+5])**2 + (yt[cell*truth_num+6] - yp[cell*truth_num+6])**2 + (yt[cell*truth_num+7] - yp[cell*truth_num+7])**2

            # cell has an object
            if yt[cell*truth_num] == 1:
                sumpcloss = pcloss
                sumxyloss = lamda_coord * xyloss
                sumwhloss = lamda_coord * whloss
                sumcloss = closs
            else:
                sumpcloss = 0
                sumxyloss = 0
                sumwhloss = 0
                sumcloss = lamda_noobj * closs
            cell_loss = sumpcloss + sumxyloss + sumwhloss + sumcloss
            totloss += cell_loss
            
    return totloss

################################################################
# Testing draw box and iou functions, values are from y_val[0] #
test_arr1 = np.array([
            [[[0,0,0,0,0,1,0,0], [1,61,63,4,0,1,0,0], [1,51,46,23,33,1,0,0], [1,35,51,55,23,1,0,0], [1,35,31,55,63,1,0,0], [1,43,27,39,55,1,0,0], [0,0,0,0,0,1,0,0]],
            [[0,0,0,0,0,1,0,0], [1,31,61,63,4,1,0,0], [1,31,11,63,23,1,0,0], [1,19,8,39,16,1,0,0], [1,11,31,23,63,1,0,0], [1,31,31,63,63,1,0,0], [1,35,11,55,23,1,0,0]],
            [[1,35,61,55,3,1,0,0], [1,31,23,63,32,1,0,0], [1,11,0,23,1,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,31,48,63,30,1,0,0], [1,31,11,63,23,1,0,0]],
            [[1,35,51,63,23,1,0,0], [1,35,11,63,23,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,31,51,63,23,1,0,0], [1,31,4,63,8,1,0,0]],
            [[1,31,51,63,23,1,0,0], [1,31,19,63,39,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,44,62,38,1,1,0,0], [1,31,32,63,45,1,0,0], [1,19,1,39,3,1,0,0]],
            [[1,27,51,54,23,1,0,0], [1,31,31,63,63,1,0,0], [1,43,31,39,63,1,0,0], [1,59,52,7,21,1,0,0], [1,31,51,63,23,1,0,0], [1,31,3,63,7,1,0,0], [0,0,0,0,0,1,0,0]],
            [[0,0,0,0,0,1,0,0], [1,19,36,39,54,1,0,0], [1,27,31,55,63,1,0,0], [1,27,31,55,63,1,0,0], [1,27,2,55,4,1,0,0], [1,10,0,21,0,1,0,0], [0,0,0,0,0,1,0,0]]],
            [[[0,0,0,0,0,1,0,0], [1,61,63,4,0,1,0,0], [1,51,46,23,33,1,0,0], [1,35,51,55,23,1,0,0], [1,35,31,55,63,1,0,0], [1,43,27,39,55,1,0,0], [0,0,0,0,0,1,0,0]],
            [[0,0,0,0,0,1,0,0], [1,31,61,63,4,1,0,0], [1,31,11,63,23,1,0,0], [1,19,8,39,16,1,0,0], [1,11,31,23,63,1,0,0], [1,31,31,63,63,1,0,0], [1,35,11,55,23,1,0,0]],
            [[1,35,61,55,3,1,0,0], [1,31,23,63,32,1,0,0], [1,11,0,23,1,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,31,48,63,30,1,0,0], [1,31,11,63,23,1,0,0]],
            [[1,35,51,63,23,1,0,0], [1,35,11,63,23,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,31,51,63,23,1,0,0], [1,31,4,63,8,1,0,0]],
            [[1,31,51,63,23,1,0,0], [1,31,19,63,39,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,44,62,38,1,1,0,0], [1,31,32,63,45,1,0,0], [1,19,1,39,3,1,0,0]],
            [[1,27,51,54,23,1,0,0], [1,31,31,63,63,1,0,0], [1,43,31,39,63,1,0,0], [1,59,52,7,21,1,0,0], [1,31,51,63,23,1,0,0], [1,31,3,63,7,1,0,0], [0,0,0,0,0,1,0,0]],
            [[0,0,0,0,0,1,0,0], [1,19,36,39,54,1,0,0], [1,27,31,55,63,1,0,0], [1,27,31,55,63,1,0,0], [1,27,2,55,4,1,0,0], [1,10,0,21,0,1,0,0], [0,0,0,0,0,1,0,0]]]
            ])

test_arr2 = np.array([
            [[[0,0,0,0,0,1,0,0], [1,61,63,4,0,1,0,0], [1,51,46,23,33,1,0,0], [1,35,51,55,23,1,0,0], [1,35,31,55,63,1,0,0], [1,43,27,39,55,1,0,0], [0,0,0,0,0,1,0,0]],
            [[0,0,0,0,0,1,0,0], [1,31,61,63,4,1,0,0], [1,31,11,63,23,1,0,0], [1,19,8,39,16,1,0,0], [1,11,31,23,63,1,0,0], [1,31,31,63,63,1,0,0], [1,35,11,55,23,1,0,0]],
            [[1,35,61,55,3,1,0,0], [1,31,23,63,32,1,0,0], [1,11,0,23,1,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,31,48,63,30,1,0,0], [1,31,11,63,23,1,0,0]],
            [[1,35,51,63,23,1,0,0], [1,35,11,63,23,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,31,51,63,23,1,0,0], [1,31,4,63,8,1,0,0]],
            [[1,31,51,63,23,1,0,0], [1,31,19,63,39,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,44,62,38,1,1,0,0], [1,31,32,63,45,1,0,0], [1,19,1,39,3,1,0,0]],
            [[0,0,0,0,0,1,0,0], [1,31,61,63,4,1,0,0], [1,31,11,63,23,1,0,0], [1,19,8,39,16,1,0,0], [1,11,31,23,63,1,0,0], [1,31,31,63,63,1,0,0], [1,35,11,55,23,1,0,0]],
            [[0,0,0,0,0,1,0,0], [1,19,36,39,54,1,0,0], [1,27,31,55,63,1,0,0], [1,27,31,55,63,1,0,0], [1,27,2,55,4,1,0,0], [1,10,0,21,0,1,0,0], [0,0,0,0,0,1,0,0]]],
            [[[0,0,0,0,0,1,0,0], [1,61,63,4,0,1,0,0], [1,51,46,23,33,1,0,0], [1,35,51,55,23,1,0,0], [1,35,31,55,63,1,0,0], [1,43,27,39,55,1,0,0], [0,0,0,0,0,1,0,0]],
            [[0,0,0,0,0,1,0,0], [1,31,61,63,4,1,0,0], [1,31,11,63,23,1,0,0], [1,19,8,39,16,1,0,0], [1,11,31,23,63,1,0,0], [1,31,31,63,63,1,0,0], [1,35,11,55,23,1,0,0]],
            [[1,35,61,55,3,1,0,0], [1,31,23,63,32,1,0,0], [1,11,0,23,1,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,31,48,63,30,1,0,0], [1,31,11,63,23,1,0,0]],
            [[1,35,61,55,3,1,0,0], [1,31,23,63,32,1,0,0], [1,11,0,23,1,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,31,48,63,30,1,0,0], [1,31,11,63,23,1,0,0]],
            [[1,31,51,63,23,1,0,0], [1,31,19,63,39,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,44,62,38,1,1,0,0], [1,31,32,63,45,1,0,0], [1,19,1,39,3,1,0,0]],
            [[1,27,51,54,23,1,0,0], [1,31,31,63,63,1,0,0], [1,43,31,39,63,1,0,0], [1,59,52,7,21,1,0,0], [1,31,51,63,23,1,0,0], [1,31,3,63,7,1,0,0], [0,0,0,0,0,1,0,0]],
            [[1,31,51,63,23,1,0,0], [1,31,19,63,39,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [1,44,62,38,1,1,0,0], [1,31,32,63,45,1,0,0], [1,19,1,39,3,1,0,0]]]
            ])
# draw_box(test_arr)

test_loss = yolo_loss(test_arr1,test_arr2)
print(test_loss)
