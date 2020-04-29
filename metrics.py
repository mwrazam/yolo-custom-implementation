import cv2
import math
import numpy as np
import tensorflow as tf
#from keras import backend as K
from tensorflow.keras import backend as K

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
                    cv2.rectangle(cell, start_p, end_p, color, thickness)
                    cv2.putText(cell, label_str, (box.xmin, box.ymin - 13), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)

