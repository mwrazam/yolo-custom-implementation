from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

lamda_coord = 5
lamda_noobj = 0.5
gridcells = 7**2
classes = 3

def yolo_pcloss(pc_true, pc_pred):
    lo = K.square(pc_true - pc_pred)
    if pc_true == 1:
        loss1 = lo
    else:
        loss1 = lamda_noobj*(lo)
    loss = K.mean(loss1)
    return loss

def yolo_xyloss(a_true, a_pred, pc_true):
    lo = K.square(a_true - a_pred)
    if pc_true == 1:
        loss1 = lamda_coord*(lo)
    else:
        loss1 = K.zeros_like(a_true)
    loss = K.mean(loss1)
    return loss

def yolo_whloss(a_true, a_pred, pc_true):
    lo = K.square(K.sqrt(a_true)-K.sqrt(a_pred))
    if pc_true == 1:
        loss1 = lamda_coord*(lo)
    else:
        loss1 = K.zeros_like(a_true)
    loss = K.mean(loss1)
    return loss

def yolo_classloss(a_true, a_pred, pc_true):
    lo = K.square(a_true - a_pred)
    if pc_true == 1:
        loss1 = lo
    else:
        loss1 = K.zeros_like(a_true)
    loss = K.mean(loss1)
    return loss

# y_true, y_pred: (1500,7,7,8)
# 8: pc,bx,by,bw,bh,c1,c2,c3
def yolo_loss(y_true, y_pred):
    # truth table format is [[confid,x,y,w,h]..,classes] for one cell
    # Slice the pc,x,y,w,h of y_true
    pc_t = tf.slice(y_true, [0,0], [-1,gridcells])
    x_t = tf.slice(y_true, [0,gridcells], [-1,gridcells])
    y_t = tf.slice(y_true, [0,gridcells*2], [-1,gridcells])
    w_t = tf.slice(y_true, [0,gridcells*3], [-1,gridcells])
    h_t = tf.slice(y_true, [0,gridcells*4], [-1,gridcells])

    # Slice the pc,x,y,w,h of y_pred
    pc_p = tf.slice(y_pred, [0,0], [-1,gridcells])
    x_p = tf.slice(y_pred, [0,gridcells], [-1,gridcells])
    y_p = tf.slice(y_pred, [0,gridcells*2], [-1,gridcells])
    w_p = tf.slice(y_pred, [0,gridcells*3], [-1,gridcells])
    h_p = tf.slice(y_pred, [0,gridcells*4], [-1,gridcells])

    # Slice c1,c2,c3 of y_true and y_pred
    classes_true = []
    classes_pred = []
    for i in range(classes):
        ct = tf.slice(y_true, [0,gridcells*(5+i)], [-1,gridcells])
        classes_true.append(ct)

    for i in range(classes):
        cp = tf.slice(y_pred, [0,gridcells*(5+i)], [-1,gridcells])
        classes_pred.append(cp)

    print(classes_true)
    print(classes_pred)

    # w_true = tf.compat.v1.placeholder(tf.float32, shape=(49,49))
    # w_pred = tf.compat.v1.placeholder(tf.float32, shape=(49,49))
    # h_true = tf.compat.v1.placeholder(tf.float32, shape=(49,49))
    # h_pred = tf.compat.v1.placeholder(tf.float32, shape=(49,49))
    # Calculate losses
    pcloss = yolo_pcloss(pc_t,pc_p)
    xloss = yolo_xyloss(x_t,x_p,pc_t)
    yloss = yolo_xyloss(y_t,y_p,pc_t)
    wloss = yolo_whloss(w_t,w_p,pc_t)
    hloss = yolo_whloss(h_t,h_p,pc_t)
    classesloss = 0
    for i in range(classes):
        cla_loss = yolo_classloss(classes_true[i], classes_pred[i], pc_t)
        classesloss += cla_loss
        
    loss = pcloss+xloss+yloss+wloss+hloss+classesloss

    return loss