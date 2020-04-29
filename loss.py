from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

lamda_coord = 5
lamda_noobj = 0.5
gridcells = 7**2
classes = 3

def yolo_pcloss(pc_true, pc_pred, t):
    lo = K.square(pc_true - pc_pred)
    loss_t = lo
    loss_f = lamda_noobj*(lo)
    loss1 = tf.where(t, loss_t, loss_f)
    loss = K.mean(loss1)
    return loss

def yolo_xyloss(a_true, a_pred, t):
    lo = K.square(a_true - a_pred)
    loss_t = lamda_coord*(lo)
    loss_f = K.zeros_like(a_true)
    loss1 = tf.where(t, loss_t, loss_f)
    loss = K.mean(loss1)
    return loss

def yolo_whloss(a_true, a_pred, t):
    lo = K.square(K.sqrt(a_true)-K.sqrt(a_pred))
    loss_t = lamda_coord*(lo)
    loss_f = K.zeros_like(a_true)
    loss1 = tf.where(t, loss_t, loss_f)
    loss = K.mean(loss1)
    return loss

def yolo_classloss(a_true, a_pred, t):
    lo = K.square(a_true - a_pred)
    loss_t = lo
    loss_f = K.zeros_like(a_true)
    loss1 = tf.where(t, loss_t, loss_f)
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
    t = K.greater(pc_t, 0.5)

    # Calculate losses
    pcloss = yolo_pcloss(pc_t,pc_p,t)
    xloss = yolo_xyloss(x_t,x_p,t)
    yloss = yolo_xyloss(y_t,y_p,t)
    wloss = yolo_whloss(w_t,w_p,t)
    hloss = yolo_whloss(h_t,h_p,t)
    classesloss = 0
    for i in range(classes):
        cla_loss = yolo_classloss(classes_true[i], classes_pred[i], t)
        classesloss += cla_loss
        
    loss = pcloss+xloss+yloss+wloss+hloss+classesloss
    # return loss
    return loss,pcloss,xloss,yloss,wloss,hloss,classesloss


# Loss function Testing
x =K.placeholder(ndim=2)
y =K.placeholder(ndim=2)
loss,confidloss,xloss,yloss,wloss,hloss,classesloss = yolo_loss(y,x)

print("### Got all loss values ###")
f = K.function([y,x], [loss,confidloss,xloss,yloss,wloss,hloss,classesloss])


print("### Training ###")
xtrain = np.ones(392*10).reshape(10,392)
ytrain = np.zeros(392*10).reshape(10,392)
ytrain[0][0]=1
ytrain[0][49]=0.1
ytrain[0][49*2]=0.2
ytrain[0][49*3]=0.3
ytrain[0][49*4]=0.4
ytrain[0][49*5]=1

print("### Print Results ###")
predictions = f([ytrain,xtrain])
print(predictions)