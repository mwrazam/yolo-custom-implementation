from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import os
import metrics as metrics

def build_network(loss_func=None, optimizer=None, weights_file=None, version=1, input_size=(448,448,3), output_size=392):
    network = build_layers(version=version, input_size=input_size, output_size=output_size);

    if(loss_func==None):
        loss_func = define_loss_function()

    if(optimizer==None):
        optimizer = define_optimizer()

    # TODO: Metrics are not correct here
    network.compile(loss=loss_func, optimizer=optimizer, metrics=['top_k_categorical_accuracy'])

    return network

def build_layers(version, input_size, output_size):
    if(version==1):
        model = keras.Sequential()

        # block 1: layers 1-2
        model.add(layers.Convolution2D(filters=64, kernel_size=(7,7) ,strides=(2,2), input_shape=input_size, padding='same')) # conv1
        model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # block 2: layers 3-4
        model.add(layers.Convolution2D(filters=192, kernel_size=(3,3), padding='same')) # conv2
        model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # block 3: layers 5-9
        model.add(layers.Convolution2D(filters=128, kernel_size=(1,1), padding='same')) # conv3
        model.add(layers.Convolution2D(filters=256, kernel_size=(3,3), padding='same')) # conv4
        model.add(layers.Convolution2D(filters=256, kernel_size=(1,1), padding='same')) # conv5
        model.add(layers.Convolution2D(filters=512, kernel_size=(3,3), padding='same')) # conv6
        model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # block 4: layers 10-20
        model.add(layers.Convolution2D(filters=256, kernel_size=(1,1), padding='same')) # conv7
        model.add(layers.Convolution2D(filters=512, kernel_size=(3,3), padding='same')) # conv8
        model.add(layers.Convolution2D(filters=256, kernel_size=(1,1), padding='same')) # conv9
        model.add(layers.Convolution2D(filters=512, kernel_size=(3,3), padding='same')) # conv10
        model.add(layers.Convolution2D(filters=256, kernel_size=(1,1), padding='same')) # conv11
        model.add(layers.Convolution2D(filters=512, kernel_size=(3,3), padding='same')) # conv12
        model.add(layers.Convolution2D(filters=256, kernel_size=(1,1), padding='same')) # conv13
        model.add(layers.Convolution2D(filters=512, kernel_size=(3,3), padding='same')) # conv14
        model.add(layers.Convolution2D(filters=512, kernel_size=(1,1), padding='same')) # conv15
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3), padding='same')) # conv16
        model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # block 5: layers 21-26
        model.add(layers.Convolution2D(filters=512, kernel_size=(1,1), padding='same')) # conv17
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3), padding='same')) # conv18
        model.add(layers.Convolution2D(filters=512, kernel_size=(1,1), padding='same')) # conv19
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3), padding='same')) # conv20
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3))) # conv21
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3), strides=(2,2))) # conv22

        # block 6: layers 27-28
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3), padding='same')) # conv23
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3), padding='same')) # conv24

        # block 7: layers 29-
        model.add(layers.Flatten())
        model.add(layers.Dense(1024))
        model.add(layers.Dense(4096))
        model.add(layers.Dropout(rate=0.1))
        model.add(layers.Dense(output_size, activation='softmax'))
        print(model.summary())
    else:
        # In the future, v2, v3, etc. could be implemented too
        print('Version not recognized')

    return model

# Define loss function to optimize network
# y_true, y_pred: (1500,7,7,8)
# 8: pc,bx,by,bw,bh,c1,c2,c3
def define_loss_function(y_true, y_pred):
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

# Optimizer to use for network training
def define_optimizer():
    return tf.keras.optimizers.Adam()

# Save optimized weights to file
def save_weights(network, file_name, weights_dir):
    if(exists(weights_dir + "/ + file_name")):
        try:
            os.rmdir(weights_dir)
        except Exception as e:
            # TODO: need to implement error handling for file delete
            raise
    os.mkdir(weights_dir)
    network.save_weights(file)
    return None

# Load optimized weights from file
def load_weights(network, file):
    if(exists(file)):
        network.load_weights(file)
    else:
        print("Could not find weights file: %s" % file)
    return network

# Train network
def train_network(network, x, y, validation_size=0.1, batch=10, iterations=10):
    # TODO: need to check input to make sure its correct
    network.fit(x = x, y= y, validation_split=validation_size, batch_size=batch, verbose=2, epochs=iterations)
    return network

# Test network
def test_network(network, x, y):
    # TODO: need to check input to make sure its correct
    predicted_output = network.evaluate(x, y, verbose=2)
    m = metrics.calculate_metrics(y, predicted_output)
    return m
