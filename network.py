from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import os
import metrics as metrics
import loss

def build_network(y, batch=6, optimizer=None, weights_file=None, version=1, input_size=(448,448,3), output_size=392):
    network = build_layers(version=version, input_size=input_size, output_size=output_size);

    if(optimizer==None):
        optimizer = define_optimizer()

    # TODO: Metrics are not correct here
    network.compile(loss=loss.custom_loss(y,batch), optimizer=optimizer, metrics=['top_k_categorical_accuracy'])
    #network.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['top_k_categorical_accuracy'])
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
def train_network(network, x, y, validation_size=0, batch=6, iterations=10):
    # TODO: need to check input to make sure its correct
    network.fit(x = x, y= y, validation_split=validation_size, batch_size=batch, verbose=2, epochs=iterations)
    return network

# Test network
def test_network(network, x, y):
    # TODO: need to check input to make sure its correct
    predicted_output = network.evaluate(x, y, verbose=2)
    y_pred = network.predict(x)
    m = metrics.calculate_metrics(y, predicted_output)
    metrics.draw_box(x, y_pred)
    return m
