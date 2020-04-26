from tensorflow.keras import layers
from tensorflow import keras
import os
import metrics as metrics

def build_network(loss_func=None, optimizer=None, weights_file=None, version=1, input_size=(None, 448,448,3), output_size=392):
    network = build_layers();

    if(loss_func==None):
        loss_func = define_loss_function()

    if(optimizer==None):
        optimizer = define_optimizer()

    # TODO: Metrics are not correct here
    network.compile(loss=loss_func, optimizer=optimizer, metrics=['top_k_categorical_accuracy'])

    return network

def build_layers(version=1):
    if(version==1):
        model = keras.Sequential()

        # block 1: layers 1-2
        model.add(layers.Convolution2D(filters=64, kernel_size=(7,7) ,strides=(2,2), input_shape=input_size)) # conv1
        model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # block 2: layers 3-4
        model.add(layers.Convolution2D(filters=192, kernel_size=(3,3))) # conv2
        model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # block 3: layers 5-9
        model.add(layers.Convolution2D(filters=128, kernel_size=(1,1))) # conv3
        model.add(layers.Convolution2D(filters=256, kernel_size=(3,3))) # conv4
        model.add(layers.Convolution2D(filters=256, kernel_size=(1,1))) # conv5
        model.add(layers.Convolution2D(filters=512, kernel_size=(3,3))) # conv6
        model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # block 4: layers 10-20
        model.add(layers.Convolution2D(filters=256, kernel_size=(1,1))) # conv7
        model.add(layers.Convolution2D(filters=512, kernel_size=(3,3))) # conv8
        model.add(layers.Convolution2D(filters=256, kernel_size=(1,1))) # conv9
        model.add(layers.Convolution2D(filters=512, kernel_size=(3,3))) # conv10
        model.add(layers.Convolution2D(filters=256, kernel_size=(1,1))) # conv11
        model.add(layers.Convolution2D(filters=512, kernel_size=(3,3))) # conv12
        model.add(layers.Convolution2D(filters=256, kernel_size=(1,1))) # conv13
        model.add(layers.Convolution2D(filters=512, kernel_size=(3,3))) # conv14
        model.add(layers.Convolution2D(filters=512, kernel_size=(1,1))) # conv15
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3))) # conv16
        model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # block 5: layers 21-26
        model.add(layers.Convolution2D(filters=512, kernel_size=(1,1))) # conv17
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3))) # conv18
        model.add(layers.Convolution2D(filters=512, kernel_size=(1,1))) # conv19
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3))) # conv20
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3))) # conv21
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3), strides=(2,2))) # conv22

        # block 6: layers 27-28
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3))) # conv23
        model.add(layers.Convolution2D(filters=1024, kernel_size=(3,3))) # conv24

        # block 7: layers 29-
        model.add(layers.Flatten())
        model.add(layers.Dense(1024))
        model.add(layers.Dense(4096))
        model.add(layers.Dropout())
        model.add(layers.Dense(output_size, activation='softmax'))
    else:
        # In the future, v2, v3, etc. could be implemented too
        print('Version not recognized')

    return model

# Define loss function to optimize for
def define_loss_function():
    # TODO: Need to implement loss function from YOLO paper
    return keras.losses.mean_squared_error

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
