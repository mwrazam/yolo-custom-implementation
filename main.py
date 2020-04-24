import numpy as np
import cv2
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import sys


def split_into_cells(A,cell_size):
    cell_num = A.shape[1] // cell_size
    blocks = np.array([A[i, x: x+cell_size, y: y+cell_size] for i in range(A.shape[0]) for x in range(0,A.shape[1], cell_size) for y in range(0,A.shape[2], cell_size)])
    print(blocks.shape)
    label_num = (blocks.shape[0]*3*64*64) // (8*7*7)
    cell = np.reshape(blocks,(label_num, A.shape[1] // blocks.shape[1], A.shape[2] // blocks.shape[2], 8))
    return cell


# data location
data_dir = "data"
file_ext = ".npy"

# object classes we are interested in, these should match the file names
# Note: these will get converted to corresponding integers for training/testing
# Note: these will look for data files of the same name in the data directory
classes = ["circle", "square", "hexagon"]
# ratio for how training / test datasets
trainingSetRatio = 0.8

# index values of training and test data elements
trainingIndices = []
testIndicies = []

# holder for full data
# TODO: Need to change to 3 channel image
x = np.empty([0, 28, 28, 1])
y = np.empty([0, 7, 7, 8])
max_size = 500
X_resize = np.empty([max_size*3, 448, 448, 3])

# load and shape data
for i, c in enumerate(classes):

    # load data file per class
    data = np.load(data_dir + "/" + c + file_ext)

    # only use a subset of the data initially, change this later to the full set
    b = data[0:max_size, :, None, None]

    # reshape
    d = np.reshape(b, (len(b), 28, 28, 1))

    # concatenate data
    x = np.concatenate((x, d), axis=0)

    # generate and concatenate labels
    labels = np.full(b.shape[0], i)
    y = np.append(y, labels)

# apply preprocessing
i = 0
for img in x:
    # TODO: Need to resize images to be 448 x 448 x 3
    np.set_printoptions(threshold=sys.maxsize)
    img2 = np.empty([28, 28, 3])
    img2[:,:,0] = img[:,:,0]
    img2[:,:,1] = img[:,:,0]
    img2[:,:,2] = img[:,:,0]
    dim = (448,448)
    img2 = cv2.resize(img2, dim)
    img2[...] = img2.astype('float64')
    img2[...] = img2 / 255.0 
    
    X_resize[i,:,:,:] = img2[:,:,:]
    i += 1
# cell_size = 64
# y = split_into_cells(X_resize,cell_size)
print("The y shape is :")
print(y.shape)
print("After resize the input, X shape is: ")
print(X_resize.shape)

# generate indices for training and test data
l = X_resize.shape[0]
trainingIndices = np.random.randint(0, l, int(trainingSetRatio * l))
testIndicies = np.arange(0,l)
testIndicies = np.delete(testIndicies, trainingIndices)

# # any additional data prep
# y = keras.utils.to_categorical(y, 3)

# # seperate data
# x_train = x[trainingIndices]
# y_train = y[trainingIndices]

# x_test = x[testIndicies]
# y_test = y[testIndicies]

# # Define NN model
# # TODO: Replace this model with YOLO model
# model = keras.Sequential()
# model.add(layers.Convolution2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(layers.Convolution2D(32, (3,3), padding='same', activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(layers.Convolution2D(64, (3,3), padding='same', activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(3, activation='softmax'))

# # train model
# # TODO: Replace generic operations with custo values for optimizer
# adam = tf.keras.optimizers.Adam()
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['top_k_categorical_accuracy'])
# print(model.summary())
# model.fit(x = x_train, y= y_train, validation_split=0.1, batch_size = 10, verbose=2, epochs=10)
# score = model.evaluate(x_test, y_test, verbose=2)

# # TODO: Below utility operations need to be incrporated properly and with error checking

# # TODO: Extract weights to file
# #model.save_weights('./weights/weights')

# # TODO: Save the model
# #model.save('model/model')

# # TODO: Incorporate this into model setup before execution
# #model.load_weights(WEIGHTS FILE NAME HERE)
