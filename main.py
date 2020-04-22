import numpy as np
import ndjson
import cv2
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

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
x = np.empty([0 , 28, 28, 1])
y = np.empty([0])
max_size = 500

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
for img in np.nditer(x, op_flags=['readwrite']):
    # TODO: Need to resize images to be 448 x 448 x 3
    #img = cv2.resize(img, dsize=(448,488), interpolation=cv2.INTER_LINEAR)
    img[...] = img.astype('float64')
    img[...] = img / 255.0

# generate indices for training and test data
l = x.shape[0]
trainingIndices = np.random.randint(0, l, int(trainingSetRatio * l))
testIndicies = np.arange(0,l)
testIndicies = np.delete(testIndicies, trainingIndices)

# any additional data prep
y = keras.utils.to_categorical(y, 3)

# seperate data
x_train = x[trainingIndices]
y_train = y[trainingIndices]

x_test = x[testIndicies]
y_test = y[testIndicies]

# Define NN model
# TODO: Replace this model with YOLO model
model = keras.Sequential()
model.add(layers.Convolution2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Convolution2D(32, (3,3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Convolution2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# train model
# TODO: Replace generic operations with custo values for optimizer
adam = tf.keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['top_k_categorical_accuracy'])
#print(model.summary())
model.fit(x = x_train, y= y_train, validation_split=0.1, batch_size = 10, verbose=2, epochs=10)
score = model.evaluate(x_test, y_test, verbose=0)
