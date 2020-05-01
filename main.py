import numpy as np
import cv2
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import load_data as ld
import network as net

# define minimum needed variables
data_dir = "data" # data location
file_ext = ".npy" # numpy bitmap arrays
classes = ["circle", "square", "hexagon"] # classes to train network on

# load data
x, y = ld.load(classes=classes, data_dir=data_dir, file_ext=file_ext, samples=10, reload_data=False)
x_train, y_train, x_test, y_test = ld.divide_into_sets(input=x, response=y, ratio=0.2)

# train network
yolo = net.build_network(y=y_train, batch=2, version=1, input_size=(448,448,3), output_size=392)
yolo = net.train_network(network=yolo, x=x_train, y=y_train, batch=6, iterations=5)
performance_metrics = net.test_network(network=yolo, x=x_test, y=y_test);
print(performance_metrics)

# test to see if the network can correctly generate the output for an example
res = yolo.predict(x_test[5].reshape(1,448,448,3))
print(res)
