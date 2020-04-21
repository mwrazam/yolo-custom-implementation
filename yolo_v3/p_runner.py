import tensorflow as tf
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from darknet import Darknet
import matplotlib.pyplot as plt

#Configuration file
import config


def load_class_names(file_name):
	pass


def load_weights(variables, file_name):
	pass


def draw_boxes():
	pass


def main():
	# load image
	# Code need to change based on the input dataset type
	img_names = 'input/dog.jpg'
	# set the required size
	image = cv2.imread(img_names)
	original_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	resized_img = cv2.resize(original_img, config._MODEL_SIZE)
	# Display the images
	plt.subplot(121)
	plt.title('Original Image')
	plt.imshow(original_img)
	plt.subplot(122)
	plt.title('Resized Image')
	plt.imshow(resized_img)
	plt.show()

	# convert to numpy array
	img = img_to_array(resized_img)

	name_file = 'data/coco.names'
	cfg_file = 'cfg/yolov3.cfg'
	
	class_names = load_class_names(name_file)
	model = Darknet(cfg)
	print("This is main function.")

	# draw_boxes(img_names, detection_result, class_names, config._MODEL_SIZE)

if __name__ == "__main__":
    main()

#
# p_runner.py ends here