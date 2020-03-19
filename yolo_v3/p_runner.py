import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from seaborn import color_palette
import numpy as np
import cv2

#Configuration file
import config
from yolo import yolov3


def load_images(img_names, model_size):
	pass


def load_class_names(file_name):
	pass


def load_weights(variables, file_name):
	pass


def draw_boxes():
	pass


def main():
	img_names = ['input/dog.jpg', 'input/person.jpg']
	for img in img_names: 
		Image.open(img).show()
	batch_size = len(img_names)
	batch = load_images(img_names, model_size=config._MODEL_SIZE)
	class_names = load_class_names('input/coco.names')
	
	# model = yolov3(n_classes=n_classes, model_size=_MODEL_SIZE,
	#                 max_output_size=max_output_size,
	#                 iou_threshold=iou_threshold,
	#                 confidence_threshold=confidence_threshold)

	yolov3()

	# Returns a Tensor that may be used as a handle for feeding a value, but not evaluated directly.
	inputs = tf.compat.v1.placeholder(tf.float32, [batch_size, 416, 416, 3])

	print("This is main function.")

	# draw_boxes(img_names, detection_result, class_names, config._MODEL_SIZE)

if __name__ == "__main__":
    main()

#
# p_runner.py ends here