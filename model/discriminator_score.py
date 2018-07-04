import tensorflow as tf
import numpy as np
import os

"""Mélanie Bernhardt - Laura Manduchi - Mélanie Gaillochet
This file is a helper file to build the score discriminant.
"""

def avg_pool_2x2(x):
	"""
	Helper function to design our 2 by 2 average pooling layer.
	"""
	return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def discriminator_score(input_images, reuse=False, out_channels1=32, out_channels2=8, filter_height=3, filter_width=3):
	"""Fonction building the score discriminator. 
	Args:
		input_images(tensor [None, 1000, 1000]): batch of images for which we want to predict the logits.
		reuse (bool): whether or not we want to reuse the variables of the discriminator in the graph.
		out_channels1 (int): number of output channels for the first convoluational layer
		out_channels2 (int): number of output channels for the second convolutional layer
		filter_height (int): height of the filters for all the concolutional layers
		filter_width (int): width of the filters for all the concolutional layers
	Returns: 
		predictions_score(tensor [None, 1]): the predicted score for each input image in the batch..
	"""
	with tf.variable_scope("discr") as scope:
		if (reuse):
			tf.get_variable_scope().reuse_variables()
		# Reshaping the input (necessary for convoluational layer input)
		input = tf.expand_dims(input_images, 3)
		with tf.name_scope("CNN"):
			with tf.name_scope("intro_pooling"):	
				h_pool = avg_pool_2x2(input)
				h_pool01 = avg_pool_2x2(h_pool)
				h_pool02 = avg_pool_2x2(h_pool01)
				h_pool03 = avg_pool_2x2(h_pool02)
			with tf.name_scope("first_conv_pool"):
				h_conv1 = tf.layers.conv2d(inputs=h_pool03, \
											filters=out_channels1, \
											kernel_size=[filter_height, filter_width], \
											padding="same", \
											activation=tf.nn.relu)	
				h_pool1 = avg_pool_2x2(h_conv1)
				print(tf.shape(h_pool1))
			with tf.name_scope("second_conv_pool"):
				h_conv2 = tf.layers.conv2d(inputs=h_pool1, \
										 	filters=out_channels2, \
											kernel_size=[filter_height, filter_width],
											padding="same",\
											activation=tf.nn.relu)								
				h_pool2 = avg_pool_2x2(h_conv2)
			with tf.name_scope("third_conv_pool"):
				h_conv3 = tf.layers.conv2d(inputs=h_pool2, \
										 	filters=out_channels2, \
											kernel_size=[filter_height, filter_width],
											padding="same",\
											activation=tf.nn.relu)
				h_pool3 = avg_pool_2x2(h_conv3)               
				print(h_pool3.get_shape())
				shape = h_pool3.get_shape()
				# flattening the layer to feed it to a dense layer
				pool3_flat = tf.reshape(h_pool3, [-1, shape[1]*shape[2]*shape[3]]) #check ok.
				print(pool3_flat.get_shape())
		with tf.name_scope("fully_connected"):
			h_fc1 = tf.layers.dense(pool3_flat, units=28, activation=tf.nn.relu)
		with tf.name_scope("output"):
			predictions_score = tf.reshape(tf.layers.dense(h_fc1, units=1, activation=None),\
												 [-1], name="score_pred")
		return predictions_score
