import tensorflow as tf
import numpy as np
import os 

def avg_pool_2x2(x):
		return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def discriminator(input_images, reuse=False, out_channels1=32, out_channels2=8, filter_height=3, filter_width=3):
	with tf.variable_scope("discr") as scope:
		if (reuse):
			tf.get_variable_scope().reuse_variables()
		input = tf.expand_dims(input_images, 3)
		with tf.name_scope("CNN"):	
			h_pool = avg_pool_2x2(input)
			h_pool01 = avg_pool_2x2(h_pool)
			h_pool02 = avg_pool_2x2(h_pool01)
			h_pool0 = avg_pool_2x2(h_pool02)
			h_conv1 = tf.layers.conv2d(inputs= h_pool0, \
											filters=out_channels1, \
											kernel_size=[filter_height, filter_width], \
											padding="same", \
											activation=tf.nn.relu)	
			h_pool1 = avg_pool_2x2(h_conv1)
			print(tf.shape(h_pool1))
			# Second Conv and Pool Layers
			h_conv2 = tf.layers.conv2d(inputs= h_pool1, \
										 	filters=out_channels2, \
											kernel_size=[filter_height, filter_width],
											padding="same",\
											activation=tf.nn.relu)								
			h_pool2 = avg_pool_2x2(h_conv2) #batch*3200*3200*outchan2
			h_conv3 = tf.layers.conv2d(inputs= h_pool2, \
										 	filters=out_channels2, \
											kernel_size=[filter_height, filter_width],
											padding="same",\
											activation=tf.nn.relu)
			h_pool3 = avg_pool_2x2(h_conv3) #batch*3200*3200*outchan2                
			print(h_pool3.get_shape())
			shape = h_pool3.get_shape()
			pool2_flat = tf.reshape(h_pool3, [-1, shape[1]*shape[2]*shape[3]]) #check ok.
			print(pool2_flat.get_shape())
		with tf.name_scope("fully_connected"):
			h_fc1 = tf.layers.dense(pool2_flat, units= 28, activation=tf.nn.relu)
		with tf.name_scope("output"):
			predictions_score = tf.reshape(tf.layers.dense(h_fc1, units=1, activation=None),\
												 [-1], name="score_pred")
		return predictions_score
