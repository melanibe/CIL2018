import tensorflow as tf
import numpy as np
import os 
"""Mélanie Bernhardt - Laura Manducchi - Mélanie Gaillochet
This file is a helper file to build the generator.
"""
def generator(input_noise, batch_size):
	print("Initializing generator")
	with tf.variable_scope("deCNN") as scope:
		with tf.name_scope("intro_layer"):
			h0 = tf.layers.dense(input_noise, units=25*25*124)
			h0_projected = tf.reshape(h0, [-1, 25, 25, 124])
		with tf.name_scope("first_deconv"):
			h1 = tf.layers.conv2d_transpose(h0_projected, \
											filters=100, \
											kernel_size=[5, 5], \
											strides = 2, \
											activation=tf.nn.relu, \
											padding='SAME')
			print(h1.get_shape()) # [None, 50, 50, 100]
			h1_bis = tf.reshape(h1,[-1, 125, 125, 16])
			print(h1_bis.get_shape())
		with tf.name_scope("second_deconv"):
			h2 = tf.layers.conv2d_transpose(h1_bis, filters=50, kernel_size=[5, 5], strides = 2, activation=tf.nn.relu, padding='SAME')
			print(h2.get_shape()) # [None, 250, 250, 50]
		with tf.name_scope("third_deconv_perturbed"):
			h3 = tf.layers.conv2d_transpose(h2, filters=10, kernel_size=[2, 2], strides = 2, activation=tf.nn.relu, padding='SAME')
			# perturbating the output of the third deconv layer
			h3_perturbed = h3 + np.random.normal(0, 1, [batch_size, 500, 500, 10])
			print(h3.get_shape()) # [None, 500, 500, 4]
		with tf.name_scope("output_deconv"):
			h4 = tf.layers.conv2d_transpose(h3_perturbed, filters = 1, kernel_size=[2, 2], strides = 2, activation=tf.nn.tanh, padding='SAME')
			print(h4.get_shape()) # [None, 1000, 1000, 1]
			# re-rescaling the ouptut on a [1, 255] scale to be consistent with the original scored images.
			h_out = ((1+tf.reshape(h4,[-1, 1000, 1000], name="output_images"))/2)*255
			print(h_out.get_shape())
		return h_out


