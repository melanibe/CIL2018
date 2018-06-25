import tensorflow as tf
import numpy as np
import os 
#from evaluate_discr import evaluate_discriminator
"""Networks construction file.
"""
cwd = os.getcwd()
run_number = 1529618182
checkpoint_file = cwd+"/runs/{}/checkpoints/model-14900".format(run_number) 

def generator(input_noise):
	print("Initializing model")
		# reshape in [batch, height, width, 1] where channels= 1 for tf.nn.conv2d
		####### build the model####
		### output should be 1000 first let's do a 5*5 then 25*25 then 50*50 then 125*125 then 250*250 then 500*500 then 1000*1000 7 deconv layer ?
	with tf.variable_scope("deCNN") as scope:
		with tf.name_scope("intro_layer"):
			h0 = tf.layers.dense(input_noise, units = 25*25)
			h0_bis = tf.reshape(h0, [-1, 25, 25, 1])
		with tf.name_scope("first_deconv"):
			h1 = tf.layers.conv2d_transpose(h0_bis, filters = 25 , kernel_size=[2, 2], strides = 2, activation=tf.nn.relu, padding='SAME')
			print(h1.get_shape())
			h1_bis = tf.reshape(h1,[-1, 250, 250, 1])
			print(h1_bis.get_shape())
		with tf.name_scope("second_deconv"):
			h2 = tf.layers.conv2d_transpose(h1_bis, filters = 4 , kernel_size=[2, 2], strides = 2, activation=tf.nn.relu, padding='SAME')
			print(h2.get_shape())
		# with tf.name_scope("third_deconv"):
		# 	h3 = tf.layers.conv2d_transpose(h2, filters = 1, kernel_size=[2, 2], strides = [1,1], activation=tf.nn.relu, padding='SAME')
		# 	h3_bis = tf.reshape(h3,[-1, 500, 500, 1])
		# 	print(h3_bis.get_shape())
		with tf.name_scope("output_deconv"):
			h4 = tf.layers.conv2d_transpose(h2, filters = 1, kernel_size=[2, 2], strides = 2, activation=tf.nn.relu, padding='SAME')
			print(h4.get_shape())
			h_out = tf.reshape(h4,[-1, 1000, 1000], name="output_images")
			print(h_out.get_shape())
		return h_out
# we want to max the score so mini the negative score 
																					# see how to import evalute dsicriminator maybe put it in other file



