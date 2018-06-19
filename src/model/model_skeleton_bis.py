import tensorflow as tf
import numpy as np

"""Networks construction file.
"""

# class discriminator_regressor(object):


class Discriminator(object):

	# parameters:
	# ksize= The size of the window for each dimension of the input tensor.
	# strides = The stride of the sliding window for each dimension of the input tensor.

	def __init__(self, reuse=False, discr_type="regressor", filter_height = 5, filter_width = 5, out_channels1 = 8):
		"""Model initializer regressor discriminator.
		parameters:
		reuse: True if we want to use Model2, False to use Model1
		discr_type: regressor / discriminator (in order to use model2)
		"""

		print("Initializing model")
		self.input_img = tf.placeholder(tf.float32, [None, 1000, 1000], name='input_img')  # dim batch * shape
		# reshape in [batch, height, width, 1] where channels= 1 for tf.nn.conv2d
		self.input = tf.expand_dims(self.input_img, 3) #change name becasue not same dimension
		print(self.input.get_shape()) #ok

		if discr_type == "regressor":
			self.scores = tf.placeholder(tf.float32, [None], name='scores')
			print(tf.shape(self.scores))
		else:
			self.labels = tf.placeholder(tf.int32, [None], name='labels')

		batch_size = tf.shape(self.input_img)[0]
		with tf.device('/gpu:0'):
			with tf.name_scope("Layer1"):
				layer1 = self.avg_pool_2x2(self.input)
			with tf.name_scope("CNN"):
				# that should solve the problem of Model2
				if (reuse):
					tf.get_variable_scope().reuse_variables()
				# First Conv and Pool Layers
				h_conv1 = tf.layers.conv2d(inputs= layer1, \
											filters=out_channels1, \
											kernel_size=[filter_height, filter_width], \
											padding="same", \
											activation=tf.nn.relu)	
				h_pool1 = self.avg_pool_2x2(h_conv1) 
				print(tf.shape(h_pool1))
				shape = h_pool1.get_shape()
				# h_pool1 has to be reshaped before dense layer !
				pool2_flat = tf.reshape(h_pool1, [-1, shape[1]*shape[2]*shape[3]]) #check ok.
				print(pool2_flat.get_shape())

			with tf.name_scope("fully_connected"):
				h_fc1 = tf.layers.dense(pool2_flat, units= 28, activation=tf.nn.relu) #28 is a purely random choice

			with tf.name_scope("output"):
				if discr_type == "regressor":
					#added maximum 8 after layer as 8 max score possible
					#predictions_score = tf.reshape(tf.minimum(8.0, tf.layers.dense(h_fc1, units=1, activation=tf.nn.relu)),\
												 #[-1], name="score_pred")
					predictions_score = tf.reshape(tf.layers.dense(h_fc1, units=1, activation=None),\
												 [-1], name="score_pred")
				else:
					logits = tf.layers.dense(pool2_flat, units=2, activation=None, name="logits")
					predictions_labels = tf.argmax(logits)

			with tf.name_scope("loss"):
				print(batch_size)
				if discr_type == "regressor":
					#using the loss of Kaggle
					self.loss = tf.losses.absolute_difference(labels=self.scores, predictions = predictions_score, reduction=tf.losses.Reduction.MEAN)
				#	self.loss= tf.losses.mean_squared_error(labels=self.scores, predictions = predictions_score)
				else:
					self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels, logits = logits))

	def avg_pool_2x2(self, x):
		return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



#################################   NOT USED STUFF aka Garbage  ####################################
	# Not used yet USELESS BECAUSE IT IS DONE IN TRAIN
	# def save(self, path):
	# 	"""
	# 	Saves the trained model
	# 	:param path: path to the trained model
	# 	"""
	# 	self.model.save(path)
	# 	print("Model saved to {}".format(path))

	# def conv2d(self, x, W):
	# 	input_x = tf.cast(x, tf.float32)
	# 	return tf.nn.conv2d(input=input_x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

# NO MAIN IT IS A CLASS FILE !
