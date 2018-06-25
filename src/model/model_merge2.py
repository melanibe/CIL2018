import tensorflow as tf
import numpy as np

"""Networks construction file.
"""

# class discriminator_regressor(object):


class Discriminator(object):

	# parameters:
	# ksize= The size of the window for each dimension of the input tensor.
	# strides = The stride of the sliding window for each dimension of the input tensor.

	def __init__(self, reuse=False, discr_type="regressor", filter_height = 5, filter_width = 5, out_channels1 = 8, out_channels2 = 16):
		"""Model initializer regressor discriminator.
		parameters:
		reuse: True if we want to use Model2, False to use Model1
		discr_type: regressor / discriminator (in order to use model2)
		"""

		print("Initializing model")
		self.input_img_score = tf.placeholder(tf.float32, [None, 1000, 1000], name='input_img_score')  # dim batch * shape
		self.input_img_label = tf.placeholder(tf.float32, [None, 1000, 1000], name='input_img_label')  # dim batch * shape
		# reshape in [batch, height, width, 1] where channels= 1 for tf.nn.conv2d
		self.input_score = tf.expand_dims(self.input_img_score, 3) #change name becasue not same dimension
		self.input_label = tf.expand_dims(self.input_img_label, 3) #change name becasue not same dimension

		#self.discr_type = tf.placeholder(tf.string, name = 'discr_type') # 'discriminator' or 'regressor'

		self.scores = tf.placeholder(tf.float32, [None], name='scores')

		self.labels = tf.placeholder(tf.int32, [None], name='labels')

		# if self.discr_type == "regressor":
		# 	self.scores = tf.placeholder(tf.float32, [None], name='scores')
		# 	print(tf.shape(self.scores))
		# else:
		# 	self.labels = tf.placeholder(tf.int32, [None], name='labels')

		#batch_size = tf.shape(self.input_img_label)[0]

		with tf.device('/gpu:0'):
			# layer 1
			with tf.variable_scope("CNN") as scope:
				# that should solve the problem of Model2
				
				pool2_flat_labels = self.CNN(self.input_score, out_channels1,filter_height, filter_width, out_channels2)
				scope.reuse_variables()
				pool2_flat_scores = self.CNN(self.input_label, out_channels1,filter_height, filter_width, out_channels2)


			with tf.name_scope("fully_connected"):
				h_fc1_labels = tf.layers.dense(pool2_flat_labels, units= 28, activation=tf.nn.relu)
				h_fc1_scores = tf.layers.dense(pool2_flat_scores, units= 28, activation=tf.nn.relu)
			with tf.name_scope("output"):
				#activation reLu beause obvisouly don't want negative number anyway
				predictions_score = tf.reshape(tf.layers.dense(h_fc1_scores, units=1, activation=tf.nn.relu), [-1], name="score_pred")
				
				logits = tf.layers.dense(h_fc1_labels, units=2, activation=tf.nn.relu, name="logits")
				predictions_labels = tf.argmax(logits)

			with tf.name_scope("loss"):
				#print(batch_size)
				loss_scores = tf.losses.absolute_difference(labels=self.scores, predictions = predictions_score, reduction=tf.losses.Reduction.MEAN)
				loss_labels = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels, logits = logits))

				self.loss = tf.reduce_mean(tf.stack([loss_scores, loss_labels], axis = 0), 0)

	def CNN(self, input, out_channels1,filter_height, filter_width, out_channels2):
		# First Conv and Pool Layers
		h_pool0 = self.avg_pool_2x2(input)
		with tf.variable_scope("conv1"):
			h_conv1 = tf.layers.conv2d(inputs= h_pool0, \
										filters=out_channels1, \
										kernel_size=[filter_height, filter_width], \
										padding="same", \
										activation=tf.nn.relu)	
		with tf.variable_scope("pool1"):
			h_pool1 = self.avg_pool_2x2(h_conv1) 
			print(tf.shape(h_pool1))

		# Second Conv and Pool Layers
		with tf.variable_scope("conv2"):
			h_conv2 = tf.layers.conv2d(inputs= h_pool1, \
									 	filters=out_channels2, \
										kernel_size=[filter_height, filter_width],
										padding="same",\
										activation=tf.nn.relu)
		with tf.variable_scope("pool2"):
			h_pool2 = self.avg_pool_2x2(h_conv2) #batch*3200*3200*outchan2
			print(h_pool2.get_shape())
			shape = h_pool2.get_shape()
		with tf.variable_scope("conv3"):
			h_conv3 = tf.layers.conv2d(inputs= h_pool2, \
									 	filters=out_channels2, \
										kernel_size=[filter_height, filter_width],
										padding="same",\
										activation=tf.nn.relu)
		with tf.variable_scope("pool3"):
			h_pool3 = self.avg_pool_2x2(h_conv3) #batch*3200*3200*outchan2
			print(h_pool3.get_shape())
			shape = h_pool3.get_shape()
		# h_pool2 has to be reshaped before dense layer !
		pool2_flat = tf.reshape(h_pool3, [-1, shape[1]*shape[2]*shape[3]]) #check ok.
		print(pool2_flat.get_shape())
		return pool2_flat

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