import tensorflow as tf
import numpy as np

"""Networks construction file.
"""

#class discriminator_regressor(object):
class discriminator(object):

	#parameters: 
	#ksize= The size of the window for each dimension of the input tensor.
	#strides = The stride of the sliding window for each dimension of the input tensor.
	def conv2d(x, W):
		return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

	def avg_pool_2x2(x):
		return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def __init__(self, im_dtype, reuse= False, discr_type= "regressor"):
		"""Model initializer regressor discriminator.
		parameters:
		reuse: True if we want to use Model2, False to use Model1
		discr_type: regressor / discriminator (in order to use model2)
        """

        
		self.input_img = tf.placeholder(im_dtype, [None,None,None], name='input_img') #dim batch * shape
		#reshape in [batch, height, width, channels] where channels= 1 for tf.nn.conv2d
		self.input_img = tf.reshape(self.input_img, [tf.shape(self.input)[0], tf.shape(self.input)[1], tf.shape(self.input)[2], 1])

		if discr_type == "regressor":
			self.scores = tf.placeholder(dtype, shape, name='scores')
		else:
			self.labels = tf.placeholder(dtype, shape, name='labels')

		batch_size = tf.shape(self.input)[0]

		with tf.device('/gpu:0'):
			# layer 1
			with tf.name_scope("CNN"):
				#TO COMPLETE maybe write a separate function that construct the CNN (see random notes)

				#parameters to investigate: 
				filter_height = 5
				filter_width = 5
				out_channels1 = 8
				out_channels2 = 16
				n_hidden_fconnected = 32

				#that should solve the problem of Model2
				if (reuse):
					tf.get_variable_scope().reuse_variables()

				#First Conv and Pool Layers
				W_conv1 = tf.get_variable('d_wconv1', [filter_height, filter_width, 1, out_channels1], initializer=tf.truncated_normal_initializer(stddev=0.02))
				b_conv1 = tf.get_variable('d_bconv1', [out_channels1], initializer=tf.constant_initializer(0))
				h_conv1 = tf.nn.relu(conv2d(self.input_img, W_conv1) + b_conv1)
				h_pool1 = avg_pool_2x2(h_conv1)

				#Second Conv and Pool Layers
				W_conv2 = tf.get_variable('d_wconv2', [filter_height, filter_width, out_channels1, out_channels2], initializer=tf.truncated_normal_initializer(stddev=0.02))
				b_conv2 = tf.get_variable('d_bconv2', [out_channels2], initializer=tf.constant_initializer(0))
				h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
				h_pool2 = avg_pool_2x2(h_conv2)

			# fully connected
			with tf.name_scope("fully connected"):

				#not sure if it's necessary (this is not shared between labeled and scored (to change?))
				h_fc1 = tf.layers.dense(h_fc1, units= n_hidden_fconnected, activation=None)

			# final output
			with tf.name_scope("output"):

				if discr_type == "regressor":
					predictions_score = tf.layers.dense(h_fc1, units=1, activation=None, name="score_pred")
				else:
					logits = tf.layers.dense(h_fc1, units=2, activation=None, name="logits")
					predictions_labels = tf.argmax(logits)

			# Compute loss
			with tf.name_scope("loss"):

				if discr_type == "regressor":
					loss_reg = tf.losses.mean_squared_error(labels=self.scores, predictions = predictions_score, name='loss_reg')
				else:
					loss_label = tf.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels, name='loss_label')


class discriminator_label(object):

	def __init__(self, n_hidden, ...):
		"""Model initializer label discriminator.
        """
		self.input_img = tf.placeholder(dtype, shape, name='input img') #dim batch * shape
		self.labels = tf.placeholder(dtype, shape, name='labels')
		batch_size = tf.shape(self.input)[0]
		with tf.device('/gpu:0'):
			# layer 1
			with tf.name_scope("CNN"):
					#TO COMPLETE maybe write a separate function that construct the CNN (see random notes)
			# fully connected
			with tf.name_scope("fully connected"):
				logits = tf.dense ...
			# final output
			with tf.name_scope("output"):					
				predictions_labels = tf.argmax(logits)
			# Compute loss
			with tf.name_scope("loss"):
				# prediction at time step t should be input word number t+1
				loss_label = tf.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels, name='loss_label')



#can also write a model with train first label then regressor reusing the same CNN