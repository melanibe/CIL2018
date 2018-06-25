import tensorflow as tf
import numpy as np

"""Networks construction file.
"""

# class discriminator_regressor(object):


class Discriminator(object):

	# parameters:
	# ksize= The size of the window for each dimension of the input tensor.
	# strides = The stride of the sliding window for each dimension of the input tensor.

	def f1(self): 
		print("equal") 
		return 0
	def f2(self): 
		print("not equal")
		return 1

	def __init__(self, reuse=False, filter_height = 5, filter_width = 5, out_channels1 = 8, out_channels2 = 16):
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

		self.discr_type = tf.placeholder(tf.int32, None, name='discr_type') #0 scores 1 labels
		zero = tf.constant(0, dtype = tf.int32)

		self.y = tf.placeholder(tf.float32, [None], name='y')

		batch_size = tf.shape(self.input_img)[0]

		with tf.device('/gpu:0'):
			# layer 1
			with tf.name_scope("reduce"):
				h_pool = self.avg_pool_2x2(self.input)
				h_pool01 = self.avg_pool_2x2(h_pool)
				h_pool0 = self.avg_pool_2x2(h_pool01)
			with tf.name_scope("CNN"):
				# that should solve the problem of Model2
				if (reuse):
					tf.get_variable_scope().reuse_variables()
				# First Conv and Pool Layers
				h_conv1 = tf.layers.conv2d(inputs= h_pool0, \
											filters=out_channels1, \
											kernel_size=[filter_height, filter_width], \
											padding="same", \
											activation=tf.nn.relu)	
				h_pool1 = self.avg_pool_2x2(h_conv1) 
				print(tf.shape(h_pool1))
				# Second Conv and Pool Layers
				h_conv2 = tf.layers.conv2d(inputs= h_pool1, \
										 	filters=out_channels2, \
											kernel_size=[filter_height, filter_width],
											padding="same",\
											activation=tf.nn.relu)
				h_pool2 = self.avg_pool_2x2(h_conv2) #batch*3200*3200*outchan2
				h_conv3 = tf.layers.conv2d(inputs= h_pool2, \
										 	filters=out_channels2, \
											kernel_size=[filter_height, filter_width],
											padding="same",\
											activation=tf.nn.relu)
				h_pool3 = self.avg_pool_2x2(h_conv3) #batch*3200*3200*outchan2
				print(h_pool3.get_shape())
				shape = h_pool3.get_shape()
				# h_pool2 has to be reshaped before dense layer !
				pool3_flat = tf.reshape(h_pool3, [-1, shape[1]*shape[2]*shape[3]]) #check ok.
				print(pool3_flat.get_shape())

			#with tf.name_scope("fully connected"):
			#	h_fc1 = tf.layers.dense(h_fc1, units= n_hidden_fconnected, activation=None)

			with tf.name_scope("output"):
				#activation reLu beause obvisouly don't want negative number anyway
				print(self.discr_type[0])

				def f1():
					#added maximum 8 after layer as 8 max score possible
					predictions_score = tf.reshape(tf.layers.dense(pool3_flat, units=1, activation=None), [-1], name="score_pred")
					print(predictions_score.dtype)
					return predictions_score

				def f2():
					logits = tf.cast(tf.layers.dense(pool3_flat, units=2, activation=tf.nn.relu, name="logits"), tf.float32)
					predictions_labels = tf.cast(tf.argmax(logits), tf.float32)
					return logits

				prediction = tf.cond(tf.equal(self.discr_type,zero), f1, f2)
				#logits = tf.cond(tf.equal(self.discr_type,zero), f1, tf.layers.dense(pool2_flat, units=2, activation=tf.nn.relu, name="logits"))

			with tf.name_scope("loss"):
				print(batch_size)
				def f3():
					#self.loss= tf.losses.mean_squared_error(labels=self.scores, predictions = predictions_score)
					return tf.losses.absolute_difference(labels=self.y, predictions = prediction, reduction=tf.losses.Reduction.MEAN)
				def f4():
					return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(self.y, tf.int32), logits = prediction)) #prediction are logits for labels

				self.loss =  tf.cond(tf.equal(self.discr_type,zero), f3, f4)

	def avg_pool_2x2(self, x):
		return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
