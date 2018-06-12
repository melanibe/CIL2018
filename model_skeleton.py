import tensorflow as tf
import numpy as np

"""Networks construction file.
"""
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


class discriminator_regressor(object):
	def __init__(self, n_hidden, ...):
		"""Model initializer regressor discriminator.
        """
		self.input_img = tf.placeholder(dtype, shape, name='input_img') #dim batch * shape
		self.scores = tf.placeholder(dtype, shape, name='scores')
		batch_size = tf.shape(self.input)[0]
		with tf.device('/gpu:0'):
			# layer 1
			with tf.name_scope("CNN"):
					#TO COMPLETE maybe write a separate function that construct the CNN (see random notes)
			# fully connected
			with tf.name_scope("fully connected"):
					#not sure if needed
			# final output
			with tf.name_scope("output"):					
				predictions_score = tf.layers.dense(previous_layer, units=1, activation=None, name="score_pred")
			# Compute loss
			with tf.name_scope("loss"):
				# prediction at time step t should be input word number t+1
				loss_reg = tf.losses.mean_squared_error(labels=self.scores, predictions = predictions_score, name='loss_reg')



#can also write a model with train first label then regressor reusing the same CNN