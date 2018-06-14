import preprocessing
from model.model_skeleton import Discriminator
from config import *
import pandas as pd

import tensorflow as tf
import numpy as np
import os
import time
import datetime

import sys

import pickle

"""Launch file for training tasks
"""

## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data used for validation")

# Model parameters
tf.flags.DEFINE_boolean("reuse", False, "reuse")
tf.flags.DEFINE_string("discr_type", "regressor", "type of discriminant")
tf.flags.DEFINE_integer("filter_height", 5, "filter_height")
tf.flags.DEFINE_integer("filter_width", 5, "filter_width")
tf.flags.DEFINE_integer("out_channels1", 8, "out_channels1")
tf.flags.DEFINE_integer("out_channels2", 8, "out_channels2")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# for running on EULER
tf.flags.DEFINE_integer("inter_op_parallelism_threads", 16,
	"TF nodes that perform blocking operations are enqueued on a pool of inter_op_parallelism_threads available in each process (default 0).")
tf.flags.DEFINE_integer("intra_op_parallelism_threads", 16,
	"The execution of an individual op (for some op types) can be parallelized on a pool of intra_op_parallelism_threads (default: 0).")

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value.value))
print("")


###############################  DATA PREPARATION   ###############################
# Load data
print("Loading and preprocessing training... \n")

# Saving or loading the objects:
if os.path.isfile("score.pickle"):
	max_bytes = 2**31 - 1
	input_size = os.path.getsize("score.pickle")
	bytes_in = bytearray(0)
	with open("score.pickle", 'rb') as f_in:
		for _ in range(0, input_size, max_bytes):
			bytes_in += f_in.read(max_bytes)
	scored_data = pickle.loads(bytes_in)

else:
	data_scored = preprocessing.load_data("scored")
	imgs = np.reshape(np.array(data_scored['img'].values), (-1,1)) #dim: 9600*1000*1000
	scores = np.reshape(np.array(data_scored['scored'].values), (-1,1)) #dim: 9600
	scored_data = np.concatenate((imgs,scores), axis = 1)
	max_bytes = 2**31 - 1
	bytes_out = pickle.dumps(scored_data)
	n_bytes = sys.getsizeof(bytes_out)
	try:
		with open("score.pickle", 'wb') as f_out:
			for idx in range(0, n_bytes, max_bytes):
				f_out.write(bytes_out[idx:idx+max_bytes])
	except:
		pass

##### same for labell
if os.path.isfile("label.pickle"):
	max_bytes = 2**31 - 1
	input_size = os.path.getsize("label.pickle")
	bytes_in = bytearray(0)
	with open("label.pickle", 'rb') as f_in:
		for _ in range(0, input_size, max_bytes):
			bytes_in += f_in.read(max_bytes)
	labeled_data = pickle.loads(bytes_in)

else:
	labeled_data = preprocessing.load_data("labeled")
	imgs = np.reshape(np.array(labeled_data['img'].values), (-1,1)) #dim: 9600*1000*1000
	labels = np.reshape(np.array(labeled_data['labeled'].values), (-1,1)) #dim: 9600
	labeled_data = np.concatenate((imgs,labels), axis = 1)
	max_bytes = 2**31 - 1
	bytes_out = pickle.dumps(labeled_data)
	n_bytes = sys.getsizeof(bytes_out)
	try:
		with open("label.pickle", 'wb') as f_out:
			for idx in range(0, n_bytes, max_bytes):
				f_out.write(bytes_out[idx:idx+max_bytes])
	except:
		pass

print("Data loaded")

# Randomly shuffle data
np.random.seed(10)

#train/test split for score
shuffled_indices = np.random.permutation(len(scored_data[:,0]))
dev_sample_index = int(FLAGS.dev_sample_percentage * float(len(shuffled_indices)))
test_indices = shuffled_indices[:dev_sample_index]
train_indices = shuffled_indices[dev_sample_index:]
train_score = scored_data[train_indices,:]
test_scored = scored_data[test_indices,:]
#needed for dev step
test_score_imgs = np.reshape(np.concatenate(test_scored[:, 0]), (-1,1000,1000))
test_score = np.reshape(test_scored[:, 1], (-1))
#just to check
#print(np.shape(train_score)) #9600 * 2 one dimension per column but first column is 9600*9600
#print(train_score[0]) # 9600*9600


#train/test split for labels
shuffled_indices = np.random.permutation(len(labeled_data))
dev_sample_index = int(FLAGS.dev_sample_percentage * float(len(shuffled_indices)))
test_indices = shuffled_indices[:dev_sample_index]
train_indices = shuffled_indices[dev_sample_index:]
train_label = labeled_data[train_indices,:]
test_labeled = labeled_data[test_indices,:]
#needed for dev step
test_label_imgs = np.reshape(np.concatenate(test_labeled[:, 0]), (-1,1000,1000))
test_label = np.reshape(test_labeled[:, 1], (-1))

# Generate training batches 
batches_label = preprocessing.batch_iter(train_label, FLAGS.batch_size, FLAGS.num_epochs) #have to check if the function still works with only one input x
batches_score = preprocessing.batch_iter(train_score, FLAGS.batch_size, FLAGS.num_epochs)


## MODEL AND TRAINING PROCEDURE DEFINITION ##

graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement,
		inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
		intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads
		)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		# Initialize model
		discr = Discriminator(reuse=FLAGS.reuse, \
							discr_type=FLAGS.discr_type, \
							filter_height = FLAGS.filter_height,\
							 filter_width = FLAGS.filter_width, \
							 out_channels1 = FLAGS.out_channels1, \
							 out_channels2 = FLAGS.out_channels2)

		# Define an optimizer with clipping the gradients
		global_step = tf.Variable(0, name="global_step", trainable= False)
		optimizer = tf.train.AdamOptimizer()
		gradient_var_pairs = optimizer.compute_gradients(discr.loss)
		vars = [x[1] for x in gradient_var_pairs]
		gradients = [x[0] for x in gradient_var_pairs]
		clipped, _ = tf.clip_by_global_norm(gradients, 5)
		train_op = optimizer.apply_gradients(zip(clipped, vars), global_step = global_step)
		
		# Output directory for models and summaries
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		print("Writing to {}\n".format(out_dir))

		# Loss summaries
		loss_summary = tf.summary.scalar("loss", discr.loss)

		# Train summaries
		train_summary_op = tf.summary.merge([loss_summary])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

		# Dev summaries
		dev_summary_op = tf.summary.merge([loss_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

		# Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

		# Initialize all variables
		sess.run(tf.global_variables_initializer())
		sess.graph.finalize()

		# Define training and dev steps (batch)
		def train_step(batch_imgs, batch_score, type= FLAGS.discr_type):
			"""
			A single training step
			"""
			if type == "regressor":
				feed_dict = {
					discr.input_img: batch_imgs,
					discr.scores: batch_score
				}
			else:
				feed_dict = {
					discr.input_img: batch_imgs,
					discr.labels: batch_score
				}				
			_, step, summaries, loss = sess.run(
				[train_op, global_step, train_summary_op, discr.loss],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}".format(time_str, step, loss))
			train_summary_writer.add_summary(summaries, step)

		def dev_step(batch_imgs, batch_score, type= FLAGS.discr_type, writer=None):
			"""
			Evaluates model on a dev set
			"""
			if type == "regressor":
				feed_dict = {
					discr.input_img: batch_imgs,
					discr.scores: batch_score
				}
			else:
				feed_dict = {
					discr.input_img: batch_imgs,
					discr.labels: batch_score
				}
			step, summaries, loss = sess.run(
				[global_step, dev_summary_op, discr.loss],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}".format(time_str, step, loss))
			if writer:
				writer.add_summary(summaries, step)

		## TRAINING LOOP ##  
		#####we only could modify this later to train one discriminator on both
		if FLAGS.discr_type=="regressor":
			for batch in batches_score:
				#print(np.shape(np.concatenate(batch[:, 0]))) #64000*1000
				#print(np.reshape(np.concatenate(batch[:, 0]), (-1,1000,1000))[0]) #need batch*1000*1000
				#print(np.all(batch[10,0]==np.reshape(np.concatenate(batch[:, 0]), (-1,1000,1000))[10])) #to test it is still equal
				batch_imgs = np.reshape(np.concatenate(batch[:, 0]), (-1,1000,1000))
				batch_score = np.reshape(batch[:, 1], (-1))
				train_step(batch_imgs, batch_score) 
				current_step = tf.train.global_step(sess, global_step)
				if current_step % FLAGS.evaluate_every == 0:
					print("\nEvaluation:")
					dev_step(test_score_imgs, test_score, writer=dev_summary_writer)
					print("")
				if current_step % FLAGS.checkpoint_every == 0:
					path = saver.save(sess, checkpoint_prefix, global_step=current_step)
					print("Saved model checkpoint to {}\n".format(path))
		else:
			for batch in batches_label:
				#print(np.shape(np.concatenate(batch[:, 0]))) #64000*1000
				#print(np.reshape(np.concatenate(batch[:, 0]), (-1,1000,1000))[0]) #need batch*1000*1000
				#print(np.all(batch[10,0]==np.reshape(np.concatenate(batch[:, 0]), (-1,1000,1000))[10])) #to test it is still equal
				batch_imgs = np.reshape(np.concatenate(batch[:, 0]), (-1,1000,1000))
				batch_score = np.reshape(batch[:, 1], (-1))
				train_step(batch_imgs, batch_score) 
				current_step = tf.train.global_step(sess, global_step)
				if current_step % FLAGS.evaluate_every == 0:
					print("\nEvaluation:")
					dev_step(test_label_imgs, test_label, writer=dev_summary_writer)
					print("")
				if current_step % FLAGS.checkpoint_every == 0:
					path = saver.save(sess, checkpoint_prefix, global_step=current_step)
					print("Saved model checkpoint to {}\n".format(path))