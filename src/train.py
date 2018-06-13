from src import preprocessing
from src.model.model_skeleton import Discriminator
from config import *


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
tf.flags.DEFINE_float("dev_sample_percentage", .0001, "Percentage of the training data used for validation")
tf.flags.DEFINE_string("data_file_path", "/data/sentences.train", "Path to the training data")

# Model parameters
#tf.flags.DEFINE_integer("n_hidden", n_hidden, "Size of hidden state")


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


## DATA PREPARATION ##

# Load data
print("Loading and preprocessing training... \n")


# Saving the objects:
if os.path.isfile("var.pickle"):
	max_bytes = 2**31 - 1
	input_size = os.path.getsize("var.pickle")
	bytes_in = bytearray(0)
	with open("var.pickle", 'rb') as f_in:
		for _ in range(0, input_size, max_bytes):
			bytes_in += f_in.read(max_bytes)
	x = pickle.loads(bytes_in)

else:
	data_scored = preprocessing.load_data("scored", score_img_folder)
	imgs = np.reshape(np.array(data_scored['img'].values), (-1,1)) #dim: 9600*1000*1000
	scores = np.reshape(np.array(data_scored['scored'].values), (-1,1)) #dim: 9600
	x = np.concatenate((imgs,scores), axis = 1)


	max_bytes = 2**31 - 1
	bytes_out = pickle.dumps(x)
	n_bytes = sys.getsizeof(bytes_out)

	with open("var.pickle", 'wb') as f_out:
		for idx in range(0, n_bytes, max_bytes):

			f_out.write(bytes_out[idx:idx+max_bytes])


# data_scored = preprocessing.load_data("scored", score_img_folder)
# imgs = np.reshape(np.array(data_scored['img'].values), (-1,1)) #dim: 9600*1000*1000
# scores = np.reshape(np.array(data_scored['scored'].values), (-1,1)) #dim: 9600
# x = np.concatenate((imgs,scores), axis = 1)

print("Data loaded")

# Randomly shuffle data
np.random.seed(10)
shuffled_indices = np.random.permutation(len(x))
x_shuffled = x[shuffled_indices]

# Split train/dev sets for validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(x_shuffled[:,0])))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]

# Generate training batches
batches = preprocessing.batch_iter(x_train, FLAGS.batch_size, FLAGS.num_epochs) #have to check if the function still works with only one input x


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
		discr = Discriminator()

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
		def train_step(x_batch, s_batch):
			"""
			A single training step
			"""
			feed_dict = {
				discr.input_img: batch[:, 0],
				discr.scores: batch[:, 1]
			}
			_, step, summaries, loss = sess.run(
				[train_op, global_step, train_summary_op, discr.loss],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}".format(time_str, step, loss))
			train_summary_writer.add_summary(summaries, step)

		def dev_step(x_batch, s_batch, writer=None):
			"""
			Evaluates model on a dev set
			"""
			feed_dict = {
				discr.input_img: batch[:, 0],
				discr.scores: batch[:, 1]
				}
			step, summaries, loss = sess.run(
				[global_step, dev_summary_op, discr.loss],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}".format(time_str, step, loss))
			if writer:
				writer.add_summary(summaries, step)

		## TRAINING LOOP ##
		for batch in batches:
			train_step(batch, s_train)
			current_step = tf.train.global_step(sess, global_step)
			if current_step % FLAGS.evaluate_every == 0:
				print("\nEvaluation:")
				dev_step(x_dev, s_dev, writer=dev_summary_writer)
				print("")
			if current_step % FLAGS.checkpoint_every == 0:
				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
				print("Saved model checkpoint to {}\n".format(path))
