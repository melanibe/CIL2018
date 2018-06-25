import preprocessing
from model.model_skeleton_gen import generator
from config import *
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import sys
from freeze import freeze_graph
import pickle

"""Launch file for training tasks
"""
run_number = 1529441275

## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_string("checkpoint_dir", "./runs/{}/checkpoints/".format(run_number), "Checkpoint directory from training run")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_steps", 2000, "Number of training esteps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")

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

############## FREEZING DISCRIMINATOR #####################
orig_checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir) # for cluster
#orig_checkpoint_file = cwd+"/runs/{}/checkpoints/model-14900".format(run_number)
freeze_graph(FLAGS.checkpoint_dir, orig_checkpoint_file, "output/score_pred,input_img") # saving freezed 
freeze_checkpoint_file = cwd+"/runs/{}/frozen/frozen_model".format(run_number)



################# MODEL AND TRAINING PROCEDURE DEFINITION #############
session_conf = tf.ConfigProto(
		allow_soft_placement=True,
		log_device_placement=False)
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
		writer = tf.summary.FileWriter(cwd, sess.graph)
		noise_input = tf.placeholder(tf.float32, shape=[None, 25*25*5])
		output_images = generator(noise_input)
		# Add the discriminator
		saver = tf.train.import_meta_graph("{}.meta".format(freeze_checkpoint_file), input_map={'input_img': output_images}) #b/c need a tensor input
		saver.restore(sess, freeze_checkpoint_file)
		input = graph.get_operation_by_name("input_img").outputs[0]
		scores = graph.get_operation_by_name("output/score_pred").outputs[0]
		
		# Defining the loss - objective score is 8.0
		loss_gen = tf.reduce_mean(tf.abs(8-scores))

		# Define an optimizer with clipping the gradients
		global_step = tf.Variable(0, name="global_step", trainable= False)
		optimizer = tf.train.AdamOptimizer()
		gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='deCNN')
		gradient_var_pairs = optimizer.compute_gradients(loss_gen, var_list=gen_vars)
		vars = [x[1] for x in gradient_var_pairs]
		gradients = [x[0] for x in gradient_var_pairs]
		clipped, _ = tf.clip_by_global_norm(gradients, 5)
		train_op = optimizer.apply_gradients(zip(clipped, vars), global_step = global_step)
		
		# Output directory for models and summaries
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_gen", timestamp))
		print("Writing to {}\n".format(out_dir))

		# Loss summaries
		loss_summary = tf.summary.scalar("loss", loss_gen)

		# Train summaries
		train_summary_op = tf.summary.merge([loss_summary])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

		# Initialize all variables
		sess.run(tf.global_variables_initializer())

		# Define training and dev steps (batch)
		def train_step(batch):
			"""
			A single training step
			"""	
			feed_dict = {noise_input: z}		
			_, step, summaries, loss = sess.run(
				[train_op, global_step, train_summary_op, loss_gen],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}".format(time_str, step, loss))
			train_summary_writer.add_summary(summaries, step)


		## TRAINING LOOP ##  
		for i in range(FLAGS.num_steps):
			z = np.random.normal(-1, 10, [FLAGS.batch_size, 25*25*5]) #input is random noise
			train_step(z) 
			current_step = tf.train.global_step(sess, global_step)
			if current_step % FLAGS.checkpoint_every == 0:
				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
				print("Saved model checkpoint to {}\n".format(path))