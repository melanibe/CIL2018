import preprocessing
from model.generator2 import generator
from model.discriminator_label import discriminator_label
from model.discriminator import discriminator
from config import *
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import sys
from skimage import io
"""Launch file for training tasks
"""
## PARAMETERS ##

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 250, "Number of training epoches")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 500)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("dev_sample_percentage", .01, "Percentage of the training data used for validation")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# for running on EULER
tf.flags.DEFINE_integer("inter_op_parallelism_threads", 16,
	"TF nodes that perform blocking operations are enqueued on a pool of inter_op_parallelism_threads available in each process (default 0).")
tf.flags.DEFINE_integer("intra_op_parallelism_threads", 16,
	"The execution of an individual op (for some op types) can be parallelized on a pool of intra_op_parallelism_threads (default: 0).")

FLAGS = tf.flags.FLAGS

#### get score data ###

data_scored = preprocessing.load_data("scored", augmented=False)
imgs = np.reshape(np.array(data_scored['img'].values), (-1,1)) #dim: 9600*1000*1000
scores = np.reshape(np.array(data_scored['scored'].values), (-1,1)) #dim: 9600
id = np.reshape(np.array(data_scored[['img']].index.values), (-1,1))
scored_data = np.concatenate((imgs,scores,id), axis = 1)
print("finished the preprocessing")
data_labeled = preprocessing.load_data("labeled",augmented=False)
imgs = np.reshape(np.array(data_labeled['img'].values), (-1,1)) #dim: 9600*1000*1000
labels = np.reshape(np.array(data_labeled['labeled'].values), (-1,1)) #dim: 9600
id = np.reshape(np.array(data_labeled[['img']].index.values), (-1,1))
labeled_data = np.concatenate((imgs,labels,id), axis = 1)

###
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
# Generate training batches 
batches_scored = preprocessing.batch_iter(train_score, FLAGS.batch_size, FLAGS.num_epochs)


shuffled_indices = np.random.permutation(len(labeled_data[:,0]))
dev_sample_index = int(FLAGS.dev_sample_percentage * float(len(shuffled_indices)))
test_indices = shuffled_indices[:dev_sample_index]
train_indices = shuffled_indices[dev_sample_index:]
train_label = labeled_data[train_indices,:]
test_labeled = labeled_data[test_indices,:]
#needed for dev step
test_label_imgs = np.reshape(np.concatenate(test_labeled[:, 0]), (-1,1000,1000))
test_label = np.reshape(test_labeled[:, 1], (-1))
# Generate training batches 
batches_labeled = preprocessing.batch_iter(train_label, FLAGS.batch_size, FLAGS.num_epochs)
###


print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value.value))
print("")


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
		noise_input = tf.placeholder(tf.float32, shape=[None, 100], name="noise_input")
		true_images = tf.placeholder(tf.float32, shape=[None, 1000,1000], name="true_images")
		true_lab_images = tf.placeholder(tf.float32, shape=[None, 1000,1000], name="true_lab_images")
		true_labels = tf.placeholder(tf.int32, shape = [None], name="true_logits")
		true_scores = tf.placeholder(tf.int32, shape = [None], name="true_scores")
		output_images = generator(noise_input, FLAGS.batch_size)
		print(output_images.get_shape())
		logits_true = discriminator_label(true_lab_images)
		logits_fake = discriminator_label(output_images, reuse=True)
		score_true = discriminator(true_images)
		score_fake = discriminator(output_images, reuse=True)
		target = tf.tile([1], [FLAGS.batch_size])
		# Add the discriminator
		#class_scores_fake = tf.minimum(2.0, score_fake)
		# Defining the loss - objective score is 2.0
		
		loss_discr_label = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = true_labels, logits = logits_true))
		loss_discr_score = tf.losses.absolute_difference(labels=true_scores, predictions = score_true, reduction=tf.losses.Reduction.MEAN)
		#loss_gen = tf.reduce_mean(tf.abs(2-class_scores_fake))
		#var_gen = tf.reduce_mean(tf.reshape(tf.nn.moments(output_images, axes=0)[1], [-1,1]))
		#loss_gen = tf.losses.absolute_difference(labels=true_scores, predictions = score_fake, reduction=tf.losses.Reduction.MEAN)+tf.abs(var_gen-1)
		loss_gen = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = target, logits = logits_fake)) + tf.reduce_mean(tf.abs(3-score_fake))
		# Define an optimizer with clipping the gradients
		global_step = tf.Variable(0, name="global_step", trainable= False)
		optimizer = tf.train.AdamOptimizer()
		discr_vars_label = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discr_label')
		discr_vars_score = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discr')
		gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='deCNN')
		gradient_var_pairs_gen = optimizer.compute_gradients(loss_gen, var_list = gen_vars)
		gradient_var_pairs_discr_label = optimizer.compute_gradients(loss_discr_label, var_list = discr_vars_label)
		gradient_var_pairs_discr_score = optimizer.compute_gradients(loss_discr_score, var_list = discr_vars_score)
		vars_gen = [x[1] for x in gradient_var_pairs_gen]
		gradients_gen = [x[0] for x in gradient_var_pairs_gen]
		clipped_gen, _ = tf.clip_by_global_norm(gradients_gen, 5)
		train_op_gen = optimizer.apply_gradients(zip(clipped_gen, vars_gen), global_step = global_step)
		vars_discr_label = [x[1] for x in gradient_var_pairs_discr_label]
		gradient_discr_label = [x[0] for x in gradient_var_pairs_discr_label]
		clipped_discr_label, _ = tf.clip_by_global_norm(gradient_discr_label, 5)
		train_op_discr_label = optimizer.apply_gradients(zip(clipped_discr_label, vars_discr_label), global_step = global_step)
		vars_discr_score = [x[1] for x in gradient_var_pairs_discr_score]
		gradient_discr_score = [x[0] for x in gradient_var_pairs_discr_score]
		clipped_discr_score, _ = tf.clip_by_global_norm(gradient_discr_score, 5)
		train_op_discr_score = optimizer.apply_gradients(zip(clipped_discr_score, vars_discr_score), global_step = global_step)		
		# Output directory for models and summaries
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_gen", timestamp))
		print("Writing to {}\n".format(out_dir))

		# Loss summaries
		loss_gen_summary = tf.summary.scalar("loss_gen", loss_gen)
		loss_discr_lab_summary = tf.summary.scalar("loss_discr_label", loss_discr_label)
		loss_discr_score_summary = tf.summary.scalar("loss_discr_score", loss_discr_score)
		# Train summaries
		train_summary_op = tf.summary.merge([loss_gen_summary,loss_discr_lab_summary, loss_discr_score_summary])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
		# Dev summaries
		dev_summary_op = tf.summary.merge([loss_discr_score_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

		# Initialize all variables
		sess.run(tf.global_variables_initializer())

		# Define training and dev steps (batch)
		def train_step(batch_noise, batch_imgs, batch_score, batch_img_lab, batch_lab):
			"""
			A single training step
			"""	
			time_str = datetime.datetime.now().isoformat()
			feed_dict = {noise_input: batch_noise, true_images: batch_imgs, true_labels: batch_lab, true_scores: batch_score, true_lab_images: batch_img_lab}
			_, _, step, summaries, loss_discriminator_lab, loss_discriminator_score = sess.run(
				[train_op_discr_label, train_op_discr_score, global_step, train_summary_op, loss_discr_label, loss_discr_score],
				feed_dict)
			print("{}: step {}, loss discr lab {:g}; loss discr score {:g}".format(time_str, step, loss_discriminator_lab, loss_discriminator_score))		
			_, step, summaries, loss_generator = sess.run(
				[train_op_gen, global_step, train_summary_op, loss_gen],
				feed_dict)
			print("{}: step {}, loss gen {:g}".format(time_str, step, loss_generator))
			train_summary_writer.add_summary(summaries, step)

		def dev_step(batch_noise, batch_imgs, batch_score, writer=None):
			feed_dict = {noise_input: batch_noise, true_images: batch_imgs, true_scores: batch_score}
			step, summaries, loss_discriminator, prod_images = sess.run(
				[global_step, dev_summary_op, loss_discr_score, output_images],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: TEST STEP step {}, loss discr score {:g}".format(time_str, step, loss_discriminator))
			curr_array = prod_images[0].astype(int)
			io.imsave(checkpoint_dir+"/step{}.png".format(step), curr_array)
			if writer:
				writer.add_summary(summaries, step)

		## TRAINING LOOP ##  
		for batch_scored, batch_labeled in zip(batches_scored, batches_labeled):
			batch_imgs = np.reshape(np.concatenate(batch_scored[:, 0]), (-1,1000,1000))
			batch_score = np.reshape(batch_scored[:, 1], (-1))
			batch_imgs_lab = np.reshape(np.concatenate(batch_labeled[:, 0]), (-1,1000,1000))
			batch_lab = np.reshape(batch_labeled[:, 1], (-1))
			batch_noise = np.random.normal(0, 1, [FLAGS.batch_size, 100]) #input is random noise
			if ((len(batch_score)==FLAGS.batch_size) and (len(batch_lab)==FLAGS.batch_size)):
				train_step(batch_noise, batch_imgs, batch_score, batch_imgs_lab, batch_lab) 
				current_step = tf.train.global_step(sess, global_step)
			if current_step % FLAGS.checkpoint_every == 0:
				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
				print("Saved model checkpoint to {}\n".format(path))
			if current_step % FLAGS.evaluate_every == 0:
				print("\nEvaluation:")
				dev_step(batch_noise, test_score_imgs[0:FLAGS.batch_size], test_score[0:FLAGS.batch_size], writer=dev_summary_writer)
				print("")            
