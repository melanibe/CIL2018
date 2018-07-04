import os
import time
import datetime
import sys
import pandas as pd
import tensorflow as tf
import numpy as np
from skimage import io

import preprocessing
from model.generator import generator
from model.discriminator_label import discriminator_label
from model.discriminator_score import discriminator_score

"""Melanie Bernhardt - Laura Manduchi - Melanie Gaillochet.

Main training file for our final 2-in-1 model. 
Run as is to train with the original parameters described in the report. 
If you wish to try other training parameters, feel free to modify them in the parameters section.
"""

##### PARAMETERS #####
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 600, "Number of training epoches")
tf.flags.DEFINE_integer("evaluate_every", 2, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("dev_sample_percentage", .01, "Percentage of the training data used for validation")

# Global tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("inter_op_parallelism_threads", 16,
	"TF nodes that perform blocking operations are enqueued on a pool of inter_op_parallelism_threads available in each process (default 0).")
tf.flags.DEFINE_integer("intra_op_parallelism_threads", 16,
	"The execution of an individual op (for some op types) can be parallelized on a pool of intra_op_parallelism_threads (default: 0).")
FLAGS = tf.flags.FLAGS


##### LOAD TRAINING DATA ####
# Scored images
data_scored = preprocessing.load_data("scored")
imgs = np.reshape(np.array(data_scored['img'].values), (-1,1)) # dim: 9600*1000*1000
scores = np.reshape(np.array(data_scored['scored'].values), (-1,1))
id = np.reshape(np.array(data_scored[['img']].index.values), (-1,1))
scored_data = np.concatenate((imgs,scores,id), axis = 1)
print("Finished the preprocessing of scored images")

# Labeled images
data_labeled = preprocessing.load_data("labeled")
imgs = np.reshape(np.array(data_labeled['img'].values), (-1,1))
labels = np.reshape(np.array(data_labeled['labeled'].values), (-1,1))
id = np.reshape(np.array(data_labeled[['img']].index.values), (-1,1))
labeled_data = np.concatenate((imgs,labels,id), axis = 1)
print("Finished the preprocessing of labeled images")


##### TRAIN / TEST SPLIT #####
# Randomly shuffle data
np.random.seed(10)
# Train/test split for score
shuffled_indices = np.random.permutation(len(scored_data[:,0]))
dev_sample_index = int(FLAGS.dev_sample_percentage * float(len(shuffled_indices)))
test_indices = shuffled_indices[:dev_sample_index]
train_indices = shuffled_indices[dev_sample_index:]
train_score = scored_data[train_indices,:]
test_scored = scored_data[test_indices,:]
# Needed for dev step - but actually we only evaluated on the first 64 (not the full 96 test set) for tensorboard.
test_score_imgs = np.reshape(np.concatenate(test_scored[:, 0]), (-1,1000,1000))
test_score = np.reshape(test_scored[:, 1], (-1))
# Generate training batches for scored images
batches_scored = preprocessing.batch_iter(train_score, FLAGS.batch_size, FLAGS.num_epochs)

# Train/test split for labels
shuffled_indices = np.random.permutation(len(labeled_data[:,0]))
dev_sample_index = int(FLAGS.dev_sample_percentage * float(len(shuffled_indices)))
test_indices = shuffled_indices[:dev_sample_index]
train_indices = shuffled_indices[dev_sample_index:]
train_label = labeled_data[train_indices,:]
test_labeled = labeled_data[test_indices,:]
# Generate training batches for labeled images
batches_labeled = preprocessing.batch_iter(train_label, FLAGS.batch_size, FLAGS.num_epochs)


##### PRINTING THE USED PARAMETERS TO THE LOG FILE #####
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value.value))
print("")


##### MODEL AND TRAINING PROCEDURE DEFINITION #####
session_conf = tf.ConfigProto(
		allow_soft_placement=True,
		log_device_placement=False)
graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement,
		inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
		intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		# Initialize model
		noise_input = tf.placeholder(tf.float32, shape=[None, 100], name="noise_input")
		true_score_images = tf.placeholder(tf.float32, shape=[None, 1000,1000], name="true_score_images")
		true_lab_images = tf.placeholder(tf.float32, shape=[None, 1000,1000], name="true_lab_images")
		true_labels = tf.placeholder(tf.int32, shape = [None], name="true_logits")
		true_scores = tf.placeholder(tf.int32, shape = [None], name="true_scores")
        
        # Get generator network definition from helper file
		output_images = generator(noise_input, FLAGS.batch_size)
		print(output_images.get_shape()) # sanity check
        
        # Get both discriminator networks definition from helper files
		logits_true = discriminator_label(true_lab_images)
		logits_fake = discriminator_label(output_images, reuse=True)
		score_true = discriminator_score(true_score_images)
		score_fake = discriminator_score(output_images, reuse=True)
        
        # Set the target labels for the generated images
		target = tf.tile([1], [FLAGS.batch_size])
		
        # Add the discriminators' losses
		loss_discr_label = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = true_labels, logits = logits_true))
		loss_discr_score = tf.losses.absolute_difference(labels=true_scores, predictions = score_true, reduction=tf.losses.Reduction.MEAN)
        
        # Add the generator loss
		loss_gen = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = target, logits = logits_fake)) + tf.reduce_mean(tf.abs(3-score_fake))
		
        # Define three optimizers with clipping gradients
		global_step = tf.Variable(0, name="global_step", trainable= False)
		optimizer = tf.train.AdamOptimizer()
		discr_vars_label = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discr_label')
		discr_vars_score = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discr')
		gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='deCNN')
		gradient_var_pairs_gen = optimizer.compute_gradients(loss_gen, var_list = gen_vars)
		gradient_var_pairs_discr_label = optimizer.compute_gradients(loss_discr_label, var_list = discr_vars_label)
		gradient_var_pairs_discr_score = optimizer.compute_gradients(loss_discr_score, var_list = discr_vars_score)
		
		# Setting the generator optimizer
		vars_gen = [x[1] for x in gradient_var_pairs_gen]
		gradients_gen = [x[0] for x in gradient_var_pairs_gen]
		clipped_gen, _ = tf.clip_by_global_norm(gradients_gen, 5)
		train_op_gen = optimizer.apply_gradients(zip(clipped_gen, vars_gen), global_step = global_step)
		
		# Setting the label discriminator optimizer
		vars_discr_label = [x[1] for x in gradient_var_pairs_discr_label]
		gradient_discr_label = [x[0] for x in gradient_var_pairs_discr_label]
		clipped_discr_label, _ = tf.clip_by_global_norm(gradient_discr_label, 5)
		train_op_discr_label = optimizer.apply_gradients(zip(clipped_discr_label, vars_discr_label), global_step = global_step)
		
		# Setting the score discriminator optimizer
		vars_discr_score = [x[1] for x in gradient_var_pairs_discr_score]
		gradient_discr_score = [x[0] for x in gradient_var_pairs_discr_score]
		clipped_discr_score, _ = tf.clip_by_global_norm(gradient_discr_score, 5)
		train_op_discr_score = optimizer.apply_gradients(zip(clipped_discr_score, vars_discr_score), global_step = global_step)		
		
		# Create output directory for models and summaries
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		print("Writing to {}\n".format(out_dir))

		# Loss summaries for tensorboard
		loss_gen_summary = tf.summary.scalar("loss_gen", loss_gen)
		loss_discr_lab_summary = tf.summary.scalar("loss_discr_label", loss_discr_label)
		loss_discr_score_summary = tf.summary.scalar("loss_discr_score", loss_discr_score)
		
		# Train summaries for tensorboard
		train_summary_op = tf.summary.merge([loss_gen_summary,loss_discr_lab_summary, loss_discr_score_summary])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
		
		# Dev summaries for tensorboard
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

		def train_step(batch_noise, batch_img_score, batch_score, batch_img_lab, batch_lab, train_gen):
			"""
			A single training step i.e. training the score discriminator, the label discriminator and if 
			necessary the generator.
			Args:
				batch_noise: input batch for the generator
				batch_img_score: input batch for the score discriminator
				batch_img_lab: input batch for the label discriminator 
				batch_score: scores corresponding the input score images
				batch_lab: labels corresponding the input label images
				train_gen(bool): whether or not to train the generator 
			"""	
			feed_dict = {noise_input: batch_noise, \
						true_score_images: batch_img_score, \
						true_labels: batch_lab, \
						true_scores: batch_score, \
						true_lab_images: batch_img_lab}
			time_str = datetime.datetime.now().isoformat()
			_, _, step, summaries, loss_discriminator_lab, loss_discriminator_score = sess.run(
				[train_op_discr_label, train_op_discr_score, global_step, train_summary_op, loss_discr_label, loss_discr_score],
				feed_dict)
			print("{}: step {}, loss discr lab {:g}; loss discr score {:g}".format(time_str, step, loss_discriminator_lab, loss_discriminator_score))		
			if train_gen:
				_, step, summaries, loss_generator = sess.run(
					[train_op_gen, global_step, train_summary_op, loss_gen],
					feed_dict)
				print("{}: step {}, loss gen {:g}".format(time_str, step, loss_generator))
			train_summary_writer.add_summary(summaries, step)

		def dev_step(batch_noise, batch_img_score, batch_score, writer=None):
			"""
			A single dev step i.e. testing the score discriminant prediction on the batch_img_score
			and saving one produced image to visualize the changes in the generated images
			during training.
			Args:
				batch_noise: input batch for the generator
				batch_img_score: input batch for the score discriminator
				batch_score: scores corresponding the input score images
			"""	
			feed_dict = {noise_input: batch_noise, true_score_images: batch_img_score, true_scores: batch_score}
			step, summaries, loss_discriminator, prod_images = sess.run(
				[global_step, dev_summary_op, loss_discr_score, output_images],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: TEST STEP step {}, loss discr score {:g}".format(time_str, step, loss_discriminator))
			curr_array = prod_images[0].astype(int)
			io.imsave(checkpoint_dir+"/step{}.png".format(step), curr_array)
			if writer:
				writer.add_summary(summaries, step)

		### TRAINING LOOP ###
		train_gen=False
		count_train = 0
		for batch_scored, batch_labeled in zip(batches_scored, batches_labeled):
			batch_imgs = np.reshape(np.concatenate(batch_scored[:, 0]), (-1,1000,1000))
			batch_score = np.reshape(batch_scored[:, 1], (-1))
			batch_imgs_lab = np.reshape(np.concatenate(batch_labeled[:, 0]), (-1,1000,1000))
			batch_lab = np.reshape(batch_labeled[:, 1], (-1))
			# input of the generator is random noise
			batch_noise = np.random.normal(0, 1, [FLAGS.batch_size, 100])
			if ((len(batch_score)==FLAGS.batch_size) and (len(batch_lab)==FLAGS.batch_size)):
				train_step(batch_noise, batch_imgs, batch_score, batch_imgs_lab, batch_lab, train_gen) 
				current_step = tf.train.global_step(sess, global_step)
				# we start training the generator only after 10000 steps for each discriminant.
				train_gen = (current_step > 10000)
				count_train +=1
			if count_train % FLAGS.checkpoint_every == 0:
				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
				print("Saved model checkpoint to {}\n".format(path))
			if count_train % FLAGS.evaluate_every == 0:
				print("\nEvaluation:")
				dev_step(batch_noise, test_score_imgs[0:FLAGS.batch_size], test_score[0:FLAGS.batch_size], writer=dev_summary_writer)
				print("")            
