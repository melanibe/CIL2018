from src import preprocessing
from config import *
from src.model.model_skeleton import *

import tensorflow as tf
import os

"""
To predict
"""

run_number =1529046033




## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_string("data_file_path", test_path, "Path to the test data. This data should be distinct from the training data.")
tf.flags.DEFINE_integer("train run number", run_number, "")

# Test parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/{}/checkpoints/".format(run_number), "Checkpoint directory from training run")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value.value))
print("")

## DATA PREPARATION ##

# Load data

## EVALUATION ##

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():

		# Load the saved meta graph and restore variables
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)
		print("Model restored.")

		# Getting names of all variables #Can remove that afterwords
		for op in graph.get_operations():
			print(op.name)
		# print([n.name for n in tf.get_default_graph().as_graph_def().node] if "Variable" in n.op)

		# Get the placeholders from the graph by name
		input_img = graph.get_operation_by_name("input_img").outputs[0]
		input = tf.expand_dims(input_img, 3) #taken from model_skeleton
		print(input)

		# Tensors we want to evaluate
		scores = graph.get_operation_by_name("output/score_pred/Relu").outputs[0]
		print(scores)

		# Load test data
		data_scored = preprocessing.load_data("scored")
		imgs = np.reshape(np.array(data_scored['img'].values), (-1, 1))  # dim: 9600*1000*1000


		# Generate batches for one epoch
		batches = preprocessing.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

		""" Output directory for models and summaries """
		# timestamp = str(run_number)
		out_dir = os.path.abspath(os.path.join(os.path.curdir, runs_dir))
		
		# # Create the directory perplexities
		# if not os.path.exists(out_dir):
		# 	os.makedirs(out_dir)

		#out_dir = os.path.abspath(os.path.join(out_dir, timestamp))
		#for final submission
		out_dir = os.path.abspath(os.path.join(out_dir, "tochange".format(exp)))
		print("Writing to {}\n".format(out_dir))

		with open("{}.txt".format(out_dir),"w") as file:
			for x_test_batch in batches: # x_test_batch dim: batch_size * 30
				batch_scores = sess.run(scores, {input: x_test_batch})
				for i in range(len(batch_scores)):
					# Write perplexity in ./perplexities/
					file.write('{}\n'.format(batch_scores[i]))
