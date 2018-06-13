from src import preprocessing
from src.model.model_skeleton import ####

import tensorflow as tf
import os

"""
To predict
"""

## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_string("data_file_path", "/data/sentences_test.txt", "Path to the test data. This data should be distinct from the training data.")
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

		# Get the placeholders from the graph by name
		input = graph.get_operation_by_name("input_img").outputs[0]

		# Tensors we want to evaluate
		scores = graph.get_operation_by_name("output/score_pred").outputs[0]

		# Generate batches for one epoch
		batches = preprocessing.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

		timestamp = str(run_number)
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "perplexities"))
		
		# Create the directory perplexities
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

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
