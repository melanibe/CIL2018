import preprocessing
from model import model_skeleton
import numpy as np
import tensorflow as tf
import os

"""
To predict
"""
cwd = os.getcwd()
## PARAMETERS ##
run_number = 1529416507
# Data loading parameters
#tf.flags.DEFINE_string("data_file_path", "/data/sentences_test.txt", "Path to the test data. This data should be distinct from the training data.")
tf.flags.DEFINE_integer("train run number", run_number, "")
# Test parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/{}/checkpoints/".format(run_number), "Checkpoint directory from training run") #for cluster
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
data_query = preprocessing.load_data("query")
imgs = np.reshape(np.array(data_query['img'].values), (-1,1))
id = np.reshape(np.array(data_query.index.values), (-1,1))
query_data = np.concatenate((imgs,id), axis = 1)

## EVALUATION ##
#checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
checkpoint_file = cwd+"/runs/{}/checkpoints/model-18900".format(run_number) #for running test locally with model from cluster
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
		input = graph.get_operation_by_name("input_img").outputs[0]

		# Tensors we want to evaluate
		scores = graph.get_operation_by_name("output/score_pred").outputs[0] #this has to be changed to output/score_pred with the new version of model

		# Generate batches for one epoch
		batches = preprocessing.batch_iter(query_data, FLAGS.batch_size, 1, shuffle=False)

		timestamp = str(run_number)
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "predictions"))
		
		# Create the directory perplexities
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		#out_dir = os.path.abspath(os.path.join(out_dir, timestamp))
		#for final submission
		out_file = os.path.abspath(os.path.join(out_dir, "{}".format(timestamp)))
		print("Writing to {}\n".format(out_file))

		with open("{}.csv".format(out_file),"w") as file:
			file.write('Id,Predicted\n') #create the header required
			for test_batch in batches: #
				batch_imgs = np.reshape(np.concatenate(test_batch[:, 0]), (-1,1000,1000))
				batch_id = np.reshape(test_batch[:, 1], (-1))
				batch_scores = sess.run(scores, {input: batch_imgs})
				for i in range(len(batch_scores)):
					# Write perplexity in ./perplexities/
					tmp = max(0.0, min(8.0, batch_scores[i]))
					file.write("{},{}\n".format(batch_id[i], tmp)) #csv id, score