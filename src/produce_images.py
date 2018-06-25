import preprocessing
from model import model_skeleton_bis
import numpy as np
import tensorflow as tf
import os
from skimage import io
"""
To produce the images
"""
cwd = os.getcwd()
## PARAMETERS ##
run_number = 1529931827
# Data loading parameters
#tf.flags.DEFINE_string("data_file_path", "/data/sentences_test.txt", "Path to the test data. This data should be distinct from the training data.")
tf.flags.DEFINE_integer("train run number", run_number, "")
# Test parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs_gen/{}/checkpoints/".format(run_number), "Checkpoint directory from training run") #for cluster
# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value.value))
print("")

## DATA PREPARATION ##

## EVALUATION ##
#checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
checkpoint_file = cwd+"/runs_gen/{}/checkpoints/model-800".format(run_number) #for running test locally with model from cluster
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
		input = graph.get_operation_by_name("noise_input").outputs[0]

		# Tensors we want to evaluate
		output_img  = graph.get_operation_by_name("deCNN/output_deconv/output_images").outputs[0] #this has to be changed to output/score_pred with the new version of model
	
		# Create the directory produced
		timestamp = str(run_number)
		out_dir = cwd+"/produced/{}/".format(run_number)
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		for i in range(1): 
			input_batch = np.random.normal(-1, 1, [FLAGS.batch_size, 100])
			produced_arrays = np.reshape(sess.run([output_img], {input: input_batch}), (FLAGS.batch_size, 1000,1000))
			print(np.shape(produced_arrays))
			for j in range(len(produced_arrays)):
				curr_array = np.maximum(0,np.minimum(255,produced_arrays[j])).astype(int)
				io.imsave(cwd+"/produced/{}/{}.png".format(run_number,j), curr_array)
