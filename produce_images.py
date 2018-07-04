import numpy as np
import tensorflow as tf
import os
import scipy
from skimage import io

import preprocessing

""" Melanie Bernhardt - Laura Manduchi - Melanie Gaillochet.

Use this file to generate 100 images with a predicted score above 2.5 from a saved tensorflow checkpoint. 
Required parameters (default parameters correspond to our final Kaggle model): 
	- run_number : folder number of the training run
	- model_number : model number i.e. identifier of the .meta file to use in the chosen folder.
The produced images are placed in the "/produced/run_number/" subfolder of the current directory.
"""

##### PARAMETERS TO ENTER #####
run_number = 1530273051
model_number = 22088 


##### AUTOMATIC SETTING OF OTHER PARAMETERS #####
cwd = os.getcwd()
# Tensorflow Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value.value))
print("")


##### LOADING THE SAVED MODEL #####
checkpoint_file = cwd+"/runs/{}/checkpoints/model-{}".format(run_number, model_number)
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
		true_input = graph.get_operation_by_name("true_images").outputs[0]
		
        # Tensors we want to evaluate with the saved model
		output_img  = graph.get_operation_by_name("deCNN/output_deconv/mul").outputs[0] #this has to be changed to output/score_pred with the new version of model
		fake_img_score =   graph.get_operation_by_name("discr_1/output/score_pred").outputs[0]
		
		# Create the directory produced
		timestamp = str(run_number)
		out_dir = cwd+"/produced/{}/".format(run_number)
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		
		# Generate and save 100 images from the model.
		k=0
		while k<100:
			input_batch = np.random.normal(0, 1, [FLAGS.batch_size, 100])
			produced_arrays, pred_scores = sess.run([output_img, fake_img_score], {input: input_batch})
			print(np.shape(produced_arrays))
			for j in range(len(produced_arrays)):
                # only save good images produced by the model
				if ((pred_scores[j]>3.0) and (k<100)): 
                    # sanity check in the console
					print(pred_scores[j]) 
                    # saving the image
					curr_array = np.maximum(0,np.minimum(255,produced_arrays[j])).astype(int) # to ensure validity of conversion to png
					io.imsave(cwd+"/produced/{}/{}.png".format(run_number,k),curr_array)
					k = k+1
