import numpy as np
import tensorflow as tf
import os

import preprocessing

"""Melanie Bernhardt - Laura Manduchi - Melanie Gaillochet.

Running this file creates a csv file containing the predicted score associated to a
particular training run of our final model. 

To choose the training model for which you want to predict the score, please specify run_number (name of the corresponding 
subfolder containing the checkpoints) and model_number (specify the number of the .meta file) in the 2 first lines of the file.

Parameter to use to predict from our final model are:
    run_number: 1530273051
    model_number: 22088

Results are placed in the predictions subfolder of the root folder, the train run_number is the name of the csv output file.
"""

##### SET UP #####
cwd = os.getcwd()

# Choose model
tf.flags.DEFINE_integer("run_number", 1530273051, "Run number (default: 1530273051)")
tf.flags.DEFINE_integer("model_number", 22088, "Model number (default: 22088)")

# Tensorflow Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value.value))
print("")

##### QUERY DATA LOADING #####
data_query = preprocessing.load_data("query")
imgs = np.reshape(np.array(data_query[['img']].values), (-1,1))
id = np.reshape(np.array(data_query[['img']].index.values), (-1,1))
query_data = np.concatenate((imgs,id), axis = 1)

##### PREDICTION #####
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
		input = graph.get_operation_by_name("true_images").outputs[0]

		# Tensors we want to evaluate
		scores = graph.get_operation_by_name("discr/output/score_pred").outputs[0]
		
        # Generate batches from the query dataset for one epoch
		batches = preprocessing.batch_iter(query_data, FLAGS.batch_size, 1, shuffle=False)
		
        # If necessary Create the directory predictions
		timestamp = str(run_number)
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "predictions"))	
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		
        # Defining the csv file name
		out_file = os.path.abspath(os.path.join(out_dir, "{}".format(timestamp)))
		print("Writing to {}\n".format(out_file))

		with open("{}.csv".format(out_file),"w") as file:
            # Create the required header
			file.write('Id,Predicted\n') 
			for test_batch in batches: 
				batch_imgs = np.reshape(np.concatenate(test_batch[:, 0]), (-1,1000,1000))
				batch_id = np.reshape(test_batch[:, 1], (-1))
				batch_scores = sess.run(scores, {input: batch_imgs})
				for i in range(len(batch_scores)):
					print(batch_scores[i])
					tmp = max(0.0, min(8.0, batch_scores[i]))
					file.write("{},{}\n".format(batch_id[i], tmp)) # csv id, score
