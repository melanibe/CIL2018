import numpy as np
import tensorflow as tf
import os

import preprocessing

""" Mélanie Bernhardt - Laura Manducchi - Mélanie Gaillochet.

Use this file to reproduce the MAE computation from the report on our final 2-in-1 model.
You do not need to specify any parameter. 
Results are printed to the console.

Note: to reproduce the experiment results for the baselines use the `reproduce_baseline_results_dev.py` file. 
"""

##### SET UP #####
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

##### DATA PREPARATION #####
# Load scored data
data_scored = preprocessing.load_data("scored")
imgs = np.reshape(np.array(data_scored['img'].values), (-1,1)) # dim: 9600*1000*1000
scores = np.reshape(np.array(data_scored['scored'].values), (-1,1))
id = np.reshape(np.array(data_scored[['img']].index.values), (-1,1))
scored_data = np.concatenate((imgs,scores,id), axis = 1)
print("finished the preprocessing")
# Randomly shuffle data - set the seed for reproducibility
np.random.seed(10)
# Separating train/test split for score (as defined in the model training)
shuffled_indices = np.random.permutation(len(scored_data[:,0]))
dev_sample_index = int(0.1 * float(len(shuffled_indices)))
test_indices = shuffled_indices[:dev_sample_index]
test_scored = scored_data[test_indices,:]
test_score_imgs = np.reshape(np.concatenate(test_scored[:, 0]), (-1,1000,1000))
test_score = np.reshape(test_scored[:, 1], (-1))


##### EVALUATION FUNCTION ##### 
def evalute_dev_set(run_number, model_number):
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
			scores = graph.get_operation_by_name("discr/output/score_pred").outputs[0] #this has to be changed to output/score_pred with the new version of model
			# Generate batches for evaluation
			batches = preprocessing.batch_iter(scored_data, FLAGS.batch_size, 1, shuffle=False)
			# Predicting the score from the model
			acc=0
			n=0
			for test_batch in batches: 
				batch_imgs = np.reshape(np.concatenate(test_batch[:, 0]), (-1,1000,1000))
				batch_score = np.reshape(test_batch[:, 1], (-1))
				batch_pred_scores = sess.run(scores, {input: batch_imgs})
				for i in range(len(batch_pred_scores)):
					tmp = max(0.0, min(8.0, batch_pred_scores[i]))
					acc += abs(tmp - batch_score[i])
					n+=1
			# Return the MAE on the dev set.
			return(acc/n) 


##### OUTPUTTING THE RESULTS FROM THE REPORT #####
if __name__ == "__main__":
	print("The MAE on the dev split for the final 2-in-1 model is: {}".format(evalute_dev_set(1530273051, 22088)))