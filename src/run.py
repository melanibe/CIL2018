import sys
import time
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import model files from project
from model import model_skeleton

"""
To run on terminal: $ python run.py -m modelname -t (to train) 
					$ python run.py -m modelname -p (to predict) 
"""


def _setup_argparser():
	"""Sets up the argument parser and returns the arguments.

	Returns:
		argparse.Namespace: The command line arguments.
	"""
	parser = argparse.ArgumentParser(description="Control program to launch all actions related to this project.")
	parser.add_argument("-m", "--model", action="store",
						choices=["model_skeleton", "model2"], # TODO Add your models
						default="model_skeleton", # TODO Modify once we have model
						type=str,
						help="the model to be used, defaults to .....")
	parser.add_argument("-t", "--train",
						help="train the given model",
						action="store_true")
	parser.add_argument("-p", "--predict",
						help="predict on a test set given the model",
						action="store_true")
	parser.add_argument("-e", "--evaluate",
						help="evaluate on a test set given the model",
						action="store_true")
	args, unknown = parser.parse_known_args()
	return args


def get_latest_model():
	"""
	Returns the latest directory of the model specified in the arguments.
	:return: path to the directory
	"""
	__file__ = "run.py"
	file_path = os.path.dirname(os.path.abspath(__file__))

	if not os.path.exists(os.path.join(file_path, "..", "trained_models", args.model)):
		print("No trained model {} exists.".format(args.model))
		sys.exit(1)
	# Path to the model
	res = os.path.join(file_path, "..", "trained_models", args.model)
	all_runs = [os.path.join(res, o) for o in os.listdir(res) if os.path.isdir(os.path.join(res, o))]
	res = max(all_runs, key=os.path.getmtime)
	print("Retrieving trained model from {}".format(res))
	return res


def get_submission_filename():
	"""
	:return:  (path to directory) + filename of the submission file.
	"""
	ts = int(time.time())
	submission_filename = "submission_" + str(args.model) + "_" + str(ts) + ".csv"
	submission_path_filename = os.path.join(get_latest_model(), submission_filename)
	return submission_path_filename



if __name__ == "__main__":
	__file__ = "run.py"
	file_path = os.path.dirname(os.path.abspath(__file__))

	args = _setup_argparser()
	out_trained_models = ""
	# Once we get ready we can decomment. This avoids creating files when things have still to be debugged
	# if args.train:
	#     out_trained_models = os.path.normpath(os.path.join(file_path,
	#                                                        "../trained_models/",
	#                                                        args.model,
	#                                                        datetime.datetime.now().strftime(r"%Y-%m-%d[%Hh%M]")))
	#     try:
	#         os.makedirs(out_trained_models)
	#     except OSError:
	#         pass
	# else:
	#     out_trained_models = os.path.join(os.path.abspath("run.py"), "..", "trained_models", args.model)
	#
	# print("Trained model will be saved in ", out_trained_models)

	if args.train:
		"""Create a field with the model"""
		if args.model == "model_skeleton":
			print("...")

			# Initialize the model
			model = model_skeleton.Discriminator()
			# model.train(save_path = out_trained_models)

		elif args.model == "model2":
			print("...")

	if args.predict:
		"""Path to the model to restore for predictions -> be sure you save the model as model.h5"""
		model_path = os.path.join(get_latest_model(), "model.h5")
		"""Submission file will be in the same folder as the model restored to predict
		   e.g trained_model/27_05_2012.../submission_modelname...."""
		submission_path_filename = get_submission_filename()

		if args.model == "model1":
			print("...")

		elif args.model == "model2":
			print("...")
