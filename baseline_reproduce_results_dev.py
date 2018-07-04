import os.path
import numpy as np
import pandas as pd

import PIL.Image
import sklearn.metrics as skmet
from sklearn.externals import joblib

from config import *
from baseline_utils import csv_to_dict, extract_feats, preprocessing_baseline_train_test

""" Mélanie Bernhardt - Laura Manducchi - Mélanie Gaillochet.

Use this file to reproduce the MAE results on the dev set indicated in the report, assuming the 
related fitted estimators are saved in the /baselines_models subfolder.

Just run the file and the dev MAE for both baselines will be printed to the console.
"""

##### HELPER FUNCTION TO REPRODUCE THE RESULTS #####
def reproduce_results(model, feat_size = 10, train_ratio = 0.99):
	"""Main function to reproduce the results
		It loads the dev split and the saved estimator and predicts the score on the dev split.
		It prints the dev MAE to the console.

		Args:
			model (str): 'Ridge' or 'RandomForest' the name of the model you want to evaluate.
			feat_size(int): number of features to use for the feature matrix
			train_ratio(float between 0 and 1): train/dev split ratio that was used for training.
	"""
	_, _, test_mat, test_y = preprocessing_baseline_train_test(train_ratio, feat_size)
	try:
		fitted_filename = cwd + '/baselines_models'+ '/fitted_' + str(model) + '_' + str(train_ratio) + '_' + str(feat_size) + '.sav'
	except:
		print('You did not train this model. File not found. Launch training again')
	# Loading the saved estimator
	ml_model = joblib.load(fitted_filename)
	# Predictiing on the dev set
	print("Predicting...")
	pred_test_y = ml_model.predict(test_mat)
	# Print MAE to the console
	print("Mean absolute error (MAE) for {} on validation set is: {}".format(model, skmet.mean_absolute_error(test_y, pred_test_y)))


##### MAIN PRITING THE RESULTS FROM THE REPORT #####
if __name__ == "__main__":
	reproduce_results('RandomForest')
	reproduce_results('Ridge')