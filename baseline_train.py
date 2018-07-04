import os.path
import csv
import numpy as np
import pandas as pd
import PIL.Image

import sklearn.linear_model as sklm
import sklearn.model_selection as skms
import sklearn.ensemble as sken
import sklearn.metrics as skmet
from sklearn.externals import joblib

from config import *
from baseline_utils import csv_to_dict, extract_feats, preprocessing_baseline_train_test

""" Mélanie Bernhardt - Laura Manducchi - Mélanie Gaillochet.

Use this file to train the baseline estimators. 

You can choose between 'Ridge' and 'RandomForest' for the model to build.
You can modify the number of features to use for the feature matrix (it defaults to 10 parameter used for the
experiments in the report). You can also modify the train/dev split ratio (it defaults to 0.99% training split, parameter used for the
experiments in the report).

The resulting estimator will be saved in the /baselines_models/ subfolder of the current directory. 
The MAE on the dev set will be printed to the console at the end of the run.
"""

##### PARAMETERS (TO MODIFY IF NEEDED) #####
model = 'RandomForest' # or 'Ridge'
feat_size = 10
train_ratio = 0.99

##### HELPER FUNCITON TO BUILD THE ESTIMATORS ######
def regressor(scorer, model='Ridge'):
	"""Defines the baseline estimators used in the report.
	Args:
		scorer (callable): score function used to choose the best model in GridSearch
		model (str): 'Ridge' or 'RandomForest' the name of the model you want to train.
	Returns:
		ml_model: initialized sklearn estimator to fit.
	"""
	if model=='Ridge':
		base_model = sklm.Ridge()
		ml_model = skms.GridSearchCV(base_model, {"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]},
									 verbose=5,
									 scoring=scorer)

	elif model=='RandomForest':
		base_model = sken.RandomForestRegressor(criterion="mae", max_features=None, oob_score=True,
												random_state = 10)
		ml_model = skms.GridSearchCV(base_model, {"n_estimators": [5, 10, 50, 100]},
									 verbose=5,
									 scoring=scorer)
	return ml_model

##### REGRESSION TRAINING #####
if __name__ == "__main__":
	# Fitting the model
	train_mat, train_y, test_mat, test_y = preprocessing_baseline_train_test(train_ratio, feat_size)
	ml_model = regressor(model=model, scorer='neg_mean_absolute_error')
	print("Fitting...")
	ml_model.fit(train_mat, train_y)
	print("Best params: {}".format(ml_model.best_estimator_))
	# Saving model
	fitted_filename = cwd + '/baselines_models'+ '/fitted_' + str(model) + '_' + str(train_ratio) + '_' + str(feat_size) + '.sav'
	print(fitted_filename)
	joblib.dump(ml_model, fitted_filename)

	## PREDICTION ON DEV SET TAKEN FROM TRAINING SET ##
	ml_model = joblib.load(fitted_filename)
	# Predictions
	print("Predicting...")
	pred_test_y = ml_model.predict(test_mat)
	prediction = pd.DataFrame({'predictions': pred_test_y, 'true_value': test_y})
	print(prediction)	
	# Classification diagnostics
	print("Mean absolute error (MAE): {}".format(skmet.mean_absolute_error(test_y, pred_test_y)))
