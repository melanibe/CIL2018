import os.path
import random
import csv

import PIL.Image
import numpy as np
import pandas as pd
import sklearn.linear_model as sklm
import sklearn.model_selection as skms
import sklearn.ensemble as sken
import sklearn.metrics as skmet
from sklearn.externals import joblib

from config import *


def csv_to_dict(csv_path):
	with open(csv_path, 'r') as fp:
		csv_fp = csv.reader(fp)
		next(csv_fp)
		d = dict(filter(None, csv_fp))
		return d


def extract_feats(img_arr, bins):
	hist, _ = np.histogram(img_arr, bins=bins)
	return hist


def regressor(scorer, model='Ridge'):
	if model=='Ridge':
		base_model = sklm.Ridge()
		ml_model = skms.GridSearchCV(base_model, {"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]},
									 verbose=5,
									 scoring=scorer)

	elif model=='RandomForest':
		base_model = sken.RandomForestRegressor(criterion="mae", max_features=None, oob_score=True,
												random_state = 42)
		ml_model = skms.GridSearchCV(base_model, {"n_estimators": [5, 10, 50, 100]},
									 verbose=5,
									 scoring=scorer)
	return ml_model


if __name__ == "__main__":

	model = 'Ridge'
	feat_size = 50

	# Get preprocessed data if available
	try:
		train_mat = np.load(data_folder + 'train_mat_' + str(feat_size) + '.npy')
		train_y = np.load(data_folder + 'train_y_' + str(feat_size) + '.npy')
		test_mat = np.load(data_folder + 'test_mat_' + str(feat_size) + '.npy')
		test_y = np.load(data_folder + 'test_y_' + str(feat_size) + '.npy')

	# If preprocessed data doesn't exist, do it
	except:
		print("Couldn't find matrices; preprocessing data")

		# Parameters
		train_ratio = 0.7
		print('feature_size: {}'.format(feat_size))

		# Paths
		scored_path = os.path.join(data_folder, "scored")
		score_file = os.path.join(data_folder, "scored.csv")

		# Initialization
		score_dict = csv_to_dict(score_file) # Image number:(0 or 1)
		img_prefixes = list(score_dict.keys()) # Image number (name)
		random.shuffle(img_prefixes)
		n_train = int(train_ratio * len(img_prefixes))

		print("n train: {}".format(n_train))
		n_test = len(img_prefixes) - n_train
		print("n test: {}".format(n_test))
		train_mat = np.zeros((n_train, feat_size))
		train_y = np.zeros(n_train)
		test_mat = np.zeros((n_test, feat_size))
		test_y = np.zeros(n_test)
		train_idx = 0
		test_idx = 0

	###### FEATURE EXTRACTION ######
		# Assemble train/test feature matrices / score vectors
		for idx, img_prefix in enumerate(img_prefixes):
			print("Image: {}/{}".format(idx + 1, len(img_prefixes)))
			raw_image = PIL.Image.open(os.path.join(scored_path, "{}.png".format(img_prefix)))
			img_arr = np.array(raw_image.getdata()).reshape(raw_image.size[0], raw_image.size[1]).astype(np.uint8)
			img_feats = extract_feats(img_arr=img_arr, bins=feat_size)
			score = float(score_dict[img_prefix])

			if idx < n_train:
				train_mat[train_idx, :] = img_feats
				train_y[train_idx] = score
				train_idx += 1

			else:
				test_mat[test_idx, :] = img_feats
				test_y[test_idx] = score
				test_idx += 1

		# Saving features/scores to disk
		print("Saving feature matrices...")
		np.save(data_folder + 'train_mat_' + str(feat_size) , train_mat)
		np.save(data_folder + 'train_y_' + str(feat_size), train_y)
		np.save(data_folder + 'test_mat_' + str(feat_size), test_mat)
		np.save(data_folder + 'test_y_' + str(feat_size), test_y)


#### REGRESSION TRAINING ######

	ml_model = regressor(model=model, scorer='neg_mean_absolute_error')

	print("Fitting...")
	ml_model.fit(train_mat, train_y)

	# print("Best params: {}".format(ml_model.cv_results_))
	print("Best params: {}".format(ml_model.best_estimator_))

# Saving model
	fitted_filename = cwd + '/fitted_' + str(model) + '_' + str(feat_size) + '.sav'
	print(fitted_filename)
	joblib.dump(ml_model, fitted_filename)

##### PREDICTION ON TEST SET TAKEN FROM TRAINING SET #####
	ml_model = joblib.load(fitted_filename)

	# Predictions
	print("Predicting...")
	pred_test_y = ml_model.predict(test_mat)

	prediction = pd.DataFrame({'predictions': pred_test_y, 'true_value': test_y})
	print(prediction)
	prediction.to_csv(str(model) + '_test_pred' + str(feat_size) + '.csv')

	# Classification diagnostics
	print("Mean absolute error (MAE): {}".format(skmet.mean_absolute_error(test_y, pred_test_y)))
