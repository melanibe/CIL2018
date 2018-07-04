import os.path
import csv
import numpy as np
import pandas as pd

import PIL.Image
from sklearn.externals import joblib

from config import *

""" Melanie Bernhardt - Laura Manduchi - Melanie Gaillochet.

This file is a helper file for the preprocessing and feature matrix building
for the baselines presented in the project.
"""

def csv_to_dict(csv_path):
	""" This function transform the score/label csv file in a dictionary.
	Args:
		csv_path: the path to the file to transform.
	Returns:
		d: the corresponding dictionary
	"""
	with open(csv_path, 'r') as fp:
		csv_fp = csv.reader(fp)
		next(csv_fp)
		d = dict(filter(None, csv_fp))
		return d


def extract_feats(img_arr, bins):
	"""This function extracts simple features from the images.
	Args:
		img_arr: one input image to process
		bins: the number of bins for the histogram features.
	Returns:
		hist: histogram of the input image used as features for the baseline regressors.
	"""
	hist, _ = np.histogram(img_arr, bins=bins)
	return hist

def preprocessing_baseline_train_test(train_ratio, feat_size):
	"""This is the main preprocessing function for feature matrix building.
	Args:
		train_ratio(float between 0 and 1): train/dev split ratio to use
		feat_size(int): number of features to use for the feature matrix
	Returns:
		(train_mat, train_y, test_mat, test_y): train_mat is the feature matrix to use for training
												train_y is a vector with the scores for the training split
												test_mat is the feature matrix to use for validation
												test_y is a vector with the scores for the dev split
	"""
	try:
 	# Get preprocessed data if already available
		train_mat = np.load(data_folder + 'train_mat_' + str(train_ratio) + '_' + str(feat_size) + '.npy')
		train_y = np.load(data_folder + 'train_y_' + str(train_ratio) + '_' + str(feat_size) + '.npy')
		test_mat = np.load(data_folder + 'test_mat_' + str(train_ratio) + '_' + str(feat_size) + '.npy')
		test_y = np.load(data_folder + 'test_y_' + str(train_ratio) + '_' + str(feat_size) + '.npy')
		return(train_mat, train_y, test_mat, test_y)
	except:
	# If preprocessed data doesn't exist, compute it and save it (because it takes time to compute)
		print("Couldn't find matrices; preprocessing data")
        ### DATA LOADING ###
		# Print parameters used
		print('feature_size: {}'.format(feat_size))
		# Default paths to the scored training data
		scored_path = os.path.join(data_folder, "scored")
		score_file = os.path.join(data_folder, "scored.csv")
		# Randomly shuffle data - seed to ensure that we use the same training data as our final model.
		np.random.seed(10)
		# Initialization of the preprocessing.
		# Create a dictionary -> Image number: score (original order not kept)
		score_dict = csv_to_dict(score_file) 
		# Create a dataframe with Id and score (original order kept)
		score_df = pd.read_csv(score_file)
		num_images = score_df.shape[0]
		print("num images: {}".format(num_images))
		shuffled_indices = list(np.random.permutation(num_images))
		n_train = int(train_ratio * num_images)
		n_test = num_images - n_train
		print("n train: {}".format(n_train))
		print("n test: {}".format(n_test))
        # Initialize the feature matrices
		train_mat = np.zeros((n_train, feat_size))
		print("train mat shape: {}".format(train_mat.shape))
		train_y = np.zeros(n_train)
		test_mat = np.zeros((n_test, feat_size))
		print("test mat shape: {}".format(test_mat.shape))
		test_y = np.zeros(n_test)
		train_idx = 0
		test_idx = 0
	    ### FEATURE EXTRACTION ###
		counter = 0
		# Assemble train/test feature matrices / score vectors
		for idx in shuffled_indices:
			if counter%500==0:
				print("Image: {}/{}".format(counter, num_images))
			img_index = int(score_df.iloc[idx]['Id'])
			# loading the image
			raw_image = PIL.Image.open(os.path.join(scored_path, "{}.png".format(img_index)))
			img_arr = np.array(raw_image.getdata()).reshape(raw_image.size[0], raw_image.size[1]).astype(np.uint8)
			# extracting features
			img_feats = extract_feats(img_arr=img_arr, bins=feat_size)
			# extracting score
			score = float(score_dict[str(img_index)])
			# defining processed matrix for dev set
			if (test_idx) < n_test:
				test_mat[test_idx, :] = img_feats
				test_y[test_idx] = score
				test_idx += 1
			# same for training set
			else:
				train_mat[train_idx, :] = img_feats
				train_y[train_idx] = score
				train_idx += 1
			counter += 1
		# Saving features/scores to disk
		print("Saving feature matrices...")
		np.save(data_folder + 'train_mat_' + str(train_ratio) + '_' + str(feat_size) , train_mat)
		np.save(data_folder + 'train_y_' + str(train_ratio) + '_' + str(feat_size), train_y)
		np.save(data_folder + 'test_mat_' + str(train_ratio) + '_' + str(feat_size), test_mat)
		np.save(data_folder + 'test_y_' + str(train_ratio) + '_' + str(feat_size), test_y)   
		return(train_mat, train_y, test_mat, test_y)
