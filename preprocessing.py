import numpy as np
import os
import pandas as pd
from skimage import io

"""Melanie Bernhardt - Laura Manduchi - Melanie Gaillochet.

Helper file to load the training data and create batches for network training.
"""

cwd = os.getcwd()
data_folder = cwd+'/data/'

def load_data(type, path=None, csv_file=None): 
	""" This function creates a dataframe with:
			- a column containg the image (array)
			- 	EITHER a column 'labeled' with the label if type = labeled
				OR a column 'scored' with the score if type = score
				OR no other column if type = query
			- the row index of the dataframe is the name of the image
	Args:
		type: 'labeled' or 'scored' or 'query' indicating which folder to preprocess.
		path(str, optional): defining the path of the data to process. 
			                 Default None corresponds to normal data path.
        csv_file(str, optional): defining the path of the score or label data to process.
                                 Default None corresponds to normal data path.
	Examples:
	>>> df_label = load_data('labeled')
	>>> df_label.loc[9397734]
		img        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
		labeled                                                    1
		Name: 9397734, dtype: object
	>>> df_score = load_data('scored')
	>>> df_score.loc[5694059]
		img       [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
		scored                                              1.83779
		Name: 5694059, dtype: object
	>>> df_query = load_data('query')
	>>> df_query.loc[1000956]
		img    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
		Name: 1000956, dtype: object
	"""
	print("Starting the preprocessing for {}...".format(str(type)))
	result = pd.DataFrame()
	images_array=[]
    # The index of the datafram is the original name of the images.
	ids = []
    # Defining the default path (if no other path is provided).
	if path == None:
		path = data_folder + "{}/".format(type)
    # Load the images
	for image in os.listdir(path):
		if ("png" in image):
			images_array.append((io.imread(os.path.join(path, image))))
			ids.append(int(image.replace(".png","")))
	result['img'] = pd.Series(images_array, index=ids)
    # If type scored/labeled add the score/label column
	if type != 'query':
		if csv_file == None:
			csv_file = data_folder + "{}.csv".format(type)
		df = pd.read_csv(csv_file)
		result[type]= pd.Series(df['Actual'].values, index=df['Id'].values)
	return result


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
    Note: this function is borrowed from a tutorial of the Natural Language Understanding class.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]
