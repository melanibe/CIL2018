import numpy as np
import os
import pandas as pd
from skimage import io
from config import *

def load_data(type, path=None, csv_file = None):  # type=labeled or scored (query special case without csv later)
	"""This function creates a dataframe with:
			- a column containg the image (array)
			- a column 'labeled' with the label if type = labeled
			OR a column 'scored' with the score if type = score
			OR no other column if type = query
			- the row index of the dataframe is the name of the image
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
	print("Starting the preprocessing...")
	result = pd.DataFrame()
	images_array=[]
	ids = []  # name of the images to retrieve in csv file
	if path == None:
		path = data_folder + "{}/".format(type)
	for image in os.listdir(path):
		try:
			images_array.append((io.imread(os.path.join(path, image), as_grey=True)))
			ids.append(int(image.replace(".png","")))
		except:
			pass
	result['img'] = pd.Series(images_array, index=ids)
	if type != 'query':
		if csv_file == None:
			csv_file = data_folder + "{}.csv".format(type)
		# loading the score or the label
		df = pd.read_csv(csv_file)
		result[type]= pd.Series(df['Actual'].values, index=df['Id'].values)
	return result


# FUNCTION DEFINED IN SERIES 5 SOLUTION
def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
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


# For testing
if __name__ == "__main__":
	x = load_data(type='labeled', path=None, csv_file=None)
	#print(min(x.iloc[0, ['img']].values), max(x.iloc[0, ['img']].values))