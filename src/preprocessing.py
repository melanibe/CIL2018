import numpy as np
import os
import pandas as pd
from skimage import io
from config import *

def load_data(type, path=None, csv_file = None, augmented=True):  # type=labeled or scored (query special case without csv later)
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
	print("Starting the preprocessing for {}...".format(str(type)))
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

	if augmented:
		print('Augmenting data...')
		prep_data = result

		# We want to augment values whose score is under 1 (bad image) - augment to 2 (given data that we have)
		if str(type)=='scored':
			prep_low = prep_data[prep_data[str(type)] < 1.1]
			num_aug = 2

		# We want to augment number of images if label is 0 (bad image) - augment to 3 (given data that we have)
		else: # type is labeled
			prep_low = prep_data[prep_data[str(type)] == 0]
			num_aug = 3

		prep_low_images = np.asarray(prep_low['img'])
		prep_low_value = np.asarray(prep_low[str(type)])

		# print("Number of images: {}".format(len(prep_data[str(type)])))
		print("Number of low value images: {}".format(len(prep_low_images)))

		print("Number of augmentation: {}".format(num_aug))
		if num_aug==2:
			for i in range(0, len(prep_low[str(type)])):
				rotated_df = pd.DataFrame({'img': [np.rot90(prep_low_images[i], 1),
												   np.rot90(prep_low_images[i], 2)
												   # np.rot90(prep_lowscored_images[i], 3)
												   ],
										   str(type): [prep_low_value[i],
													  prep_low_value[i]
													  # prep_lowscored_scores[i]
													  ]})
				prep_data = prep_data.append(rotated_df)

		elif num_aug == 3:
			for i in range(0, len(prep_low[str(type)])):
				rotated_df = pd.DataFrame({'img': [np.rot90(prep_low_images[i], 1),
												   np.rot90(prep_low_images[i], 2),
												   np.rot90(prep_low_images[i], 3)
												   ],
										   str(type): [prep_low_value[i],
													   prep_low_value[i],
													   prep_low_value[i]
													   ]})
				prep_data = prep_data.append(rotated_df)

			# io.imsave(cwd +"/data/modified image{}_1.png", np.rot90(prep_lowscored_images[i], 1))
			# io.imsave(cwd +"/data/modified image{}_2.png", np.rot90(prep_lowscored_images[i], 2))
			# io.imsave(cwd +"/data/modified image{}_3.png", np.rot90(prep_lowscored_images[i], 3))

		if str(type) == 'scored':
			print("Num images low value after augmentation: {}".format(
			len(np.asarray(prep_data[prep_data[str(type)] < 1.1]['img']))))
			print("Num images high value : {}".format(
			len(np.asarray(prep_data[prep_data[str(type)] >= 1.1]['img']))))
		else:
			print("Num images low value after augmentation: {}".format(
			len(np.asarray(prep_data[prep_data[str(type)] == 0]['img']))))
			print("Num images high value: {}".format(
			len(np.asarray(prep_data[prep_data[str(type)] != 0]['img']))))

		print("Total number of images: {}".format(prep_data.shape[0]))

		print("Saving increased data..")
		path = data_folder + "/increased_" + str(type) + ".csv"
		prep_data.to_csv(path, sep=',')
		print("Saved the preprocessed data successfully as {}".format(path))
		return prep_data

	else:
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
    print(x["img"][0])
