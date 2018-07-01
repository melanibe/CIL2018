from config import *
import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib
import sklearn.metrics as skmet
import PIL.Image

from tweaked_cosmology import extract_feats


# Parameters
model = 'RandomForest' # Ridge or RandomForest
feat_size = 50
y_data = ()


try:
    query_mat = np.load(data_folder + 'query_mat_' + str(feat_size) + '.npy')

except:
    # Preprocessing
    print(len([name for name in os.listdir(query_img_folder)]))
    num_img = len([name for name in os.listdir(query_img_folder)])
    query_mat = np.zeros((num_img, feat_size))

    i = 0
    index_mat = np.zeros(len([name for name in os.listdir(query_img_folder)]))

    for image in os.listdir(query_img_folder):
        raw_image = PIL.Image.open(os.path.join(query_img_folder, image))
        img_arr = np.array(raw_image.getdata()).reshape(raw_image.size[0], raw_image.size[1]).astype(np.uint8)
        img_feats = extract_feats(img_arr=img_arr, bins=feat_size)
        # index_mat = np.append(index_mat, image.split(".")[0])

        query_mat[i, :] = img_feats
        if i%100==0:
            print("Preprocessed {} images".format(i))
        i+=1

    print("Saving feature matrices...")
    np.save(data_folder + 'query_mat_' + str(feat_size), query_mat)
    # np.save(data_folder + 'query_mat_index_' + str(feat_size), query_mat)

# Getting array with all image names
index_mat = []
for image in os.listdir(query_img_folder):
    print(image.split(".")[0])
    index_mat.append(image.split(".")[0])
index_mat = np.asarray(index_mat)
# np.save(data_folder + 'query_mat_index_' + str(feat_size), query_mat)

query_mat = pd.DataFrame(data=query_mat, index=index_mat)

# Loading model fitted model
fitted_filename = cwd + '/fitted_' + str(model) + '_' + str(feat_size) + '.sav'
ml_model = joblib.load(fitted_filename)

# Predictions
print("Predicting...")
pred_data = ml_model.predict(query_mat)

try:
    prediction = pd.DataFrame({'Id': index_mat, 'Predicted': pred_data, 'True_value': y_data})
    print(prediction)
    prediction.to_csv(str(model) + '_test_pred.csv')
    print('Saved predictions')
    print("Mean absolute error (MAE): {}".format(skmet.mean_absolute_error(y_data, pred_data)))
except:
    print("No values given")
    prediction = pd.DataFrame(data=pred_data, columns=['Predicted'], index=index_mat)
    prediction.to_csv(str(model) + '_' + str(feat_size) + '_query_pred.csv', index=True)
    print('Saved predictions')

