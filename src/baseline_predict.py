import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib
import sklearn.metrics as skmet
import PIL.Image

from config import *
from baseline_train import extract_feats


# Parameters
model = 'RandomForest' # Ridge or RandomForest
feat_size = 10
train_ratio = 0.99
y_data = ()

folder = data_folder + 'generated_images'
# folder = query_img_folder

try:
    if folder == query_img_folder:
        print("folder == query_img_folder")
        query_df = pd.load_csv(data_folder + 'query_df_' + str(feat_size) + '.csv')
    else:
        query_df = pd.load_csv(data_folder + 'query_df_' + str(feat_size) + '.csv')

except:
    # Preprocessing
    print(len([name for name in os.listdir(folder)]))
    num_img = len([name for name in os.listdir(folder)])
    query_mat = np.zeros((num_img, feat_size))

    i = 0
    index_mat = []

    for image in os.listdir(folder):
        raw_image = PIL.Image.open(os.path.join(folder, image))
        img_arr = np.array(raw_image.getdata()).reshape(raw_image.size[0], raw_image.size[1]).astype(np.uint8)
        img_feats = extract_feats(img_arr=img_arr, bins=feat_size)
        index_mat.append(image.split(".")[0])
        query_mat[i, :] = img_feats

        if i%100==0:
            print("Preprocessed {} images".format(i))
        i+=1

    index_mat = np.asarray(index_mat)

    # Creating dataframe
    query_df = pd.DataFrame(data=query_mat, index=index_mat)

    print("Saving feature matrices...")
    if folder == query_img_folder:
        query_df.to_csv(data_folder + 'query_df_' + str(feat_size) + '.csv')


# Loading model fitted model
fitted_filename = cwd + '/fitted_' + str(model) + '_' + str(train_ratio) + '_' + str(feat_size) + '.sav'
ml_model = joblib.load(fitted_filename)

# Predictions
print("Predicting...")
pred_data = ml_model.predict(query_df)

# Saving Predictions
try:
    prediction = pd.DataFrame({'Id': index_mat, 'Predicted': pred_data, 'True_value': y_data})
    print(prediction)
    print("Mean absolute error (MAE): {}".format(skmet.mean_absolute_error(y_data, pred_data)))
except:
    print("No true values given")
    prediction = pd.DataFrame(data=pred_data, columns=['Predicted'], index=index_mat)
    print(prediction)
    if folder == query_img_folder:
        prediction.to_csv(str(model) + '_' + str(feat_size) + '_query_pred.csv', index=True)
    else:
        prediction.to_csv(str(model) + '_' + str(feat_size) + '_trial_pred.csv', index=True)
    print('Saved predictions')

