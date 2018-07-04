import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib
import sklearn.metrics as skmet
import PIL.Image

from config import *
from baseline_train import extract_feats

""" Mélanie Bernhardt - Laura Manduchi - Mélanie Gaillochet.

This file is used to output the predictions on the query dataset for Kaggle.
It can also be run the output the prediction on any custom dataset (for example to
check the predicted score on the generated images).
It assumes you have trained your estimator first and saved it to /baselines_models/ subfolder.

You have to enter the 
    model name: 'Ridge' and 'RandomForest' for the model to build to use for prediction.
    feat_size: the number of features that were used for training (we used 10).
    train_ratio: the train/dev split ratio (we used 0.99 training split).
    folder: the folder were the images to evaluate are located. By default it is the query images set.
"""

##### PARAMETERS (TO MODIFY IF NECESSARY) #####
model =  'RandomForest' # or 'Ridge'
feat_size = 10
train_ratio = 0.99
y_data = ()

folder = query_img_folder # for Kaggle competition
# folder = cwd + '/produced/1530273051/' #for evaluating the score of the generated images

try:
    # if available get the already processed features matrix
    query_df = pd.read_csv(folder + 'query_df_' + str(feat_size) + '.csv', index_col=0)
    index_mat = query_df.index.values
except:
    # if not preprocess the images to create the feature matrix
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
    # Save it for later use.
    print("Saving feature matrices...")
    query_df.to_csv(folder + 'query_df_' + str(feat_size) + '.csv')


##### CSV PREDICTIONS OUTPUT #####
# Loading fitted model
fitted_filename = cwd +'/baselines_models' + '/fitted_' + str(model) + '_' + str(train_ratio) + '_' + str(feat_size) + '.sav'
ml_model = joblib.load(fitted_filename)

# Predictions
print("Predicting...")
pred_data = ml_model.predict(query_df)
print(np.mean(pred_data))
print(np.std(pred_data))

# Saving Predictions
prediction = pd.DataFrame(data=pred_data, columns=['Predicted'], index=index_mat)
prediction['Id'] = index_mat
if folder == query_img_folder:
    prediction.to_csv(cwd+'/predictions/' + str(model) + '_' + str(feat_size) + '_query_pred.csv', index=False)
else:
    prediction.to_csv(cwd+'/predictions/' + str(model) + '_' + str(feat_size) + '_trial_pred.csv', index=False)
print('Saved predictions')

