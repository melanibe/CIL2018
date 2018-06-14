""""
File containing configuration variables used across the project
"""

# path to data folder and data sets
#RUN IT FROM MAIN CODE FOLDER so no .. !!!!!!

import os
global cwd
cwd = os.getcwd()
## KEEP CWD !
data_folder = cwd+'/data/'

data_label_path = data_folder + "labeled.csv"
data_score_path = data_folder + "scored.csv"
label_img_folder = data_folder + "labeled/"
score_img_folder = data_folder + "scored/"
query_img_folder = data_folder + "query/"

