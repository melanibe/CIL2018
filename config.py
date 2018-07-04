import os
global cwd

""" Melanie Bernhardt - Laura Manduchi - Melanie Gaillochet.

File containing configuration variables used across the project.
Do not change please.
"""

cwd = os.getcwd()
data_folder = cwd+'/data/'

data_label_path = data_folder + "labeled.csv"
data_score_path = data_folder + "scored.csv"
label_img_folder = data_folder + "labeled/"
score_img_folder = data_folder + "scored/"
query_img_folder = data_folder + "query/"

test_path = data_folder + "query/"

# path to runs
runs_dir = cwd + "/runs"
