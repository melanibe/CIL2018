import pandas as pd
import numpy as np
import os

"""Melanie Bernhardt - Laura Manduchi - Melanie Bernhardt

This small script calculate the descriptive statistics on the score distribution 
of the SCORED training dataset as mentioned in the report.
"""

cwd = os.getcwd()
df = pd.read_csv(cwd+'/data/scored.csv')
scores = df['Actual'].values
print("The 20% percentile of the scores on the scored training set is {}". format(np.percentile(scores, 20)))
print("The standard deviation for the scores that are above 0.74 is {}".format(np.std(scores[scores>0.74])))
print("The mean score for the scores above 0.74 is {}".format(np.mean(scores[scores>0.74])))