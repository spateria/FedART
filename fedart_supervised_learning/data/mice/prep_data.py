# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:13:10 2023

@author: shubhamp
"""

import pandas as pd
import os
from sklearn.impute import KNNImputer
import numpy as np
import shutil

os.chdir(os.getcwd())

source = 'Mice_protein'

df = pd.read_csv('data/' + source + '/data_orig_missing.csv')

if 1:
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df_filled = pd.DataFrame(imputer.fit_transform(df.iloc[:, 1:-4]))
    df.iloc[:, 1:-4] = df_filled
else:
    df = df.fillna(df.median())

df = df.drop(columns=['Genotype', 'Treatment', 'Behavior'])
df.to_csv('data/' + source + '/data.csv', index=False)

#print(df.describe())
