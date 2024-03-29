# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:24:26 2023

@author: shubhamp
"""

import numpy as np
import pandas as pd
from collections import Counter
import math
import sys
import random

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

with open('wdbc.data') as fl:
    dat = fl.read().split('\n')

cols = ['ID', 'class']
for j in range(30):
    cols.append('feat'+str(j))
    
df = pd.DataFrame(columns=cols)
idx = 0

for i in range(len(dat)-1):
    dat[i] = dat[i].split(',')
    row = [dat[i][0]] #ID
    row.append(dat[i][1]) #class
    for j in range(2,len(dat[i])):
        row.append(float(dat[i][j])) #feature
    df.loc[idx] = row
    idx += 1

df.to_csv('data.csv', index=False)