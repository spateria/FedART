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

'''df = pd.read_csv('data/' + source + '/data_orig_missing.csv')

if 1:
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df_filled = pd.DataFrame(imputer.fit_transform(df.iloc[:, 1:-4]))
    df.iloc[:, 1:-4] = df_filled
else:
    df = df.fillna(df.median())

df = df.drop(columns=['Genotype', 'Treatment', 'Behavior'])
df.to_csv('data/' + source + '/data.csv', index=False)

print(df.describe())'''

def dfsplit(_df, _group_splits):
    temp = _df[_df.class_n == _group_splits[0]]
    for j in range(1,len(_group_splits)):
        temp = temp.append(_df[_df.class_n == _group_splits[j]], ignore_index=True)
    return temp

def full_hetero():
    #fullhetero client split --- split into 4 clients with 2 unique groups each
    print("FULLHETERO SPLIT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    df = pd.read_csv('data.csv')
    
    group_splits = [['c-CS-s', 't-CS-s'],  ['c-CS-m', 't-CS-m'], 
                    ['c-SC-s', 't-SC-s'], ['c-SC-m', 't-SC-m']]
    
    subdf = [None for i in range(len(group_splits))]
    for i in range(len(group_splits)):
        
        subdf[i] = dfsplit(df, group_splits[i])
        
        print(subdf[i]['class_n'].value_counts(), '\n')
        subdf[i].to_csv('fullhetero/data' + str(i) +'.csv', index=False)
    

def mix_hetero():
    print("MIXHETERO SPLIT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    df = pd.read_csv('data/' + source + '/data.csv')
    
    #mixed split ---- split the data into 4 clients which have 4 groups each..the groups are listed below
    group_splits = [['c-CS-s', 't-CS-s', 'c-SC-s', 't-SC-s'],  
                    ['c-CS-m', 't-CS-m', 'c-SC-m', 't-SC-m']]
    
    #first we'll split the original df by class groups, then for each split, we'll further randomly split into two clients
    subdf = [None for i in range(len(group_splits))]
    client_i = 0
    for i in range(len(group_splits)):
        
        subdf[i] = dfsplit(df, group_splits[i])
        
        #randomly split subdf into two clients
        df1 = subdf[i].sample(frac=0.5, random_state=42)
        df2 = subdf[i].drop(df1.index)
        print(df1['class_n'].value_counts(), '\n')
        print(df2['class_n'].value_counts(), '\n')
        df1.to_csv('mixhetero/data' + str(client_i) +'.csv', index=False)
        df2.to_csv('mixhetero/data' + str(client_i+1) +'.csv', index=False)
        client_i += 2

def random_split():
    #randomized client split -- 4 clients, should ideally have 8 groups each since it is random homogeneous split
    print("RANDOM HOMOGENEOUS SPLIT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    df = pd.read_hdf('data/' + source + '/data.hd5')
    
    num_clients = 4
    
    # Randomly splitting the DataFrame
    df = df.sample(frac=1, random_state=int(np.random.uniform(10,100,1)[0]))  # Randomly shuffle the DataFrame
    
    # Splitting the DataFrame into four parts
    split_size = len(df) // num_clients
    subdf = [None for i in range(num_clients)]
    for i in range(num_clients):
        subdf[i] = df[(i)*split_size:(i+1)*split_size]
        
        #print(subdf[i]['class_n'].value_counts(), '\n')
        subdf[i].to_csv('randomhomo/data' + str(i) +'.csv', index=False)
        
if __name__ == '__main__':
    
    shutil.copy('data.csv', 'Baseline_nonfed/data0.csv')
    df = pd.read_csv('data.csv')
    df.to_hdf('data.hd5', key='df', mode='w')