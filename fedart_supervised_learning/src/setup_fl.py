# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:03:01 2023

@author: shubhamp
"""

import pandas as pd
import os, sys
import argparse

from experiment_coordinator import run_coordinator

import warnings

####### ~~~~~~~~~~~~~~~ ATTENTION: Change this for your dataset ~~~~~~~~~~~~~~~~~~ ######
def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--runFedART', type=int, default=1, help='This flag is used to activate FedART... only useful when other algorithms are also added and we only want to run other but not FedART')
    parser.add_argument('--split_type', type=str, default='IID', help='Type of data partitioning among clients')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed used in this experiment... for data partitioning, train-test split, etc.')
    
    args = parser.parse_args()
    
    if args.dataset == 'breast_cancer':
        args.class_column = 'class' #Provide name of class columns given in the dataframe
        args.class_dict = {'M': 1, 'B': 0} #Provide class names and numerical labels for them
        args.junk_columns = ['ID'] #provide column names that do not contain features that are relevant to learning 
        args.norm_type = 'feature_norm' # feature_norm or unit_length norm...way to normalize data
        args.modalities = ['data_channel1', 'label'] #only one input channel and one label channel...change according to data modality
        args.num_clients = 5 
        
        # FedART args
        args.client_alpha = [0.1, 0.1]
        args.client_beta = [1.0, 1.0]
        
        args.server_alpha = [0.1, 0.1]
        args.server_beta = [1.0, 1.0]
        
        args.basemodel_alpha = [0.1, 0.1]
        args.basemodel_beta = [1.0, 1.0]
        
        args.server_use_averaging_aggregation = 1 #this means the learned code weights of server model are replaced with average of similar client                                                         #code weights. The similar client code weights can be identified using resSearch after server model                                                      #training is completed 
        
        args.use_match_tracking_clients = 1
        args.use_match_tracking_server = 1
        args.use_match_tracking_basemodel = 1
        
        if args.use_match_tracking_clients:
          args.client_iterations = 100
          args.client_rho = [0.0, 1.0]
          args.client_gamma = [1.0, 0.0]
        else:
          args.client_iterations = 5
          args.client_rho = [0.5, 1.0]
          args.client_gamma = [0.5, 0.5]
          
        if args.use_match_tracking_server:
          args.server_iterations = 100
          args.server_rho = [0.6, 1.0]   #keeping base vigilance high in the server model because we do not want to overgeneralize the aggregated model
          args.server_gamma = [1.0, 0.0]
        else:
          args.server_iterations = 5
          args.server_rho = [0.8, 1.0]
          args.server_gamma = [0.5, 0.5]
          
        if args.use_match_tracking_basemodel:
          args.basemodel_iterations = 100
          args.basemodel_rho = [0.6, 1.0]
          args.basemodel_gamma = [1.0, 0.0]
        else:
          args.basemodel_iterations = 1
          args.basemodel_rho = [0.6, 1.0]
          args.basemodel_gamma = [0.5, 0.5]
      
    elif args.dataset is None:
        print('\nPlease define the dataset to be used as --dataset=<name> in the commands above.')
        return None
    else:
        print('\nArguments not defined for this dataset, please check.')
        return None
        
    return args
    
    
if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=UserWarning)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    args = get_args()
    
    if args is not None: 
        run_coordinator(args)
