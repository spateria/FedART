# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:33:21 2023

@author: shubhamp
"""

import numpy as np
import pandas as pd
import sys, os
import copy
import pickle
import csv
from sklearn.metrics import classification_report, accuracy_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class ExperimentCoordinator:
    def __init__(self, args):
        
        self.args = args #complete dataset specific arguments or parameters
        
        self.dataset = args.dataset #data source directory name
        self.modalities = args.modalities #data modalities 
        self.class_column = args.class_column #name of the class column in data
        self.class_dict = args.class_dict
        self.junk_columns = args.junk_columns #names of columns not containing training-relevant features
        self.norm_type = args.norm_type #type of data normalization
        self.num_clients = args.num_clients
        self.split_type = args.split_type #IID or non-IID
        self.random_seed = args.random_seed #random seed used for data partitioning among clients and train-test split... seeds should be same across different algorithms in different runs for fair evaluation
        
        self.data_pkg_clients = None
        self.data_pkg_server = None
        self.data_pkg_global = None
        
    
    def in_single_modality_format(self, data):
        temp = []
        for d in data:
            temp.append([d])
        return temp

    def restructure_data(self, data): #change from multimodal format to single combined vector..required for Euclidean distance calculations
        if len(np.array(data).shape) < 3: return data #the data is not in modality format
        
        data = copy.deepcopy(data)
        for i in range(len(data)): #num samples
            t = []
            d = [list(data[i][j]) for j in range(len(data[i]))]
            for md in d: t += md
            data[i] = t
        return data

    def normalize_data(self, chn_train, chn_test):
        chn_train = np.array(copy.deepcopy(chn_train), dtype='float')
        chn_test = np.array(copy.deepcopy(chn_test), dtype='float')
        
        if self.norm_type == 'unit_length':
          pass
          
        elif self.norm_type == 'feature_norm':
          chn_scaler = MinMaxScaler()

          chn_scaler.fit(chn_train)
          
          chn_train_norm = chn_scaler.transform(chn_train)
          
          chn_test_norm = chn_scaler.transform(chn_test)
          
        else:
          print('normalization type not defined!')
          sys.exit(0)
        
        return chn_train_norm, chn_test_norm
    
    def extract_train_test_data(self, df):
        #extract, split, and normalize data
        
        if (self.class_column is not None):
          classes = list(df[self.class_column])
          df = df.drop(self.class_column, axis=1) #selarate classes
        else:
          classes = None
        
        if (self.junk_columns is not None):
          otherpayload = {}
          for pc in self.junk_columns:
            otherpayload[pc] = list(df[pc]) #separate payload attributes not used for prediction
            df = df.drop(pc, axis=1)
        else:
          otherpayload = None
        
        #now we have purely numerical data
        data = df.to_numpy()
        
        chn_train, chn_test, label_train, label_test = train_test_split(data, classes, test_size=0.30, random_state=self.random_seed) #split
        
        #scale data 
        chn_train, chn_test = self.normalize_data(chn_train, chn_test) #MinMax scaling
        
        #convert labels into one-hot
        ltrain = []
        for lbl in label_train:
          temp = [0 for _ in self.class_dict.keys()]
          temp[self.class_dict[lbl]] = 1
          ltrain.append(temp)
        
        ltest = []
        for lbl in label_test:
          temp = [0 for _ in self.class_dict.keys()]
          temp[self.class_dict[lbl]] = 1
          ltest.append(temp)
        
        label_train = ltrain
        label_test = ltest
        
        if len(self.modalities[:-1]) == 1: #excluding last one, which is 'label'
          chn_train = self.in_single_modality_format(chn_train)
          chn_test = self.in_single_modality_format(chn_test)
        else:
          pass
        
        ## dummy data and class are used to set Fusion ART model architecture for each client..see fedclient.py
        dummy_data = np.zeros(np.array(chn_train).shape[1:])
        dummy_class = [0 for _ in range(len(self.class_dict))]
        
        pkg = {'train_data': chn_train, 'test_data': chn_test, 'train_classes': label_train, 'test_classes': label_test, 
                'otherpayload': otherpayload, 'modalities': self.modalities, 'dummy_data': dummy_data, 'dummy_class': dummy_class}
              
        return pkg
        
        
    def prep_client_data(self, source_data):
        #prepares train-test split data for each client
        
        df = source_data
        classes = list(self.class_dict.keys())
        num_classes = len(classes)
        
        #Step1: Do non-IID or IID split among clients
        sub_dfs = {}
        for _class in classes:
            if self.split_type == 'nonIID':
                beta = 0.5
                #create dirichlet fractions for this class
                split_fractions = np.random.default_rng(seed=self.random_seed).dirichlet(np.repeat(beta, self.num_clients))
            else:
                #create uniform fractions for this class
                split_fractions = np.repeat(1.0/self.num_clients, self.num_clients)
            
            #print(split_fractions)
            #get class-specific data
            class_df = df.loc[df[self.class_column] == _class] #get data for this class only
            class_df = class_df.sample(frac=1, ignore_index=True) #random shuffle
            
            #create the split sizes of class dataframe according to Dirichlet fractions
            idx = 0
            split_at_idx = []
            for frac in split_fractions[:-1]: #skipping the last one because np.split below uses indices at which split occurs... so for 5 splits we need four indices... last one is automatically taken care of
                idx += max(1,int(frac * len(class_df))) #because we don't want any split to be empty
                split_at_idx.append(idx)
                
            #print(split_at_idx)
            #split into sub dataframes corresponding to each client
            sub_dfs[_class] = np.split(class_df, split_at_idx) #split is done
        
        #now sub_dfs contains the partitioned data index by class names, with number of partitions equal to the number of clients
        #we need to merge data for each client
        client_dfs = [None for _ in range(self.num_clients)]
        for i in range(self.num_clients):
            sub_dfs_to_merge = tuple([sub_dfs[_class][i] for _class in sub_dfs]) #get data for the i^th client corresponding to each class 
            client_dfs[i] = pd.concat(sub_dfs_to_merge, axis=0, ignore_index=True).sample(frac=1, ignore_index=True) #concat and shuffle
        
        
        #Step2: Call extract_train_test_data on all client data (we call it here instead of in client processes because we also want to merge it into global test data)
        self.data_pkg_clients = [{} for _ in range(self.num_clients)]
        self.data_pkg_server = {}

        for i in range(self.num_clients): #for each client
            self.data_pkg_clients[i] = self.extract_train_test_data(client_dfs[i]) #extraction, split, and normalization
            
            print('\nClient: ', i)
            print('Client Train Data Shape: ', np.array(self.data_pkg_clients[i]['train_data']).shape)
            print('Client Test Data Shape: ', np.array(self.data_pkg_clients[i]['test_data']).shape)
        
    
    def prep_global_data(self):

        train_data = []
        train_labels = []
        for i in range(self.num_clients):
            ctd = self.data_pkg_clients[i]['train_data']
            ctl = self.data_pkg_clients[i]['train_classes']
            train_data += ctd
            train_labels += ctl
            
        test_data = []
        test_labels = []
        for i in range(self.num_clients):
            ctd = self.data_pkg_clients[i]['test_data']
            ctl = self.data_pkg_clients[i]['test_classes']
            test_data += ctd
            test_labels += ctl
        
        
        #Note: dummy data and dummy class are used to define nonFL Fusion ART model architecture
        self.data_pkg_global = {'train_data': train_data, 'train_classes': train_labels, 'test_data': test_data, 'test_classes': test_labels, 
                                      'dummy_data': self.data_pkg_clients[0]['dummy_data'], 'dummy_class': self.data_pkg_clients[0]['dummy_class'],
                                      'modalities': self.modalities}
        
        
        #Note: dummy data and dummy class are used to define server Fusion ART model architecture
        self.data_pkg_server = {
                                'dummy_data': self.data_pkg_clients[0]['dummy_data'], 
                                'dummy_class': self.data_pkg_clients[0]['dummy_class'],
                                'modalities': self.modalities
                                }
        
    
    def create_storages(self):
        
        #data storage
        dr = '../partitioned_data/' + self.dataset
        if not os.path.exists(dr):
            os.makedirs(dr)
            os.makedirs(dr + '/client_data')
            os.makedirs(dr + '/server_data')
            os.makedirs(dr + '/global_data')
        self.args.data_storage_path = dr
            
        #learned model storage
        dr = '../learned_models/' + self.dataset
        if not os.path.exists(dr):
            os.makedirs(dr)
            os.makedirs(dr + '/client_models')
            os.makedirs(dr + '/server_models')
            os.makedirs(dr + '/global_models') #nonFL, centralized model
        self.args.model_storage_path = dr
        
        #save args for use by client and server processes
        dr = '../saved_args/' + self.dataset
        if not os.path.exists(dr):
            os.makedirs(dr)
        with open(dr + '/args.pkl', 'wb') as f:
            pickle.dump(self.args, f)
            
        #save evalution results after federated learning
        dr = '../evaluation_results/' + self.dataset
        if not os.path.exists(dr):
            os.makedirs(dr)
            
    def save_fl_data(self):
        
        dr = self.args.data_storage_path
        
        for i in range(self.num_clients):
            with open(dr + '/client_data/' + f'/data_{i}.pkl', 'wb') as f:
                pickle.dump(self.data_pkg_clients[i], f)
        
        with open(dr + '/server_data/' + '/data.pkl', 'wb') as f:
            pickle.dump(self.data_pkg_server, f)
            
        with open(dr + '/global_data/' + '/data.pkl', 'wb') as f:
            pickle.dump(self.data_pkg_global, f)

    def get_eval_report(self, data, true_labels, pred_labels):
           
            report = {}
            
            data = copy.deepcopy(data)
            report['PR'] = classification_report(true_labels, pred_labels, target_names=list(self.class_dict.keys()), output_dict=True)
            report['PR'] = pd.DataFrame.from_dict(report['PR'])
            
            report['accuracy'] = accuracy_score(true_labels, pred_labels)
            
            report['num_codes'] = None #TODO: implement how to get this from the models
            
            return report
    

def run_coordinator(args):
    
    ec = ExperimentCoordinator(args)
    
    #load full data here from source data csv or hd5
    f = '../data/' + ec.dataset + '/data.csv'
    f_alt = '../data/' + ec.dataset + '/data.hd5'
    if os.path.isfile(f):
      df = pd.read_csv(f)
    else:
      df = pd.read_hdf(f_alt)
      
    source_data = df
    
    ec.prep_client_data(source_data)
    ec.prep_global_data()
    
    #Create storages for FL data and learned models
    ec.create_storages()
    
    #save data... models will be saved by the client and server processes
    ec.save_fl_data()


        

    
    
                                                                                               
            
       
#########################################################################################################


    