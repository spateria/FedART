# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 01:09:08 2024

@author: shubhamp
"""

import numpy as np
import pickle
import argparse
import copy
import pandas as pd
import csv
import os
from sklearn.metrics import classification_report, accuracy_score

from FedART.base.parent_class import FedARTBase

def get_eval_report(args, true_labels, pred_labels, num_codes=None):
       
        report = {}
        
        report['PR'] = classification_report(true_labels, pred_labels, target_names=list(args.class_dict.keys()), output_dict=True)
        report['PR'] = pd.DataFrame.from_dict(report['PR'])
        
        report['accuracy'] = accuracy_score(true_labels, pred_labels)
        
        report['num_codes'] = num_codes #TODO: implement how to get this from the models
        
        return report
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='')
    pa = parser.parse_args()
    
    if pa.dataset is None:
        print('\nPlease provide a dataset name as --dataset=<name> in the command above.')
    else:
        arg_file = '../saved_args/' + pa.dataset + '/args.pkl'
        
        with open(arg_file, 'rb') as f:
            args = pickle.load(f)
            
        client_data_dir = f'../partitioned_data/{pa.dataset}/client_data/'
        global_data_dir = f'../partitioned_data/{pa.dataset}/global_data/'
        
        client_model_dir = f'../learned_models/{pa.dataset}/client_models/'
        server_model_dir = f'../learned_models/{pa.dataset}/server_models/'
        
        num_clients = args.num_clients
        
        local_test = [{} for _ in range(num_clients)]
        global_test = {}
        
        client_models = [None for _ in range(num_clients)]
        server_model = None
        
        # load local data and models
        for i in range(num_clients):
            
            with open(client_data_dir + f'data_{i}.pkl', 'rb') as f:
                d = pickle.load(f)
                local_test[i]['test_data'] = d['test_data']
                local_test[i]['test_labels'] = d['test_classes']
                
            with open(client_model_dir + f'model_{i}.pkl', 'rb') as f:
                d = pickle.load(f)
                client_models[i] = d['model']
        
        
        with open(global_data_dir + 'data.pkl', 'rb') as f:
            d = pickle.load(f)
            global_test['test_data'] = d['test_data']
            global_test['test_labels'] = d['test_classes']
            
        with open(server_model_dir + 'model.pkl', 'rb') as f:
            d = pickle.load(f)
            server_model = d['model']
            fl_rounds = d['fl_rounds']
        
        b = FedARTBase()
        
        ## Test Server Model on Global Test Data
        data = b.complement_coded(global_test['test_data'])
        pred_labels, num_codes = b.predictFCART(server_model, data, args.modalities)
        server_report = get_eval_report(args, global_test['test_labels'], pred_labels, num_codes=num_codes)
        
        print('\n============== SERVER MODEL REPORT=================\n')
        print('Precision-Recall:\n', server_report['PR'])
        print('Accuracy:', server_report['accuracy'])
        print('Num. Codes:', server_report['num_codes'])
        
        ## Test Client Models on Global Test Data
        all_precision_recalls = []
        all_accuracies = []
        all_num_codes = []
        for i in range(num_clients):
            pred_labels, num_codes = b.predictFCART(client_models[i], global_test['test_data'], args.modalities)
            client_report = get_eval_report(args, global_test['test_labels'], pred_labels, num_codes=num_codes)
            all_precision_recalls.append(client_report['PR'])
            all_accuracies.append(client_report['accuracy'])
            all_num_codes.append(client_report['num_codes'])
            
        prdf = pd.concat(all_precision_recalls)
        prdf = prdf.groupby(level=0)
        accs = np.array(all_accuracies)
        ncs = np.array(all_num_codes)
        
        print('\n============== AVERAGE CLIENT MODEL REPORT=================\n')
        print('Precision-Recall:\n', prdf.mean())
        print('Accuracy:', np.mean(accs))
        print('Num. Codes:', np.mean(ncs))
        
        
        ### Save the results
        res_file = '../evaluation_results/' + pa.dataset + '/classification_results.csv'
        if not os.path.exists(res_file):
            fields=['fl_rounds','server acc','server ncodes', 'avg client acc', 'avg client ncodes']
            with open(res_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
            
        #['fl_rounds','server acc','server ncodes', 'avg client acc', 'avg client ncodes']
        fields=[fl_rounds, server_report['accuracy'], server_report['num_codes'], np.mean(accs), np.mean(ncs)]
        with open(res_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        