# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 02:00:41 2024

@author: shubhamp
"""

import pickle
import argparse
import multiprocessing

from FedART.fedclient import FedARTClient

def run_clients(client):
    client.train()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='')
    parser.add_argument('--fl_rounds', type=str, default=None, help='')
    pa = parser.parse_args()
    
    if pa.dataset is None:
        print('\nPlease provide a dataset name as --dataset=<name> in the command above.')
    else:
        arg_file = '../saved_args/' + pa.dataset + '/args.pkl'
        
        with open(arg_file, 'rb') as f:
            args = pickle.load(f)
        
        data_dir = args.data_storage_path
          
        clients = [None for  _ in range(args.num_clients)]
        
        for i in range(args.num_clients):
            with open(data_dir + f'/client_data/data_{i}.pkl', 'rb') as f:
                data_pkg = pickle.load(f)
            
            print(f'\nCreating Client {i}')
            clients[i] = FedARTClient(i, data_pkg, args)
            
        #Run client processes in parallel
        with multiprocessing.Pool() as pool:
            pool.map(run_clients, clients)
    