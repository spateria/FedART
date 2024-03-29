# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:34:38 2023

@author: shubhamp
"""

import numpy as np
import pickle
import copy
import os, sys
import socket
import time
from sklearn.metrics import classification_report

from FedART.base.fusionART import FusionART
from FedART.base.parent_class import FedARTBase

class FedARTClient(FedARTBase):
    def __init__(self, client_id, data_pkg, args):
        
        self.id = client_id
        self.data_pkg = data_pkg
        self.args = args
        self.model_storage_path = args.model_storage_path
        
        self.modalities = self.data_pkg['modalities'] 
        self.train_data = self.data_pkg['train_data']
        self.train_classes = self.data_pkg['train_classes']
        self.dummy_data = self.data_pkg['dummy_data']
        self.dummy_class = self.data_pkg['dummy_class']

        self.itrs = args.client_iterations
        self.match_tracking = args.use_match_tracking_clients
        self.use_complement_coding = True
        
        alpha = args.client_alpha
        beta = args.client_beta 
        gamma = args.client_gamma 
        rho = args.client_rho
        
        self.model_params = {'alpha': alpha, 'beta': beta,
                                           'gamma': gamma, 'rho': rho
                            }
        

        model_schema = []
        for j,md in enumerate(self.modalities[:-1]): #excluding 'label' which is last
            temp = {'name': md, 'compl': self.use_complement_coding, 'attrib': ['v'+str(i) for i in range(len(self.dummy_data[j]))]}
            model_schema.append(temp)
        
        model_schema.append({'name': self.modalities[-1], 'compl': self.use_complement_coding, 
                                                          'attrib': ['v'+str(i) for i in range(len(self.dummy_class))]})

        self.model = FusionART(schema = model_schema,
                                             beta=beta, alpha=alpha,
                                             gamma=gamma, rho=rho)
        
        if self.match_tracking:
            self.model.pmcriteria = [1,0]
        
        #self.model.displayNetwork()
        #self.model.F1Fields
        
        # Connect to server
        self.HOST = '127.0.0.1'  # Server's IP address (localhost)
        self.PORT = 12345
        

    def train(self):
    
        self.update_model()
        
        data = self.train_data
        labels = self.train_classes
        
        #print('Complement Coded Client Data Shape: ', np.array(data).shape)
        for epoch in range(self.itrs): #train and cluster
            cnt = 0
            J_train = []
            pred_labels = []
            
            for i in range(len(data)):
    
                input_schema = []
                for j,md in enumerate(self.modalities[:-1]): #excluding 'label' which is last
                    temp = {'name': md, 'val': list(data[i][j])} #jth modality
                    input_schema.append(temp)
                
                input_schema.append({'name': self.modalities[-1], 'val':labels[i]})
                                  
                self.model.updateF1bySchema(input_schema)
                
                if self.match_tracking:
                    J = self.model.resSearch(mtrack=[0]) # resonance search to select J
                    #print(f'last rho {self.model.lastActRho} at node {J} total codes so far {len(self.model.codes)}')
                    if self.model.perfmismatch:
                        print(f"PERFECT MISMATCH AT {J}")
                    else:
                        #model_fa.autoLearn(J)    # learning by expanding F2 or updating node J
                        if self.xAutoLearn(fusart=self.model, J=J):
                             cnt += 1
                        J_train.append(J)        # record the selected node J
                else:
                    J = self.model.resSearch() # resonance search to select J
                    self.model.autoLearn(J)    # learning by expanding F2 or updating node J
                    J_train.append(J)        # record the selected node J
                
                self.model.doReadout(J,1)  # readout the weights of J to channel 2 in activityF1 
                self.model.TopDownF1()     # update the readout values in activityF1 to the schema
                f1_label = self.model.F1Fields[1]['val'] #get the label value vector
                pred_labels.append(f1_label)  # record the returned label
                
            #TODO: get F1 and other scores during training here...we have labels and pred_labels
            
            #report = classification_report(labels, pred_labels, target_names=list(self.args.class_dict.keys()), output_dict=True)
            print('\n~~~~~~~~~~~~~~~~Client, Epoch, ncodes: ', self.id, epoch, len(self.model.codes))
            
            if self.match_tracking and cnt <= 0:
               break
            
        ##########Save model and send to server################
        
        # Create a socket object
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect to the server
        self.client_socket.connect((self.HOST, self.PORT))
        
        data_dump = {'client_id': self.id, 'model': self.model}
        with open(self.model_storage_path + f'/client_models/model_{self.id}.pkl', 'wb') as f:
            pickle.dump(data_dump, f)
            
        print('=============', len(self.model.codes))
        
        with open(self.model_storage_path + f'/client_models/model_{self.id}.pkl', 'rb') as f:
            model_data = f.read() #reads serialized pickled data
            
        self.client_socket.sendall(model_data)
        
        self.client_socket.shutdown(1) #1 means sent all
        ###########Receive new model from server###############
        
        # Create a socket object
        print(f'Client {self.id} waiting for new server model')
        sdata = b""
        while True:
            packet = self.client_socket.recv(4096) #receive is packet sizes of 4096 bytes
            if not packet: break
            sdata += packet
        
        self.client_socket.close()
        
        new_model_data = pickle.loads(sdata) #model_data contains client id and model
        print(f'Client {self.id} received new server model')
        
    
    def update_model(self):
        new_cluster_codes = None
        f = 'comms/' + str(self.id) + '_client_server_com.pkl'
        try:
          with open(f, 'rb') as handle:
              pkg = pickle.load(handle)
        except:
          return
        
        if list(pkg.keys())[0] == 'server_sent': #server sent new codes
            new_cluster_codes = pkg['server_sent']
            os.remove(f)
    
        '''TODO: write code to insert/merge the new F2 nodes in the client model'''
        if new_cluster_codes is not None:
            print('New codes received from server: ', np.array(new_cluster_codes).shape)
            pass

    def send_local_eval_scores(self):
        #print('\nCodes learned by client ', self.id, ' ', self.get_only_codes(), '\nlearned code shape: ', np.array(self.get_only_codes()).shape)

        scores = {}
        data = self.data_pkg['test_data']
        model = self.model
        true_labels = self.data_pkg['test_classes']
        
        pred_labels, ncodes = self.predictFCART(model, data)
        scores = self.get_eval_scores(data, true_labels, pred_labels)
        scores.loc['#ncodes'] = 0
        scores['#ncodes'] = 0
        scores.loc['#ncodes', '#ncodes'] = ncodes
        
        return scores
        
        
        
        
        
        