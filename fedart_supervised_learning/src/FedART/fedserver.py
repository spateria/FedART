# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:38:48 2023

@author: shubhamp
"""

import numpy as np
import pickle
import copy
import os
from sklearn.metrics import classification_report

from FedART.base.ARTxtralib import ART2ACModelOverride
from FedART.base.fusionART import FusionART
from FedART.base.parent_class import FedARTBase

class FedARTServer(FedARTBase):
    def __init__(self, num_clients, data_pkg, args):
        
        self.num_clients = num_clients
        self.data_pkg = data_pkg
        self.args = args
        self.model_storage_path = args.model_storage_path
        
        self.modalities = self.data_pkg['modalities'] 
        self.dummy_data = self.data_pkg['dummy_data']
        self.dummy_class = self.data_pkg['dummy_class']
        
        self.itrs = args.server_iterations
        self.match_tracking = args.use_match_tracking_server

        self.client_codes = None
        
        alpha = args.server_alpha
        beta = args.server_beta 
        gamma = args.server_gamma 
        rho = args.server_rho
        
        self.model_params = {'alpha': alpha, 'beta': beta,
                                           'gamma': gamma, 'rho': rho
                            }
        
        model_schema = []
        for j,md in enumerate(self.modalities[:-1]): #excluding 'label' which is last
            temp = {'name': md, 'compl': False, 'attrib': ['val'+str(i) for i in range(len(self.dummy_data[j]) * 2)]}
            model_schema.append(temp)
        
        model_schema.append({'name': self.modalities[-1], 'compl': True, 
                                                          'attrib': ['val'+str(i) for i in range(len(self.dummy_class))]})
        
        self.model = FusionART(schema = model_schema,
                                             beta=beta, alpha=alpha,
                                             gamma=gamma, rho=rho)
        
        if self.match_tracking:
            self.model.pmcriteria = [1,0]
        
    def train(self):
        if self.client_codes is None:
            print('Cannot run server level clustering. No client model received!')
            return
        
        data = [sublist[:-1] for sublist in self.client_codes]
        labels = [sublist[-1] for sublist in self.client_codes]
        
        print('server input:', np.array(data).shape)
        
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
                        #print(f"PERFECT MISMATCH AT {J}")
                        pass
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
            print('\n~~~~~~~~~~~~~~~~Server, Epoch, ncodes: ', epoch, len(self.model.codes))
            
            if self.match_tracking and cnt <= 0:
                break
                
        #client code averaging based aggregation
        temp = []
        avg_codes = self.average_aggregate_weights()
        
        if self.args.server_use_averaging_aggregation:
          #replace server model code weights with these averaged weights
          for j in range(len(self.model.codes)-1): #exclude last one since that is all 1 uncommited node
            if j in avg_codes:
              for ch in range(len(self.model.codes[j])-1): #do not modify label channel, which is the last one
                self.model.codes[j]['weights'][ch] = list(avg_codes[j][ch])
              temp.append(self.model.codes[j])
            else:
              print('Code ', j, ' not in avg codes')
          
          temp.append(self.model.codes[-1]) #because we excluded the last one in the loop above
          self.model.codes = temp
          print('\nLearned Server Codes after weight averaging: ', len(self.model.codes))
          
          #Save model
          with open(self.model_storage_path + '/server_models/model.pkl', 'wb') as f:
              pickle.dump(self.model, f)
              
          return self.model
            
    def average_aggregate_weights(self):
        
        data = [sublist[:-1] for sublist in self.client_codes]
        labels = [sublist[-1] for sublist in self.client_codes]
        similar_code_groups = {}
        
        for i in range(len(data)):
            
            input_schema = []
            for j,md in enumerate(self.modalities[:-1]): #excluding 'label' which is last
                temp = {'name': md, 'val': list(data[i][j])} #jth modality
                input_schema.append(temp)
            
            input_schema.append({'name': self.modalities[-1], 'val':labels[i]})
                              
            self.model.updateF1bySchema(input_schema)
            J = self.model.resSearch()
            if J not in similar_code_groups:
                similar_code_groups[J] = [data[i]]
            else:
                similar_code_groups[J].append(data[i])
        
        testJ = None #just to check if averaging works fine
        avg_codes = {}
        for J in similar_code_groups:
            code_group = np.array(similar_code_groups[J])
            if code_group.shape[0] == 2 and testJ == None:
              testJ = J
            avg_codes[J] = np.mean(similar_code_groups[J], axis=0)
        
        #print(similar_code_groups[testJ], '\n', avg_codes[testJ])
        
        return avg_codes
        
    def get_client_model_codes(self, client_models):
        
        new_cluster_codes = []
        
        for i in range(self.num_clients):
            cluster_codes = self.get_only_codes(client_models[i]) 
            
            for i in range(len(cluster_codes)):
                cluster_codes[i][-1] = self.complement_coding_removed(cluster_codes[i][-1], 'vector') #remove complement from label channel
        
            new_cluster_codes.append(cluster_codes)
        
        #concatenate codes received from various clients
        if new_cluster_codes != []:
            temp = []
            for cc in new_cluster_codes:
                for j in range(len(cc)):
                    temp.append(cc[j])
            self.client_codes = temp
            
        #print(self.cluster_codes)
        
    def update_clients(self): #send clustered codes to clients
        updated_cluster_codes = self.get_only_codes(self.model)
        #print(updated_cluster_codes)
        #NOTE: not removing complement coding since the client models need to integrate new codes in a complement codes form anyway
        for i in range(self.num_clients):
                pkg = {'server_sent': updated_cluster_codes}
                with open('comms/' + str(i) + '_client_server_com.pkl', 'wb') as handle:
                    pickle.dump(pkg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

    def send_global_model_to_coordinator(self):
        return copy.deepcopy(self.model)
        
        
        
        
        
        