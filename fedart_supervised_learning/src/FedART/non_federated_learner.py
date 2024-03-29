# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:34:38 2023

@author: shubhamp
"""

import numpy as np
from FedClustART.base.fusionART import FusionART
import copy
from sklearn.metrics import classification_report

class CentralizedLearner:
    def __init__(self, inpkg):
        
        self.data_pkg = copy.deepcopy(inpkg[0])
        self.FCARTfuncs = copy.deepcopy(inpkg[1])
        self.args = copy.deepcopy(inpkg[2])
        
        self.modalities = self.data_pkg['modalities'] 
        self.train_data = self.data_pkg['train_data']
        self.train_classes = self.data_pkg['train_classes']
        self.dummy_data = self.data_pkg['dummy_data']
        self.dummy_class = self.data_pkg['dummy_class']

        self.itrs = self.args.basemodel_iterations
        self.match_tracking = self.args.use_match_tracking_basemodel
        self.use_complement_coding = True
        
        alpha = self.args.basemodel_alpha
        beta = self.args.basemodel_beta 
        gamma = self.args.basemodel_gamma 
        rho = self.args.basemodel_rho
        
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

    def run_basemodal(self):
        
        data = self.train_data
        labels = self.train_classes
        print('basemodel data:', np.array(data).shape)
        
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
                        if self.FCARTfuncs.xAutoLearn(fusart=self.model, J=J):
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
            print('\n~~~~~~~~~~~~~~~~BaseModel, Epoch, ncodes: ', epoch, len(self.model.codes))
            
            if self.match_tracking and cnt <= 0:
               break
        
    
    def send_basemodel_to_coordinator(self):
        return copy.deepcopy(self.model)    
        