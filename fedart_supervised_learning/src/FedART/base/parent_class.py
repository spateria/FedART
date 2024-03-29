# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 00:48:28 2024

@author: shubhamp
"""

import copy 
import numpy as np

#Base class for Fusion ART models used by clients, server, and noFL model... contains common functions    
class FedARTBase:
    
    def complement_coded(self, data):
        data = copy.deepcopy(data) #do not want to manipulate original data
        for i in range(len(data)):
            for j in range(len(data[i])):
                temp = 1.0 - np.array(data[i][j])
                data[i][j] = np.concatenate((data[i][j], temp))
        return data
        
    def complement_coding_removed(self, data, in_type):
        if in_type == 'datamatrix':
          data = copy.deepcopy(data) #do not want to manipulate original data
          for i in range(len(data)):
              for j in range(len(data[i])):
                  _len = len(data[i][j])
                  data[i][j] = data[i][j][:int(_len/2)] #assuming second half has complement coding
          return data
        elif in_type == 'vector':
          vector = copy.deepcopy(data)
          _len = len(vector)
          return vector[:int(_len/2)]
        
    def get_only_codes(self, model):
        code_dict = copy.deepcopy(model.codes)
        only_codes = []
        for cd in code_dict:
            only_codes.append(cd['weights'])
        return only_codes
        
    def xAutoLearn(self, fusart=None, J=None, mindelta=0.00001):
        newcode = fusart.uncommitted(J)
        bw = copy.deepcopy(fusart.codes[J]['weights']) #keep the weights values before of J before learning
        fusart.autoLearn(J)
        if not newcode:
            return any([any([abs(bw[k][i]-fusart.codes[J]['weights'][k][i]) > mindelta for i in range(len(bw[k]))]) for k in range(len(bw))])
        else:
            return True
    
    def predictFCART(self, pred_model, data):
        J_pred = []
        match_value = []
        pred_labels = []
        model = copy.deepcopy(pred_model) #to avoid modifying the original model during prediction
        
        model.setParam('gamma', [1.0,0.0]) #setParam is used to change the parameter of fusion ART 
        model.setParam('rho', [0,0])         #setParam is used to change the parameter of fusion ART
        
        for i in range(len(data)):
        
            input_schema = []
            for j,md in enumerate(self.modalities[:-1]): #excluding 'label' which is last
                temp = {'name': md, 'val': list(data[i][j])} #jth modality
                input_schema.append(temp)

            model.updateF1bySchema(input_schema)
        
            J = model.resSearch() # use resSearch to select node J
            J_pred.append(J)      # record the selected node J
            
            model.doReadout(J,1)  # readout the weights of J to channel 2 in activityF1 
            model.TopDownF1()     # update the readout values in activityF1 to the schema
            f1_label = model.F1Fields[1]['val'] #get the value vector of the label
            pred_labels.append(f1_label)            #record the retrieved label
        
        num_learned_codes = len(self.get_only_codes(model))
        return pred_labels, num_learned_codes
    
    def get_eval_report(self, data, true_labels, pred_labels):
           
            report = {}
            
            data = copy.deepcopy(data)
            report['PR'] = classification_report(true_labels, pred_labels, target_names=list(self.class_dict.keys()), output_dict=True)
            report['PR'] = pd.DataFrame.from_dict(report['PR'])
            
            report['accuracy'] = accuracy_score(true_labels, pred_labels)
            
            report['num_codes'] = None #TODO: implement how to get this from the models
            
            return report