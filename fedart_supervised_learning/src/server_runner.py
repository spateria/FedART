# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 02:00:41 2024

@author: shubhamp
"""

import pickle
import argparse
import socket, threading
import time

from FedART.fedserver import FedARTServer

#client_models = []
#server_model = None

def get_models_from_clients(client_socket, client_models):
    
    #global client_models
    #global server_model
    
    try:
        print(f"Connected to client: {client_socket.getpeername()}")

        # Receive model from client
        cdata = b""
        while True:
            packet = client_socket.recv(4096) #receive is packet sizes of 4096 bytes
            if not packet: break
            cdata += packet
            
        model_data = pickle.loads(cdata) #model_data contains client id and model
        #print('model data received:', model_data)
        
        #client_models.append(model_data)
        client_models[model_data['client_id']] = model_data['model']
        
    finally:
        #client_socket.close()
        pass
        
def send_model_to_clients(client_socket, server_model):
    
    try:
        #print(f"Connected to client: {client_socket.getpeername()}")
        model_data = pickle.dumps(server_model)
        client_socket.sendall(model_data)
        
    finally:
        #client_socket.close()
        pass
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='')
    parser.add_argument('--fl_rounds', type=int, default=1, help='')
    pa = parser.parse_args()
    
    if pa.dataset is None:
        print('\nPlease provide a dataset name as --dataset=<name> in the command above.')
    else:
        arg_file = '../saved_args/' + pa.dataset + '/args.pkl'
        
        with open(arg_file, 'rb') as f:
            args = pickle.load(f)
        
        data_dir = args.data_storage_path
        
        with open(data_dir + '/server_data/data.pkl', 'rb') as f:
            data_pkg = pickle.load(f)
            
        print('\nCreating Server')
        server = FedARTServer(args.num_clients, data_pkg, args, pa.fl_rounds)

        # Server configuration
        HOST = '127.0.0.1'  # Localhost
        PORT = 12345
        
        for _round in range(pa.fl_rounds):
            print(f'\n\nFL round {_round+1}')
            
            # Create a socket object
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Bind the socket to the address and port
            server_socket.bind((HOST, PORT))
            # Listen for incoming connections
            server_socket.listen()
            
            print("Server is listening...")
            
            # Accept connections from multiple clients
            client_models = [None for _ in range(args.num_clients)]
            client_threads = []
            client_sockets = []
            
            try:
                #Connect to clients
                while True:
                    client_socket, client_address = server_socket.accept()
                    client_sockets.append(client_socket)
            
                    # Limit the number of concurrent clients
                    if len(client_sockets) >= args.num_clients:
                        break
                
                #Get data from clients
                for client_socket in client_sockets:
                    client_thread = threading.Thread(target=get_models_from_clients, args=(client_socket,client_models,))
                    client_thread.start()
                    client_threads.append(client_thread)
                    
                #Run server process
                print('\n\nWaiting to receive all data from clients...')
                while None in client_models:
                    pass #wait
                server.get_client_model_codes(client_models)
                
                print('\n\nRunning server')
                server_model = server.train()
                
                #Send new models back to the clients
                print('\n\nSending new models back to the clients...')
                for client_socket in client_sockets:
                    client_thread = threading.Thread(target=send_model_to_clients, args=(client_socket,server_model,))
                    client_thread.start()
                    client_threads.append(client_thread)
            
                
            finally:
                for csc in client_sockets:
                    csc.close()
                    
                # Wait for all client threads to complete
                for thread in client_threads:
                    thread.join()
                    
                server_socket.close()
            
    
        
        
        
    