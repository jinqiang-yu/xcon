
from __future__ import print_function, division
import time


import os
import argparse
import pickle

#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torchvision import datasets, transforms
#from torch.autograd import Variable
#from torchvision.transforms import ToTensor
#from torchvision.datasets import ImageFolder
#from torch.utils.data import Dataset, DataLoader
#import os
#from PIL import Image
#from PIL import ImageOps
#import torchvision.models as models


#start = time.time()
from sklearn.preprocessing import OneHotEncoder
#end = time.time()
#print("initial2", end - start)
#
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from adam import Adam
#from modules import *
#from localmodels import *
from .tabulardataset import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Ignore warnings
import warnings
#from matplotlib.pyplot import axis
warnings.filterwarnings("ignore")
# Helper function to show a batch

def load_subsets(config, enc, path_data, file_data):
    
    filename = os.path.join(path_data, file_data)
    #print(f"Loading from {filename}")
    pd_data = pd.read_csv(filename)

    Y = pd_data[pd_data.columns[-1]].astype(np.int64).values.reshape(-1, 1)
    X = pd_data[pd_data.columns[:-1]].astype(np.int64).values
    if (not config["data"]["donot_convert_data"]):
        X = enc.transform(X)
    return X, Y
def prepare_tabular(config):

    path_data = config["data"]["data_dir"]  
    file_data = config["data"]["data_file"]
    test_size = config["data"]["test_size"]
    seed = config["manual_seed"]
    ############################3
    # Load dataset
    #############################    
    filename = os.path.join(path_data, file_data)
    testing = False
    if (testing):
        print("reading from", filename)
    pd_data = pd.read_csv(filename)
    #print(pd_data)

    Y = pd_data[pd_data.columns[-1]].astype(np.int64).values.reshape(-1, 1)
    X = pd_data[pd_data.columns[:-1]].astype(np.int64).values
    
    
    if(config["data"]["use_one_hot"]):
        enc = OneHotEncoder(handle_unknown='ignore', sparse = False)
        enc.fit(X)
        one_hot_size= enc.transform(X).shape[1]
        if (not config["data"]["donot_convert_data"]):
            X = enc.transform(X)
    else:
        enc = lambda x: np.asarray(x)   
    
    #print(X.shape)
    #exit()
    #min_max_scaler = preprocessing.MinMaxScaler()
    #X = min_max_scaler.fit_transform(X)
    #print(X)

    if (config["data"]["train_dir"] != "") and (config["data"]["test_dir"] != ""):  
        #print("Loading from partition train/test")      
        path_data =  config["data"]["train_dir"]
        file_data =  config["data"]["train_file"]
        X_train, Y_train = load_subsets(config, enc, path_data, file_data)

        path_data =  config["data"]["test_dir"]
        file_data =  config["data"]["test_file"]
        X_test, Y_test = load_subsets(config, enc, path_data, file_data)
    else:
        X_train, X_test, Y_train, Y_test = \
                            train_test_split(X, Y, test_size=test_size,
                                    random_state = seed)
    

    #exit()
    if (True):  
        extra_file = filename+".pkl"
        try:
            f =  open(extra_file, "rb")
            #print("Attempt: loading extra data from ", extra_file)
            extra_info = pickle.load(f)
            #print("loaded")
            f.close()            

            categorical_features = extra_info["categorical_features"]
            categorical_names = extra_info["categorical_names"]
            feature_names = extra_info["feature_names"]
            class_names = extra_info["class_names"]
            #self.base_dataset  = extra_info["base_dataset"].copy()
            #categorical_onehot_names  = []#extra_info["categorical_names"].copy()
            #feature_onehot_names  = list(range(X.shape[1]))


            for i, name in enumerate(class_names):
                class_names[i] = str(name).replace("b'","'")
    
            
            for c in categorical_names.items():
                clean_feature_names = []
                clean_onehot_feature_names = []
                for i, name in enumerate(c[1]):
                    name = str(name).replace("b'","'")
                    #print(i, name)
                    clean_feature_names.append(name)
                    #categorical_onehot_names.append(str(c[0]) + "_" + name + ": " + str(i))
                    categorical_names[c[0]] = clean_feature_names
                #categorical_onehot_names[c[0]] = clean_onehot_feature_names
                #print(self.categorical_names[c[0]])
                #print(self.categorical_onehot_names[c[0]])                
            #print(self.categorical_names)
    
            #print(self.categorical_features)
            #print(self.categorical_names)
        except Exception as e:   
            print("Please provide info about categorical features or omit option -c", e)
            f.close()             
            exit()
    aggregated_data = {}
    #print(len(categorical_onehot_names))
    #exit()
    aggregated_data["X"] = X
    aggregated_data["Y"] = Y
    aggregated_data["X_train"] = X_train
    aggregated_data["X_test"] = X_test
    aggregated_data["Y_train"] = Y_train
    aggregated_data["Y_test"] = Y_test
    aggregated_data["categorical_features"] = categorical_features
    aggregated_data["categorical_names"] = categorical_names
    aggregated_data["class_names"] = class_names
    aggregated_data["feature_names"] = feature_names
    aggregated_data["one_hot_size"] = one_hot_size    
    aggregated_data["encoder"] = enc
    #aggregated_data["feature_onehot_names"] = feature_onehot_names
    
    
    

    return aggregated_data


