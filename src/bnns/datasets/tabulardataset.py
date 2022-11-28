
from __future__ import print_function, division
#import os
#import argparse
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torchvision import datasets, transforms
#from torch.autograd import Variable
#from torchvision.transforms import ToTensor
#from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset#, DataLoader
#from PIL import Image
#from PIL import ImageOps
#import torchvision.models as models
#from  sklearn.preprocessing import OneHotEncoder

#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from adam import Adam
#from modules import *
#from localmodels import *
#import pandas as pd 
#import os, pickle

# Ignore warnings
#import warnings
#from matplotlib.pyplot import axis
#warnings.filterwarnings("ignore")


class tabulardataset(Dataset):
    def __init__(self, X, Y, transform=None):

        self.X = X
        self.Y = Y
        #self.X = np.transpose(X)
        #encoder = OneHotEncoder(categories='auto', sparse=False)
        #encoder.fit(Y)
        #self.Y = np.transpose(encoder.transform(Y).astype(int))

        #print(pd_data.columns[:-1])
        #print(self.y)
        #print(self.X[0])
        #print([self.X[:,0], self.Y[:,0]])
        #exit()
        #print(X.shape)
        #print(Y)
        self.transform = transform

    def __len__(self):
        return self.X.__len__()

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]
