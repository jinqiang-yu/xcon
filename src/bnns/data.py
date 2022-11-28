import os
import time
#import torchvision.datasets as datasets
#import torchvision.transforms as transforms
#import datasets


from datasets import  tabulardataset, prepare_tabular


_DATASETS_MAIN_PATH = './bench'
_dataset_path = {
    'zoo': os.path.join(_DATASETS_MAIN_PATH, 'zoo'),
    'zoo2': os.path.join(_DATASETS_MAIN_PATH, 'zoo2'),
    'adult': os.path.join(_DATASETS_MAIN_PATH, 'adult'),
    'german': os.path.join(_DATASETS_MAIN_PATH, 'german'),
    'lending': os.path.join(_DATASETS_MAIN_PATH, 'lending'),
    'recidivism': os.path.join(_DATASETS_MAIN_PATH, 'recidivism'),
    'propublica': os.path.join(_DATASETS_MAIN_PATH, 'propublica')
}

_dataset_file = {
    'zoo':  'zoo_data.csv',
    'zoo2':  'zoo_data.csv',
    'adult':  'adult_data.csv',
    'german':  'german_data.csv',
    'lending':  'lending_data.csv',
    'recidivism': 'recidivism_data.csv',
    'propublica': 'propublica_fairml_data.csv'
}

def prepare_dataset(config):
    return prepare_tabular(config)
    

def get_dataset_from_data(name, X, Y):
    if (True):
    #if name in {'zoo2', 'adult', 'german', 'lending', 'recidivism', 'propublica', 'cancer'}:
        #
        tabulardata = tabulardataset(X, Y)
        return tabulardata