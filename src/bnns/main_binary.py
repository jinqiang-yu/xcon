import argparse
import os
from utils import *
from datetime import datetime
import numpy as np 
import json
import random    
from main_routine import main_load, main_train, main_encode, evaluate_sample
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


from main_routine import main_load, main_train, main_encode

data = {}
data["tae"] = {}
data["bupa"] = {}

data["tae"]["lr"] = 0.07
data["bupa"]["lr"] = 0.025


data["tae"]["layers"] = [15, 10, 5, 3]
data["bupa"]["layers"] = [10, 5, 5, 2]


 

def fill_config(args, config):



    train_file = os.path.basename(args.train)
    train_dirname = os.path.dirname(args.train)

    test_file = os.path.basename(args.test)
    test_dirname = os.path.dirname(args.test)

    data_file = os.path.basename(args.data)
    data_dir = os.path.dirname(args.data)

    config["data"]["train_dir"]  = train_dirname + "/"
    config["data"]["train_file"] = train_file
    config["data"]["test_dir"]   = test_dirname + "/"
    config["data"]["test_file"]  = test_file
    config["data"]["data_dir"]   = data_dir + "/"
    config["data"]["data_file"]  = data_file

    name = train_dirname.split("/")[-1]
    config["name"]  = name


    config["data"]["dataset"]  = name
    if not (args.results is None):
        config["save_dir"]  =  args.results
    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])

    datapath = os.path.join(data_dir, data_file)
    print(f"Loading from {datapath}")
    pdata = pd.read_csv(datapath)
    
    print( (pdata.iloc[: , -1].value_counts()).size)
    config["data"]["num_classes"]  = (pdata.iloc[: , -1].value_counts()).size
    config["data"]["input_size"]   = pdata.iloc[1,:].size-1
    config["model"]["layers"][-1]   = config["data"]["num_classes"] 
    
    if name in data:
        if "lr" in data[name]:
            config["train"]["lr"] =  data[name]["lr"]
            config["model"]["layers"]=  data[name]["layers"]

    







    return config
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
    parser.add_argument('-c', '--config', default=None, type=str,
                            help='config file path (default: None)')
    parser.add_argument('-l', '--load', default=None, type=str,
                            help='load stored model from a dir')
    parser.add_argument('-e', '--encode', type=bool,
                  help='encoder to CNF')    
    parser.add_argument('-n', '--neighprec', type=int, default= -1, 
                  help='neighbourhood prec') 

    parser.add_argument('-a', '--train', type=str, default= None, 
                  help='train dataset')    
    parser.add_argument('-t', '--test', type=str, default= None, 
                  help='test dataset')    
    parser.add_argument('-d', '--data', type=str, default= None, 
                  help='original dataset')  
                  
    parser.add_argument('-r', '--results', type=str, default= None, 
                  help='saved results')    

    parser.add_argument('-i', '--id', type=int,  default= 0, 
                  help='sample id')        
    args = parser.parse_args()
    
    if not (args.load is None):
            #load model
        model_dir = os.path.basename(args.load)
        try:
            config = json.load(open(os.path.join(args.load, CONFIG_DEFAULT_FILE_NAME)))
            random.seed(config["manual_seed"])

        except Exception as e:
            print("Error in reading {} from {}, error {}".format(args.config, args.load, e))
            exit()
            
        model, train_loader, val_loader, aggregated_data = main_load(args, config)
        if (args.encode):
            main_encode(args, config, model, train_loader, val_loader, aggregated_data, True)
        
            
    else:
        if (args.config is None):
            print("Please specify a configuration file")
            exit()

        if (args.train is None):
            print("Please specify a train file")
            exit()

        if (args.test is None):
            print("Please specify a test file")
            exit()

        if (args.data is None):
            print("Please specify a alll data file")
            exit()

        try:
            config = json.load(open(args.config))
            random.seed(config["manual_seed"])
            config = fill_config(args, config)

            main_train(args, config)
        except Exception as e:
            print("Error in reading {}, error {}".format(args.config, e))
            exit()

#    if args.evaluate:
#        save_path = '/tmp'
    




# parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
#                     help='results dir')
# parser.add_argument('--save', metavar='SAVE', default='',
#                     help='saved folder')
# parser.add_argument('--num_classes', metavar='num_classes', default=10,
#                     help='num_classes')
# parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
#                     help='dataset name or folder')
# parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
#                     choices=model_names,
#                     help='model architecture: ' +
#                     ' | '.join(model_names) +
#                     ' (default: alexnet)')
# parser.add_argument('--input_size', type=int, default=None,
#                     help='image input size')
# parser.add_argument('--model_config', default='',
#                     help='additional architecture configuration')
# parser.add_argument('--type', default='torch.cuda.FloatTensor',
#                     help='type of tensor - e.g torch.cuda.HalfTensor')
# parser.add_argument('--gpus', default='1',
#                     help='gpus used for training - e.g 0,1,3')
# parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
#                     help='number of data loading workers (default: 8)')
# parser.add_argument('--epochs', default=100, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=1024, type=int,
#                     metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
#                     help='optimizer function used')
# parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
#                     metavar='LR', help='initial learning rate')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--print-freq', '-p', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
#                     help='evaluate model FILE on validation set')
# parser.add_argument('-l', '--load', type=str, metavar='FILE',
#                     help='load model FILE')


    # optionally resume from a checkpoint
#     if args.evaluate:
#         if not os.path.isfile(args.evaluate):
#             parser.error('invalid checkpoint: {}'.format(args.evaluate))
#         checkpoint = torch.load(args.evaluate)
#         model.load_state_dict(checkpoint['state_dict'])
#         logging.info("loaded checkpoint '%s' (epoch %s)",
#                      args.evaluate, checkpoint['epoch'])
#     elif args.resume:
#         checkpoint_file = args.resume
#         if os.path.isdir(checkpoint_file):
#             results.load(os.path.join(checkpoint_file, 'results.csv'))
#             checkpoint_file = os.path.join(
#                 checkpoint_file, 'model_best.pth.tar')
#         if os.path.isfile(checkpoint_file):
#             logging.info("loading checkpoint '%s'", args.resume)
#             checkpoint = torch.load(checkpoint_file)
#             args.start_epoch = checkpoint['epoch'] - 1
#             best_prec1 = checkpoint['best_prec1']
#             model.load_state_dict(checkpoint['state_dict'])
#             logging.info("loaded checkpoint '%s' (epoch %s)",
#                          checkpoint_file, checkpoint['epoch'])
#         else:
#             logging.error("no checkpoint found at '%s'", args.resume)
# 
#     elif args.load:
#         if not os.path.isfile(args.load):
#             parser.error('invalid checkpoint: {}'.format(args.load))
#         checkpoint = torch.load(args.load)
#         model.load_state_dict(checkpoint['state_dict'])
#         model.eval()
#         logging.info("loaded checkpoint '%s' (epoch %s)",
#                      args.load, checkpoint['epoch'])
#         


   
   
    
    #criterion.type(config["model"]["type"])




# 
# 
#     if args.evaluate:
#         validate(val_loader, model, criterion, 0)
#         return
#     
# 
# 
#     if (args.load):
#         #compute explanations
#         #sample = [1, 4, 4, 2, 1, 4, 2, 0, 2, 2, 1, 9]
#         for i, (inputs, target) in enumerate(train_loader):
#             for k, sample in enumerate(inputs):
#                 anchor_call(args, aggregated_data, model, sample)
# 
#         return