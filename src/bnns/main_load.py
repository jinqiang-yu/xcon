import time
start = time.time()
import argparse
import os
from utils import *
#from datetime import datetime
import numpy as np 
import json
import random    
#start = time.time()
from main_routine import main_load, main_train, main_encode, evaluate_sample, main_encode_sample, main_execute_sample, solve_formula
from pysat.solvers import Solver  # standard way to import the library
#end = time.time()
#print("initial", end - start)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    init_start = time.time()

    parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
    parser.add_argument('-c', '--config', default=None, type=str,
                            help='config file path (default: None)')
    parser.add_argument('-l', '--load', default=None, type=str,
                            help='load stored model from a dir')
    parser.add_argument('-e', '--encode', type=bool,
                  help='encoder to CNF')    

    parser.add_argument('-i', '--id', type=int, default= 0, 
                  help='sample index')  
        
    args = parser.parse_args()
    
    assert (not (args.load is None))
    model_dir = os.path.basename(args.load)
    try:
        config = json.load(open(os.path.join(args.load, CONFIG_DEFAULT_FILE_NAME)))
        random.seed(config["manual_seed"])
    except Exception as e:
        print("Error in reading {} from {}, error {}".format(args.config, args.load, e))
        exit()
        
    is_profile  = False
    start = time.time()
    model, train_loader, val_loader, aggregated_data = main_load(args, config)
    if (True):
        xgb = XGBClassifier()
        xgb.fit(aggregated_data["X_train"], aggregated_data["Y_train"])
        #print(model)
        y_pred = xgb.predict(aggregated_data["X_test"])
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(aggregated_data["Y_test"], predictions)
        print("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))    

    model.is_profile = is_profile
    end = time.time()
    if (is_profile):
        print("main_load: ", end - start)        

    basic_test = True
    if (args.encode):
        counter = 0
        #######################################
        # formula contains "forward path" and "one-hot encoding"
        #######################################
        start = time.time()
        test_sample = torch.tensor(val_loader.dataset.X[0])
        #print(test_sample)
        #exit()
        formula, extra_information = main_encode_sample(args, config , model, test_sample, aggregated_data, save_path = None, is_checking = False)   
        mapping = extra_information["features2vars"]
        end = time.time()
        if (is_profile):        
            print("main_encode_sample: ", end - start)        

        is_printing = True
        for h, (inputs, target) in enumerate(val_loader):
            for p, sample in enumerate(inputs):         
                
                if (args.id != counter):
                   # print("pass on ", counter)
                    counter = counter + 1
                    continue   
                print("Looking at sample {}: {} ".format(counter, sample))
                start = time.time()                
                extra_info_sample  = main_execute_sample(config, model, sample, aggregated_data, extra_information)
                end = time.time()      
                if (is_profile):                          
                    print("main_execute_sample: ", end - start)        


                if (is_printing):
                    #######################################
                    # Mapping
                    #######################################
                    for k, v in mapping.items():
                        print("feature {}, name = `{}':  vars = {}".format(k,  v["name"], v["vars"]))    
                        print("onhotlabels {}".format(v["onhotlabels"]))
                    #######################################
                    # Assignment
                    #######################################
                    assignment =  extra_info_sample["input2lits"]                    
                    print("assignment {}".format(assignment))
    
                    #######################################
                    # Not Prediction 
                    #######################################                
                    lit_to_make_prediction_false =  extra_info_sample["makeunsat"]
                    print(lit_to_make_prediction_false)

                    
                    #############################
                    # Examples
                    #############################
                    if (basic_test):
                    
                                
                        
                        print("Example: check forward path+onehot")  
                        # for cl in formula:
                        #     print(cl)
                        print(assignment)
                        sol = solve_formula(formula, assignment)     
                        if (sol == None):
                            print("Smth is wrong, forward path encoding is UNSAT")
                        else:
                            winner_lit = extra_info_sample["winner_lit"]
                            loser_lit = extra_info_sample["loser_lit"]                                    
                            print(winner_lit, sol[winner_lit-1])
                            print(loser_lit, sol[loser_lit-1])
                            assert(sol[winner_lit-1] > 0)
                            print("Pass test: Formula (forward path) is SAT, prediction lit is TRUE")
                        #exit()
                        
                        print("Example: to flip prediction add {}".format(lit_to_make_prediction_false))
                        loser_formula  = formula.copy()  
                        loser_formula.append(lit_to_make_prediction_false)     
                       #exit()           
                        # for cl in loser_formula:
                        #     print (cl)
                        sol = solve_formula(loser_formula, assignment)                
                        if (sol == None):
                            print("Pass test: Formula is UNSAT")
                            pass
                        else:
                            # Note it might be the case that we have a tie In this case, formula will not be UNSAT  
                            # we need to ignore this sample   
                            winner_lit = extra_info_sample["winner_lit"] 
                            loser_lit = extra_info_sample["loser_lit"]                    
                            if (int((sol[winner_lit-1] > 0)) and int((sol[loser_lit-1] > 0))):                                           
                                print("We have a tie, ignore this sample")
                                exit()
                            else:
                                assert(sol == None)                                            
                if (args.id == counter):
                    init_end = time.time()
                    print("Total time = {}".format(init_end - init_start))
                    exit()
                                    
                counter = counter + 1
                         

    
    cnt = 0
    cnt_hit = 0
    for h, (inputs, target) in enumerate(val_loader):
        for p, sample in enumerate(inputs):
            output  = evaluate_sample(sample, model, aggregated_data, config)
            cnt +=1
            if (target[p][0].item() == output):
               cnt_hit += 1 
            print("true {}, predict {} (acc {})".format(target[p][0].item(), output, cnt_hit/cnt))
            
    #for i in range(200):
    #    print("python3.6 bnn_explained/main_load.py --load ./bnn_explained/results_adult/2020-07-26_14-05-32/ --encode  True --id {}; ".format(i))
    #exit()            