import argparse
import os
import time
import logging
import torch
import torch.nn as nn
#import torch.nn.parallel
#import torch.backends.cudnn as cudnn
from pysat.card import *

import torch.optim

import torch.utils.data


import localmodels


from torch.autograd import Variable

from pysat.solvers import Solver  # standard way to import the library



from data import prepare_dataset, get_dataset_from_data

from preprocess import get_transform
from utils import *
from datetime import datetime
#from ast import literal_eval
#from torchvision.utils import save_image
from datasets import  tabulardataset, prepare_tabular
import numpy as np 
#from anchor_wrap import anchor_call
from encoding import EncodingBNN
import pickle

import json
import random    
import shlex, subprocess

model_names = sorted(name for name in localmodels.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(localmodels.__dict__[name]))

def solve_formula(formula, assumptions = []):
        s_test = Solver(name='g3')
        s_test.append_formula(formula.clauses)
        #print(self.formula.clauses)
        s_test.solve(assumptions = assumptions)
        solution = s_test.get_model()  
        if (solution is None):
            #print("UNSAT")
            return None
                
        #print("SOLVED")
        return  solution           
   

def evaluate_sample(sample, model, aggregated_data, config):
    model.eval()
    sample = sample.cpu().detach().numpy()
    input_encoder  = aggregated_data["encoder"]        
    bin_sample = []
    #bin_sample.append(input_encoder.transform([sample])[0])
    bin_sample.append(input_encoder.transform([sample])[0])    
    bin_sample = np.asarray(bin_sample)        
    output = model.forward((torch.FloatTensor(bin_sample)).type(config["model"]["type_model"]))
    _, pred = output.float().topk(1, 1, True, True)

    return (pred[0].cpu().detach().numpy())[0]
            
def model_load(config):
    printing = False
    logging.info("creating model %s", config["model"]["name"])
    model = localmodels.__dict__[config["model"]["name"]]    
    model = model(config)
    logging.info("created model with configuration: %s", config["model"])
    if (printing):
        print(model)        
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)
    model.type(config["model"]["type_model"])

    # define loss function (criterion) and optimizer    
    if (config["train"]["loss"] in "BCELoss"):
        criterion = nn.BCELoss()
    if (config["train"]["loss"] in "CrossEntropyLoss"):
        criterion = nn.CrossEntropyLoss()
        
    criterion.type(config["model"]["type_model"])
    
    return model, criterion 

def data_load(config):
    aggregated_data = prepare_dataset(config)
    

    val_data = get_dataset_from_data(config["data"]["dataset"], aggregated_data["X_test"], aggregated_data["Y_test"])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config["train"]["batch_size"], shuffle=False,
        num_workers=config["train"]["workers"], pin_memory=True)

    train_data = get_dataset_from_data(config["data"]["dataset"], aggregated_data["X_train"], aggregated_data["Y_train"])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config["train"]["batch_size"], shuffle=True,
        num_workers=config["train"]["workers"], pin_memory=True)
    
    if(config["data"]["use_one_hot"]):
        config["data"]["input_size"] = aggregated_data["X"].shape[1]   

    return train_loader, val_loader, aggregated_data

def run_train(config, model, criterion, train_loader, val_loader, results, save_path):
    if ( config["train"]["optimizer"] == "SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr= config["train"]["lr"], 
                                momentum = config["train"]["momentum"],
                                weight_decay = config["train"]["weight_decay"])

    if ( config["train"]["optimizer"] == "Adam"):
        optimizer = torch.optim.Adam(model.parameters(), 
                                lr = config["train"]["lr"], 
                                betas=[0.5, 0.999])
    logging.info('training regime: %s', model.regime)
    best_prec1 = 0
    for epoch in range(config["train"]["epochs"]):
        
        for W in list(model.parameters()):
                if(len(W.shape) < 2):
                    continue
                if hasattr(W,'org'):
                    t =  1
                    for i, v in enumerate(W.shape):
                        t = v * t  
                    print(t - (torch.abs(W.org) <config["train"]["small_weight"]).sum() , W.org.shape)
                    #print(W[0])
                    print(t - (torch.abs(W) < config["train"]["small_weight"]).sum() , W.shape)
                    #print(sW)
                    
                
        optimizer = adjust_optimizer(optimizer, epoch, model.regime)
        if (epoch == 0):
            print(optimizer)
        # train for one epoch
        train_loss, train_prec1 = train(config, train_loader, model, criterion, epoch, optimizer)

        # evaluate on validation set
        val_loss, val_prec1 = validate(config, val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = (val_prec1 > best_prec1) and (epoch > config["train"]["epochs"]*0.8)
        if (epoch > config["train"]["epochs"]*0.8):
            best_prec1 = max(val_prec1, best_prec1)
        print(save_path)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config["model"]["name"],
            'config': config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': model.regime
        }, is_best, path=save_path)
        
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1))

        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1*100, val_error1=100 - val_prec1*100)
        results.save()



   
def forward(config, data_loader, model, criterion, epoch=0,  training=True, optimizer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        #print(i)
        if (config["train"]["loss"] in "BCELoss"):
            target = target.float()
        if (config["train"]["loss"] in "CrossEntropyLoss"):
            target = target.view(target.shape[0])
            target = torch.LongTensor(target)
        #print(target)
        # measure data loading time
        data_time.update(time.time() - end)
        if config["train"]["gpus"] != -1:
            target = target.cuda()
        input_var = Variable(inputs.type(config["model"]["type_model"]), volatile=not training)
        target_var = Variable(target)
        # compute output

        output = model.forward(input_var)#(epoch > max_epoch*0.2))

        #output1 = Variable(torch.cuda.FloatTensor([[0,0],[0,1],[0,1]])).view(3,2)
        
        #target1 = Variable(torch.cuda.LongTensor([0,1,1]))

        #criterion1 = nn.CrossEntropyLoss()
        #print(output1, target1)
        #print(output1.shape, target1.shape)
        #print(type(output1), type(target1))
        #print(criterion)
        #loss = criterion(output, target)
        #print(loss)
        
        #print(output.shape, target_var.shape)
        #print(output, target_var)
        #print(type(output1), type(target_var))
        loss = criterion(output, target_var)
        #exit()

        if type(output) is list:
            output = output[0]
        
        # measure accuracy and record loss
        if (config["train"]["loss"] in "BCELoss"):  
            prec1  = bce_accuracy(output.data, target)
        if (config["train"]["loss"] in "CrossEntropyLoss"):
            prec1  = accuracy(output.data, target)
             
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        #top5.update(prec5, inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config["train"]["print_freq"] == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1))

    return losses.avg, top1.avg


def train(config, data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(config, data_loader, model, criterion, epoch, training=True, optimizer=optimizer)


def validate(config, data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(config, data_loader, model, criterion, epoch, 
                   training=False, optimizer=None)



def main_train(args, config):

    #args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(config["save_dir"])#, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config_file = os.path.join(save_path, CONFIG_DEFAULT_FILE_NAME)

    with open(config_file, 'w') as f:
        json.dump(config,  f, indent=4, sort_keys=True,)
    
        
    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in config["model"]["type"]:
        torch.cuda.set_device(config["train"]["gpus"])
        cudnn.benchmark = True
        print("using cuda")

    # load data
    train_loader, val_loader, _ =  data_load(config)


    # create model
    model, criterion = model_load (config)

    #run training
    run_train(config, model, criterion, train_loader, val_loader, results, save_path)
        
def main_execute_sample(config, model, sample, aggregated_data, extra_information):
    model.eval()   
    printing = False
    samples  = [sample.clone().cpu().detach().numpy()]
    input_encoder  = aggregated_data["encoder"]
    bin_sample = []
    bin_sample.append(input_encoder.transform([samples[0]])[0])
    bin_sample.append(input_encoder.transform([samples[0]])[0])    
    bin_sample = np.asarray(bin_sample)
    if (printing):        
        print("Unary sample: ", bin_sample[0])
    
    
    cuda_sample = torch.tensor(bin_sample).type(config["model"]["type_model"])            
    #print(cuda_sample)
    #exit()
    result  = model(cuda_sample)
    _, pred = result.float().topk(1, 1, True, True)
    winner = pred.cpu().detach().numpy()[0][0]
    #print("Prediction: {}". format(winner))
    
    winner_lit = extra_information["tails"][winner]              
    
    if (len(model.encoder.winners_lits) == 2):       
        loser_lit =  extra_information["tails"][1 - winner]
    else:
        loser_lit = build_loser_lit(model, winner)

    if (printing):                
        print(winner, winner_lit, loser_lit, extra_information["tails"])
    extra_info_sample = {}
    extra_info_sample["makeunsat"] = [loser_lit]

    first_layer_vars = extra_information["first_layer_vars"]
    
    input_assump = []
    #print(inputs)
    for i, var_id in enumerate(first_layer_vars):
        if (bin_sample[0][i] > 0):
            input_assump.append(var_id)
        else:
            input_assump.append(-var_id)      
    
    extra_info_sample["input2lits"] = input_assump
    extra_info_sample["winner_lit"] = winner_lit          
    extra_info_sample["loser_lit"] = loser_lit   
    return extra_info_sample   
    
def main_encode_sample(args, config , model, sample, aggregated_data, save_path = None, card_enc = CARD_ENC_SEQ_COUNT, is_checking = False):
    
    model.eval()   
    model.card_encoding = card_enc

    solver_name  = "g3"
    if (card_enc == CARD_ENC_NAIVE):
        solver_name  = "minicard"    
    samples  = [sample.clone().cpu().detach().numpy()]
    if (is_checking):
        print("\n Working with new sample............. {}".format(sample))
    #target_sample = target[p]
    #print(samples)#,  target_sample)
    
    

    
    #############################            
    #Input variables
    ############################
    
    # We assume unary encoding
    #print(aggregated_data)
    enc =  EncodingBNN(config)

    #continue
    #############################
    # create variables
    # We assume layer-wise structure
    ############################
    enc.create_variables_by_layers(model.get_number_neurons())
    #unary_over_inputs
    categorical_names = aggregated_data["categorical_names"]
    categorical_features = aggregated_data["categorical_features"]
    feature_names= aggregated_data["feature_names"]
    point_id_layer_0 = 0
    total_solutions = 1
    
    unary_literals_per_var = []
    
    #print(aggregated_data)
    #exit()
    enc.extra_information = {}
    enc.extra_information["features2vars"] = {}
    enc.extra_information["input2lits"] = {}
    
    total_solutions = 1
    for i, (cat_names, values) in enumerate(categorical_names.items()):
        #print(cat_names, values)
        nb_in_unary =  len(values)
        vars = list(range(point_id_layer_0, point_id_layer_0 + nb_in_unary))
        
        #####################
        #Unary encodign of inputs
        #####################
        literals = enc.unary_over_inputs(vars)        
        unary_literals_per_var.append(literals)
        #####################
        #print(vars, cat_names, values)
        enc.extra_information["features2vars"][categorical_features[i]] = {}
        enc.extra_information["features2vars"][categorical_features[i]]["name"] = feature_names[i]
        enc.extra_information["features2vars"][categorical_features[i]]["vars"] = literals
        enc.extra_information["features2vars"][categorical_features[i]]["onhotlabels"] = values
        
        point_id_layer_0 += nb_in_unary
        total_solutions =  total_solutions * nb_in_unary
    enc.extra_information["total_solutions"] = total_solutions
    #print(vars)
    #print("Total inputs", total_solutions)
    #print(enc.mapping_features2vars)
    #enc.all_solutions()
    #exit()
    #############################################
    # encode network
    #############################################
    
    # get a sample for checking
   



    
    # we binarize and duplicate sample to be able to run it through network
    input_encoder  = aggregated_data["encoder"]    
    bin_sample = []
    bin_sample.append(input_encoder.transform([samples[0]])[0])
    bin_sample.append(input_encoder.transform([samples[0]])[0])    
    bin_sample = np.asarray(bin_sample)  
    if (is_checking):
        print("Unary sample: ", bin_sample[0])
    
    
    cuda_sample = torch.tensor(bin_sample).type(config["model"]["type_model"])            
    result  = model(cuda_sample)  
    _, pred = result.float().topk(1, 1, True, True)
    winner = pred.cpu().detach().numpy()[0][0]
    if (is_checking):
        print("Prediction: {}". format(winner))
    #if (winner != target_sample.item()):
    #    continue
                
    model.encoder  = enc
    
    outputs_by_layers = model.forward_encode(cuda_sample, winner = None)
    #exit()             

    if (is_checking):
        print("Checking...................  layers (SAT)")
        orig_formula  = enc.formula.copy()                 
        #enc.formula.to_file("test41.cnf")                 
        #orig_formula.to_file("test42.cnf")
                
        sol = enc.solve_formula(orig_formula, name =  solver_name)
        assert(sol != None)
        nb_layers = len(model.get_number_neurons())
        for i in range(1):
            vars_per_layer = enc.get_vars_per_layer(i)
            sum_pos  =  0 
            for j, v in enumerate(vars_per_layer):
                bin_v = int((sol[v-1] > 0))
                #print(bin_v, sol[v-1])
                sum_pos = sum_pos + bin_v
            assert(sum_pos == len(samples[0]))

        # for j, v in enumerate(enc.get_vars_per_layer(nb_layers-1)):                    
        #     print("last layer",  j, sol[v-1])                               
            
        print("OK")
    
    enc.extra_information["winners"] = model.encoder.winners_lits
    #assert(not (model.winner_formula is None) )
    #assert(not (model.loser_formula is None) )
    first_layer_vars = enc.get_vars_per_layer(0)
    
    
  
    ### file
    if not (save_path is None):
        str_sample = '_'.join(str(e) for e in  samples[0])
        save_path  = save_path + "/" + str_sample + "/"            
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        first_str = "c ind {} 0\n".format(' '.join(str(e) for e in first_layer_vars) )               
        file_name = '{}/sample.cnf'.format(save_path)            
        model.encoder.formula.to_file(file_name)
        with open(file_name, 'r') as original: data = original.read()
        with open(file_name, 'w') as modified: modified.write( first_str + data)

    #################################################33
    # Assumptions
    ####################################################    

    #model.encoder.formula = loser_formula
    #model.encoder.formula = enc.formula.copy()                        
    inputs = outputs_by_layers[0]
    input_assump = []
    #print(inputs)
    for i, var_id in enumerate(first_layer_vars):
        if (inputs[i] > 0):
            input_assump.append(var_id)
        else:
            input_assump.append(-var_id)                
    #print(input_assump)
    #print(outputs_by_layers[0])
    #################################################33
    # Make to lose
    #################################################### 
    enc.extra_information["input2lits"] = input_assump
    winner_lit = model.encoder.winners_lits[winner]              


    #print(winner_lit)
    #loser_lit = model.encoder.winners_lits[1 - winner]       

    #print(model.encoder.winners_lits)
    #exit()
    #print(f"---------> winner {winner}")
    if (len(model.encoder.winners_lits) == 2):       
        loser_lit = model.encoder.winners_lits[1 - winner]         
    else:
        loser_lit = build_loser_lit(model, winner)


      
    if (is_checking):
        print(winner, winner_lit, loser_lit, enc.winners_lits)
    
    enc.extra_information["makeunsat"] = [loser_lit]
    enc.extra_information["tails"] = model.encoder.winners_lits
    enc.extra_information["first_layer_vars"] = first_layer_vars               
    enc.extra_information["winner_lit"] = winner_lit           
    enc.extra_information["loser_lit"] = loser_lit  
    enc.extra_information["top_id_cnf"] =  model.encoder.vpool        

    # enc.extra_information["makeunsat"] = [loser_lit]
    # model.encoder.winners_lits = {}
    # model.encoder.winners_lits["winner"] = winner_lit
    # model.encoder.winners_lits["loser"] = loser_lit
    # enc.extra_information["tails"] = model.encoder.winners_lits
    # enc.extra_information["first_layer_vars"] = first_layer_vars               
    # enc.extra_information["winner_lit"] = winner_lit
    # enc.extra_information["loser_lit"] = loser_lit              
    # enc.extra_information["top_id_cnf"] =  model.encoder.vpool        

    #################################################33
    # Checking
    ####################################################
    #print("Checking...................")
    checking  = is_checking
    checking_non_winner = is_checking
    checking_winner = is_checking


        
    if (checking_winner):
        print("Checking................... winner(should be SAT)")
        
        orig_formula  = enc.formula.copy()
        #enc.formula.to_file("test51.cnf")                 
        #orig_formula.to_file("test52.cnf")
        #exit()

        
        #prob_sol = [-1, -3, -4, -6, -12, -20, -26, -28, -32, -37, -38, -41, -47, 52, -53, -59, -62, -63, -69]
        #for a in prob_sol:
        #    orig_formula.append([a]) 
        
        #print(input_assump)
        sol = enc.solve_formula(orig_formula, assumptions = input_assump, name =  solver_name)
        assert(sol != None)
        nb_layers = len(model.get_number_neurons())
        #testvars = list(abs(np.asarray([-1, -3, -4, 6, -12, 20, 26, -28, -32, -37, -38, 41, 47, -52, 53, -59, 62, 63, -69, -69, -69, -69, -69, -69, -69, -69, -69, -69])))
        #testvars_sign = [-1, -3, -4, 6, -12, 20, 26, -28, -32, -37, -38, 41, 47, -52, 53, -59, 62, 63, -69, -69, -69, -69, -69, -69, -69, -69, -69, -69]
        #test_sum = 0
        #test_ass = []
        #orig_formula.to_file("test1.cnf")
        for i in range(nb_layers-1):
            vars_per_layer = enc.get_vars_per_layer(i)
            layer_act = outputs_by_layers[i]
            for j, v in enumerate(vars_per_layer):
                bin_v = int((sol[v-1] > 0))
                #if (i == 1):
#                 if v in testvars:
#                     ind = testvars.index(v)
#                     t = testvars_sign[ind]
#                     
#                     print(bin_v, layer_act[j], sol[v-1], t)
#                     test_ass.append(sol[v-1])
#                     if (sol[v-1] < 0) and t < 0:
#                         test_sum = test_sum + 1
#                     if (sol[v-1] > 0) and t > 0:    
#                         test_sum = test_sum + 1                    
                assert(bin_v == layer_act[j])
            # for j, v in enumerate(vars_per_layer):                
            #     bin_v = int((sol[v-1] > 0))
            #     print(f" v[{j}] = {bin_v} ({layer_act[j]})", end  = "")
            # print("--")
#                print(test_sum)                
#            print(test_ass)
            
            #-1 -3 -4 6 -12 20 26 -28 -32 -37 -38 41 47 -52 53 -59 62 63 -69 -69 -69 -69 -69 -69 -69 -69 -69 -69 <= 18
            # 1  1  1 0  1   0 0   1   1   1   1   0  0  0  0   1  0   0  1   1    1   1   1   1   1   1  1   1
            # 1 3 4 -6 12 -20 -26 28 32 37 38 -41 -47 52 -53 59 -62 -63 69 69 69 69 69 69 69 69 69 <= 18
            
            
            print("layer {} is OK".format(i))
        # chekc the last layer
        last_layer_act = outputs_by_layers[-1]
        #print("last_layer_act", last_layer_act)
        vars_per_last_layer = enc.get_vars_per_layer(nb_layers-1)
        #print("{} truth {}, prec {} ".format(vars_per_last_layer, " ", winner))
        
        # for j, v in enumerate(vars_per_last_layer):                    
        #     print(j, sol[v-1])
        #print("winner {} , lit {} should be true".format(winner, model.encoder.winners_lits[winner]))
        
        winnner_lit = model.encoder.winners_lits[winner]
        bin_v = int((sol[winnner_lit-1] > 0))      
        # s = [2454, 2467, 2486, 2495, 2498, 2522]
        # for c in s:
        #     print(f"{c} lit {sol[c-1]}")
        assert(bin_v)  



        print("last layer is OK")
        print("Tested formula using sample", samples[0])
    if (checking_non_winner):
        print("Checking................... loser(should be UNSAT)")
        nb_layers = len(model.get_number_neurons())

        
        loser_formula  = enc.formula.copy()  
        #print(loser_lit)
        loser_formula.append([loser_lit])
        
        sol = enc.solve_formula(loser_formula, assumptions = input_assump, name =  solver_name)
        if (sol != None):
            vars_per_last_layer = enc.get_vars_per_layer(nb_layers-1)
            # for j, v in enumerate(vars_per_last_layer):                    
            #     print(j, sol[v-1])                    
            assert(sol == None)
                
    #print(enc.extra_information)
    return enc.formula, enc.extra_information
def build_loser_lit(model, winner):
    card_formula = CNF()      
    #lose <-> win_class_nonwinner1 or win_class_nonwinner2 or .. 

    var_lose = model.encoder.create_indexed_variable_name("to_fail_formula", [1])
    lose_id = model.encoder.get_varid(var_lose)
    lose_id = model.encoder.lookup_varid(var_lose)        
        
    cl = [-lose_id]                 
    for cls, l in  model.encoder.winners_lits.items():
        if cls == winner:
            continue
        cl.append(l)
    card_formula.append(cl)


    cl = [lose_id]                 
    for cls, l in  model.encoder.winners_lits.items():
        if cls == winner:
            continue
        cl.append(-l)
        card_formula.append(cl)
    # for c in card_formula:
    #     print(c)
    var_lose = model.encoder.append(card_formula)
    loser_lit = lose_id    
    return loser_lit
def main_encode(args, config , model, train_loader, val_loader, aggregated_data, card_enc = CARD_ENC_SEQ_COUNT,  is_checking = False):
    save_path_orig  = args.load + "/cnfs_arround_2/"
    if not os.path.exists(save_path_orig):
        os.makedirs(save_path_orig)
    
    model.eval()
    #print("start encoding")
    countet_temp = 0
    for h, (inputs, target) in enumerate(val_loader):
        #print(h)
        #if(h < 2):
        #    continue
        #print(inputs.shape)
        for p, sample in enumerate(inputs):
            #if (p < 2):
            #    continue
            if (args.id != countet_temp):
                print("pass on ", countet_temp)
                countet_temp = countet_temp + 1
                continue
            countet_temp = countet_temp + 1
            save_path = save_path_orig
            return main_encode_sample(args, config , model, sample, aggregated_data, save_path, card_enc,  is_checking) 

def main_load(args, config):
    save_path = '/tmp'
    
    model_file = os.path.join(args.load, BEST_MODEL_DEFAULT_FILE_NAME)

    train_loader_onehot, val_loader_onehot, aggregated_data_onehot =  data_load(config)
    if(config["data"]["use_one_hot"]):
        config["data"]["input_size"] = aggregated_data_onehot["one_hot_size"] 

    #print("Loading from {}".format(model_file))
    checkpoint = torch.load(model_file)    
    model, criterion = model_load (config)
    model.load_state_dict(checkpoint['state_dict'])
    logging.info("loaded checkpoint '%s' (epoch %s)", model_file, checkpoint['epoch'])    
    config["train"]["print_freq"] = 2

    #val_loss_onehot, val_prec1_onehot = validate(config, val_loader_onehot, model, criterion, 0)
    #print('\n Validation Prec@1 {val_prec1:.3f} \t'
    #                 .format(val_prec1=val_prec1_onehot))
    
    
    
    config["data"]["donot_convert_data"] = True    
    train_loader, val_loader, aggregated_data =  data_load(config)
#     if(False):
#         for i, (inputs, target) in enumerate(train_loader):
#             for k, sample in enumerate(inputs):
#                 anchor_call(config, aggregated_data, model, sample, target[k])
#                 break
#             break
    
    return model, train_loader, val_loader, aggregated_data
