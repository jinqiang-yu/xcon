#!/usr/bin/env python
#-*- coding:utf-8 -*-

#
#==============================================================================
from __future__ import print_function
import argparse
import json
import statistics
import os
import sys
import numpy as np
import pandas as pd
import pickle
import time
import resource
import csv
import random
import lime
import lime.lime_tabular
import shap
#from anchor import utils
from anchor import anchor_tabular
from itertools import chain
import torch
from bnns.localmodels.mlp_binary import MLPNetOWT_BN
import torch.nn as nn
import math
from bnns.main_routine import data_load

#
#==============================================================================

class HExplainer(object):
    #HeuristicExplainer
    def __init__(self, global_model_name, appr, X_train, y_train, model):
        self.global_model_name = global_model_name
        self.appr = appr
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.init_explainer(appr)

    def init_explainer(self, appr):
        if appr.lower() == 'lime':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,
                                                               # feature_names=self.X_train.columns,
                                                               discretize_continuous=False)
            ###
            #explainer = lime.lime_tabular.LimeTabularExplainer(
            #    xgb.X_train,
            #    feature_names=xgb.feature_names,
            #    categorical_features=xgb.categorical_features if xgb.use_categorical else None,
            #    class_names=xgb.target_name,
            #    discretize_continuous=True,
            #)
            ###
        elif appr.lower() == 'shap':
            self.explainer = shap.Explainer(self.model,
                                            # feature_names=self.X_train.columns,
                                            self.X_train)

        elif appr.lower() == 'anchor':
            self.explainer = anchor_tabular.AnchorTabularExplainer(
                class_names=[False, True],
                feature_names=self.X_train.columns,
                train_data=self.X_train.values,
                categorical_names={})
        else:
            print('Wrong approach input')
            exit(1)

    def explain(self, X, y):
        pred = self.model.predict(X)[0]

        inst = X.iloc[0]
        preamble = []
        for fid, f in enumerate(inst.index):
            preamble.append(f'{f} = {inst[fid]}')

        print('\n  Explaining: IF {} THEN defect = {}'.format(' AND '.join(preamble), pred))

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime

        if self.appr.lower() == 'lime':
            self.lime_explain(X, y, pred)
        elif self.appr.lower() == 'shap':
            self.shap_explain(X, y, pred)
        elif self.appr.lower() == 'anchor':
            self.anchor_explain(X, y, pred)

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        print(f'  time: {self.time}\n')

    def lime_explain(self, X, y, pred):
        #predict_fn = lambda x: self.model.predict_proba(x).astype(float)

        expl = self.explainer.explain_instance(X.iloc[0, :],
                                          self.model.predict_proba, # prediction function ( predict probability)
                                          # num_features=10, 10 is the default value
                                          top_labels=1)

        prob0, prob1 = self.model.predict_proba(X)[0]
        pred = False if prob0 > prob1 else True
        expl = sorted(expl.as_list(label=int(pred)), key=lambda l: int(l[0]))

        if prob0 == prob1:
            # Reverse the direction of feature importance
            # Since when prob0 = prob1, the target class value is class 1 in the explainer,
            # where the predicted value in the global model is class 0
            expl = list(map(lambda l: (l[0], -l[1]), expl))


        y_expl = list(filter(lambda l: l[1] >= 0, expl))
        ynot_expl = list(filter(lambda l: l[1] < 0, expl))
        print('  expl(pos class):', [(self.X_train.columns[int(f)], imprt) for f, imprt in y_expl])
        print('  size:', len(y_expl))
        print('  expl(neg class):', [(self.X_train.columns[int(f)], imprt) for f, imprt in ynot_expl])
        print('  size:', len(ynot_expl))

        # print('  explanation: IF {0} THEN defect = {1}'. format(' AND '.join(preamble), pred))
        # print('  importance:', importance)
        #print('  size: {0}'.format(len(preamble)))

        #if prob0 == prob1:
        #    exit()

    def shap_explain(self, X, y, pred):
        shap_values = self.explainer.shap_values(X)
        shap_values_sample = shap_values[int(pred)][0] if self.global_model_name == 'RF' else shap_values[0]

        predicted_value = [round(self.explainer.expected_value[idx] + np.sum(shap_values[idx]), 3)
                           for idx in range(len(self.explainer.expected_value))] \
            if self.global_model_name == 'RF' else np.sum(shap_values_sample) + self.explainer.expected_value

        print("base_value = {}, predicted_value = {}".format(self.explainer.expected_value, predicted_value))
        expl = [(idx, shap_values_sample[idx]) for idx in range(len(shap_values_sample))]

        y_expl = list(filter(lambda l: l[1] >= 0, expl))
        ynot_expl = list(filter(lambda l: l[1] < 0, expl))
        print('  expl(pos class):', [(self.X_train.columns[int(f)], imprt) for f, imprt in y_expl])
        print('  size:', len(y_expl))
        print('  expl(neg class):', [(self.X_train.columns[int(f)], imprt) for f, imprt in ynot_expl])
        print('  size:', len(ynot_expl))

    def anchor_explain(self, X, y, pred):
        exp = self.explainer.explain_instance(X.values[0], self.model.predict, threshold=0.95)

        # explanation
        expl = [name for f, name in sorted(zip(exp.features(), exp.names()))]

        preamble = ' AND '.join(expl)

        print('  expl: IF {0} THEN defect = {1}'.format(preamble, pred))
        print('  size:', len(expl))
        #print('  Anchor: %s' % (' AND '.join(exp.names())))
        #print('  Precision: %.2f' % exp.precision())
        #print('  Coverage: %.2f' % exp.coverage())

#
#==============================================================================
def parse_options():
    """
        Basic option parsing.
    """

    def group_to_range(group):
        group = ''.join(group.split())
        sign, g = ('-', group[1:]) if group.startswith('-') else ('', group)
        r = g.split('-', 1)
        r[0] = sign + r[0]
        r = sorted(int(__) for __ in r)
        return range(r[0], 1 + r[-1])

    def rangeexpand(txt):
        ranges = chain.from_iterable(group_to_range(_) for _ in txt.split(','))
        return sorted(set(ranges))

    parser = argparse.ArgumentParser(description='Heuristic explainer')
    parser.add_argument('-a', '--approach', type=str, default=None,
                        help='Whether extracting useful rules')
    parser.add_argument('-I', '--inst', type=int, default=100,
                        help='The number of instances being computed (default: 100)')
    parser.add_argument('-l', '--load', default=None, type=str,
                            help='Load stored model from a dir')
    parser.add_argument('-m', '--model', default=None, type=str,
                        help='Model name (default: false)')
    parser.add_argument('-t', '--train', default=None, type=str,
                        help='Train dataset (default: false)')
    parser.add_argument('-T', '--test', default=None, type=str,
                        help='Test dataset (default: false)')
    ret = parser.parse_args()

    # multiple samples
    #ret.ids = rangeexpand(ret.ids)

    # casting xnum to integer
    #ret.xnum = -1 if ret.xnum == 'all' else int(ret.xnum)

    return ret

if __name__ == '__main__':
    args = parse_options()
    # python hexp.py -a lime -I 100
    # -m bt
    # -l ./bt/btmodels/q6/appendicitis_train1_data/appendicitis_train1_data_nbestim_25_maxdepth_3_testsplit_0.2.mod.pkl
    # -t ../bench/cv/train/quantise/q6/penn-ml/appendicitis/appendicitis_train1_data.csv
    # -T ../bench/cv/test/quantise/q6/penn-ml/appendicitis/appendicitis_test1_data.csv

    appr = args.approach
    nof_inst = args.inst
    model_path = args.load
    model_name = args.model
    train = args.train
    test = args.test

    if args.model == 'bnn':
        config = json.load(open(os.path.join(args.load, 'config.json')))
        #loading the model
        '''
        def load(args, config):

            model_file = os.path.join(args.load, 'model_best.pth.tar')

            train_loader_onehot, val_loader_onehot, aggregated_data_onehot = data_load(config)

            if (config["data"]["use_one_hot"]):
                config["data"]["input_size"] = aggregated_data_onehot["one_hot_size"]

                # print("Loading from {}".format(model_file))
            checkpoint = torch.load(model_file)

            model, criterion = model_load(config)
            model.load_state_dict(checkpoint['state_dict'])

            config["train"]["print_freq"] = 2

            config["data"]["donot_convert_data"] = True
            train_loader, val_loader, aggregated_data = data_load(config)

            return model, train_loader, val_loader, aggregated_data

        def model_load(config):
            model = MLPNetOWT_BN(config)
            model.type(config["model"]["type_model"])

            # define loss function (criterion) and optimizer
            if (config["train"]["loss"] in "BCELoss"):
                criterion = nn.BCELoss()
            if (config["train"]["loss"] in "CrossEntropyLoss"):
                criterion = nn.CrossEntropyLoss()

            criterion.type(config["model"]["type_model"])

            return model, criterion

        model, train_loader, val_loader, aggregated = load(args, config)

        exit()
        '''

        #samples = set(map(lambda l: tuple(l.tolist()), samples))
        #samples = sorted(samples)
        #samples = list(map(lambda l: torch.LongTensor(l), samples))
        #sample = torch.LongTensor([0, 0, 0, 0, 0, 0, 0])
        sample = []
        #f =  open(filename, "wb")
        #pickle.dump(data, f)
        #    f.close()
        # bnn: tensor([0, 0, 0, 0, 0, 0, 0])
        def execute(config, model, sample, aggregated_data):
            model.eval()
            samples = [sample.clone().cpu().detach().numpy()]
            input_encoder = aggregated_data["encoder"]
            bin_sample = []
            bin_sample.append(input_encoder.transform([samples[0]])[0])
            bin_sample.append(input_encoder.transform([samples[0]])[0])
            bin_sample = np.asarray(bin_sample)
            cuda_sample = torch.tensor(bin_sample).type(config["model"]["type_model"])
            result = model(cuda_sample)
            _, pred = result.float().topk(1, 1, True, True)
            winner = pred.cpu().detach().numpy()[0][0]
            print('result:', result)
            print('pred:', pred)
            print('winner:', winner)

        with open('q5_zoo.pkl', 'rb') as f:
            model = pickle.load(f)
        print(model.eval())
        exit()
        # emulation of classification for the given sample
        sinfo = execute(config, model, sample, aggregated)
    exit()
    # test:
    # dl:
    # bt:
    # bnn: tensor([0, 0, 0, 0, 0, 0, 0])

    # all features used
    proj_name = sys.argv[1]
    global_model_name = sys.argv[2]
    appr = sys.argv[3]
    nof_inst = int(sys.argv[4])
    #batch = int(sys.argv[-1])
    #print('batch:', batch)
    #print('Computing explanations using', appr)

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    k2name = {'X_train': './dataset/{}_X_train.csv'.format(proj_name),
              'y_train': './dataset/{}_y_train.csv'.format(proj_name),
              'X_test': './dataset/{}_X_test.csv'.format(proj_name),
              'y_test': './dataset/{}_y_test.csv'.format(proj_name)}

    path_X_train = './dataset/{}_X_train.csv'.format(proj_name)
    path_y_train = './dataset/{}_y_train.csv'.format(proj_name)
    X_train = pd.read_csv(path_X_train)
    y_train = pd.read_csv(path_y_train).iloc[:, 0]
    indep = X_train.columns
    dep = 'defect'

    path_X_explain = './dataset/{}_X_test.csv'.format(proj_name)
    path_y_explain = './dataset/{}_y_test.csv'.format(proj_name)
    X_explain = pd.read_csv(path_X_explain)
    y_explain = pd.read_csv(path_y_explain).iloc[:, 0]

    if global_model_name == 'RF':
        path_model = './global_model/{}_RF_30estimators_global_model.pkl'.format(proj_name)
    else:
        path_model = './global_model/{}_LR_global_model.pkl'.format(proj_name)

    with open(path_model, 'rb') as f:
        model = pickle.load(f)
        
    explainer = HExplainer(global_model_name, appr, X_train, y_train, model)

    """
    
    Explaining
    
    """

    selected_ids = set(range(len(X_explain)))

    if len(X_explain) > nof_inst:
        random.seed(1000)
        selected_ids = set(random.sample(range(len(X_explain)), nof_inst))

    #selected_ids = set(filter(lambda l: l % 90 == batch, range(len(X_explain))))
    #random.seed()

    times = []
    nof_inst = 0

    preds = model.predict(X_explain)

    for i in range(len(X_explain)):

        if i not in selected_ids:
            continue

        nof_inst += 1

        if i < len(X_explain) - 1:
            X = X_explain.iloc[i: i+1,]
            y = y_explain.iloc[i: i+1,]
        else:
            X = X_explain.iloc[i: , ]
            y = y_explain.iloc[i: , ]

        explainer.explain(X, y)

        times.append(explainer.time)

    #print(f'times: {times}\n')
    print()
    print('# of insts:', nof_inst)
    print(f'tot time: {sum(times)}')

    exit()
