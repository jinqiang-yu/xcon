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
import math
from main_routine import main_load as load
import collections

#
#==============================================================================

class HExplainer(object):
    #HeuristicExplainer
    def __init__(self, appr, X_train, y_train, model, dinfo, fvmap):
        self.appr = appr
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.dinfo = dinfo
        self.fvmap = fvmap
        self.init_explainer(appr)

    def init_explainer(self, appr):
        if appr.lower() == 'lime':
            categorical_names = {}
            for f in self.dinfo['categorical_names']:
                categorical_names[f] = [str(v) for v in self.dinfo['categorical_names'][f]]

            self.explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,
                                                                    categorical_names=categorical_names,
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
            #self.explainer = shap.Explainer(self.model,
                                            # feature_names=self.X_train.columns,
                                            #self.X_train)

            self.explainer = shap.KernelExplainer(self.model.predict, shap.sample(self.X_train, 50))

        elif appr.lower() == 'anchor':
            categorical_names = {}
            for f in self.dinfo['categorical_names']:
                categorical_names[f] = [str(v) for v in self.dinfo['categorical_names'][f]]
            self.explainer = anchor_tabular.AnchorTabularExplainer(
                class_names=list(range(len(self.dinfo['class_names']))),
                feature_names=self.X_train.columns,
                train_data=self.X_train.values,
                categorical_names=categorical_names)
        else:
            print('Wrong approach input')
            exit(1)

    def explain(self, X):
        pred = self.model.predict(X)[0]

        inst = X.iloc[0]
        preamble = []

        for fid, f in enumerate(inst.index):
            preamble.append(f'{f} = {self.fvmap.opp[tuple([f, inst[fid]])][-1]}')

        label = self.dinfo['feature_names'][-1]

        print('explaining: IF {} THEN {} = {}'.format(' AND '.join(preamble),
                                                      label,
                                                      self.fvmap.opp[tuple([label, pred])][-1]))

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime

        if self.appr.lower() == 'lime':
            self.lime_explain(X, pred)
        elif self.appr.lower() == 'shap':
            self.shap_explain(X, pred)
        elif self.appr.lower() == 'anchor':
            self.anchor_explain(X, pred)

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        print(f'  time: {self.time}\n')

    def lime_explain(self, X, pred):
        expl = self.explainer.explain_instance(X.iloc[0, :],
                                          self.model.predict_proba, # prediction function ( predict probability)
                                          num_features=len(X.columns), #10 is the default value
                                          top_labels=1)

        expl = sorted(expl.as_list(label=int(pred)), key=lambda l: int(l[0]))
        self.expl = list(filter(lambda l: l[1] >= 0, expl))
        self.expl = list(map(lambda l: int(l[0]), self.expl))

        expl = list(map(lambda l: tuple([X.columns[int(l[0])], l[1]]), expl))
        print('  All classes:', self.dinfo['class_names'].tolist())
        print('  Features in explanation:', expl)


    def shap_explain(self, X, pred):
        shap_values = self.explainer.shap_values(X)
        try:
            shap_values = shap_values[pred]
        except:
            shap_values = shap_values[-1]
        expl = []
        self.expl = []
        for i, v in enumerate(shap_values):
            expl.append(tuple([X.columns[i], v]))
            if v > 0:
                self.expl.append(i)

        print('  All classes:', self.dinfo['class_names'].tolist())
        print('  Features in explanation:', expl)

        '''
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
        '''

    def anchor_explain(self, X, pred):
        exp = self.explainer.explain_instance(X.values[0], self.model.predict, threshold=0.95)
        print('  Anchor: %s' % (' AND '.join(exp.names())))
        print('  Precision: %.2f' % exp.precision())
        print('  Coverage: %.2f' % exp.coverage())
        self.expl = sorted(exp.features())

        '''
        # explanation
        expl = [name for f, name in sorted(zip(exp.features(), exp.names()))]
        print(expl)
        exit()
        preamble = ' AND '.join(expl)

        print('  expl: IF {0} THEN defect = {1}'.format(preamble, pred))
        print('  size:', len(expl))
        #print('  Anchor: %s' % (' AND '.join(exp.names())))
        #print('  Precision: %.2f' % exp.precision())
        #print('  Coverage: %.2f' % exp.coverage())

        '''

class BNNmodel(object):
    def __init__(self, args):
        self.config = json.load(open(os.path.join(args.load, 'config.json')))
        random.seed(self.config['manual_seed'])
        # loading the model
        self.model, self.train_loader, self.val_loader, self.aggregated = load(args, self.config)

    def predict(self, sample):
        results = self.predict_proba(sample)
        winners = []
        for result in results:
            winner = np.argmax(result)
            winners.append(winner)

        winners = np.asarray(winners)
        return winners

    def predict_proba(self, sample):
        self.model.eval()
        try:
            sampless = sample.to_numpy()
        except:
            sampless = sample

        sampless = list(map(lambda l: [l], sampless))
        results = []
        for samples in sampless:
            input_encoder = self.aggregated["encoder"]
            bin_sample = []
            bin_sample.append(input_encoder.transform([samples[0]])[0])
            bin_sample.append(input_encoder.transform([samples[0]])[0])
            bin_sample = np.asarray(bin_sample)
            cuda_sample = torch.tensor(bin_sample).type(self.config["model"]["type_model"])
            result = self.model(cuda_sample)
            results.append(result[0].detach().numpy())#.detach().numpy())
        results = np.asarray(results)
        return results


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

    parser = argparse.ArgumentParser(description='Heuristic explainer')
    parser.add_argument('-a', '--approach', type=str, default=None,
                        help='Whether extracting useful rules')
    parser.add_argument('-b', '--batch', type=str, default=None,
                        help='Batch')
    parser.add_argument('-I', '--inst', type=int, default=None,
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

    return ret


if __name__ == '__main__':

    args = parse_options()
    # python ./bnns/hexp.py -I 100 --load ./bnns/bnnmodels/large/quantise/q6/other/fairml/compas/compas_test1/ -t ../bench/cv/train/quantise/q6/other/fairml/compas/compas_train1_data.csv -T ../bench/cv/test/quantise/q6/other/fairml/compas/compas_test1_data.csv -a lime -m bnn
    # python ./bnns/hexp.py -I 100 --load ./bnns/bnnmodels/small/quantise/q6/other/zoo/zoo_test1/ -t ../bench/cv/train/quantise/q6/other/zoo/zoo_train1_data.csv -T ../bench/cv/test/quantise/q6/other/zoo/zoo_test1_data.csv -a shap -m bnn
    X_train = pd.read_csv(args.train).iloc[:, :-1]
    y_train = pd.read_csv(args.train).iloc[:, -1]

    FVMap = collections.namedtuple('FVMap', ['dir', 'opp'])
    fvmap = FVMap(dir={}, opp={})

    with open(args.train + '.pkl', 'rb') as f:
        dinfo = pickle.load(f)

    for fid in dinfo['categorical_names']:
        feature = dinfo['feature_names'][fid]
        for i in range(len(dinfo['categorical_names'][fid])):
            real_value = dinfo['categorical_names'][fid][i]
            fvmap.dir[tuple([feature, real_value])] = tuple([feature, i])
            fvmap.opp[tuple([feature, i])] = tuple([feature, real_value])
    for lid in range(len(dinfo['class_names'])):
        real_value = dinfo['class_names'][lid]
        label = dinfo['feature_names'][-1]
        fvmap.dir[tuple([label, real_value])] = tuple([label, lid])
        fvmap.opp[tuple([label, lid])] = tuple([label, real_value])

    model = BNNmodel(args)
    explainer = HExplainer(args.approach, X_train, y_train, model, dinfo, fvmap)

    """

    Explaining

    """
    samples = []
    for b, (inputs, target) in enumerate(model.val_loader):
        samples.extend(inputs)

    samples = set(map(lambda l: tuple(l.tolist()), samples))
    samples = sorted(samples)
    if args.inst is not None and len(samples) > args.inst:
        random.seed(1000)
        samples = random.sample(samples, args.inst)

    if args.batch is not None:
        b1, b2 = args.batch.split(',')
        b1, b2 = int(b1), int(b2)
        selected_ids = list(filter(lambda l: l % b2 == b1, range(len(samples))))

        samples = [samples[i] for i in selected_ids]

    X_explain = pd.DataFrame(samples, columns=model.aggregated['feature_names'][:-1])

    times = []
    nof_inst = 0

    for i in range(len(X_explain)):

        nof_inst += 1

        if i < len(X_explain) - 1:
            X = X_explain.iloc[i: i + 1, ]
        else:
            X = X_explain.iloc[i:, ]

        explainer.explain(X)

        times.append(explainer.time)

    # print(f'times: {times}\n')
    print()
    print('# of insts:', nof_inst)
    print(f'tot time: {sum(times)}')


