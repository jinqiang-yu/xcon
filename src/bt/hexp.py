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
import math
import collections
from xgbooster import XGBooster
from xgboost import XGBClassifier, Booster
from options import Options

#
#==============================================================================

class HExplainer(object):
    #HeuristicExplainer
    def __init__(self, appr, X_train, model, dinfo, fvmap):
        self.appr = appr
        self.X_train = X_train
        self.model = model
        self.dinfo = dinfo
        self.fvmap = fvmap
        self.init_explainer(appr)

    def init_explainer(self, appr):
        if appr.lower() == 'lime':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,
                                                                    feature_names=self.model.xgb.feature_names,
                                                                    categorical_features=self.model.xgb.categorical_features,
                                                                    class_names=self.model.xgb.target_name,
                                                                    discretize_continuous=True)
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

            #self.explainer = shap.KernelExplainer(self.model.predict, shap.sample(self.X_train, 50))
            self.explainer = shap.TreeExplainer(self.model.xgb.model)

        elif appr.lower() == 'anchor':
            categorical_names = {}
            for f in self.dinfo['categorical_names']:
                categorical_names[f] = [str(v) for v in self.dinfo['categorical_names'][f]]

            self.explainer = anchor_tabular.AnchorTabularExplainer(
                class_names=list(range(len(self.dinfo['class_names']))),
                feature_names=self.model.xgb.feature_names,
                train_data=self.X_train,
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
                                          num_features=len(X.columns),
                                          top_labels=1)
        columns = list(X.columns)
        expl = expl.as_list(label=int(pred))
        expl = list(map(lambda l: tuple([columns.index(l[0].split('=', maxsplit=1)[0].strip()), l[1]]), expl))
        expl = sorted(expl, key=lambda l: int(l[0]))
        self.expl = list(filter(lambda l: l[1] >= 0, expl))
        self.expl = list(map(lambda l: int(l[0]), self.expl))

        expl = list(map(lambda l: tuple([X.columns[int(l[0])], l[1]]), expl))
        print('  All classes:', self.dinfo['class_names'].tolist())
        print('  Features in explanation:', expl)


    def shap_explain(self, X, pred):
        feat_sample_exp = np.expand_dims(X.values[0], axis=0)
        feat_sample_exp = self.model.xgb.transform(feat_sample_exp)
        shap_values = self.explainer.shap_values(feat_sample_exp)
        #shap_values = self.explainer.shap_values(X)

        try:
            shap_values = shap_values[pred]
        except:
            shap_values = shap_values[-1]

        # we need to sum values per feature
        sum_values = []
        p = 0
        for f in self.model.xgb.categorical_features:
            nb_values = len(self.model.xgb.categorical_names[f])
            sum_v = 0
            for i in range(nb_values):
                if len(shap_values) == 1:
                    sum_v = sum_v + shap_values[-1][p + i]
                else:
                    sum_v = sum_v + shap_values[p + i]
            p = p + nb_values
            sum_values.append(sum_v)

        expl = []
        self.expl = []
        for i, v in enumerate(sum_values):
            expl.append(tuple([X.columns[i], v]))
            if v > 0:
                self.expl.append(i)

        print('  All classes:', self.dinfo['class_names'].tolist())
        print('  Features in explanation:', expl)

    def anchor_explain(self, X, pred):
        exp = self.explainer.explain_instance(X.values[0], self.model.predict, threshold=0.95)
        print('  Anchor: %s' % (' AND '.join(exp.names())))
        print('  Precision: %.2f' % exp.precision())
        print('  Coverage: %.2f' % exp.coverage())
        self.expl = sorted(exp.features())

class BTmodel(object):
    def __init__(self, args):
        self.xgb = XGBooster(options, from_model=options.files[1])

    def pickle_load_file(self, filename):
        f =  open(filename, "rb")
        data = pickle.load(f)
        f.close()
        return data

    def predict(self, sample):
        results = self.predict_proba(sample)
        winners = []
        for result in results:
            winner = np.argmax(result)
            winners.append(winner)

        winners = np.asarray(winners)
        return winners

    def predict_proba(self, sample):

        try:
            sampless = sample.to_numpy()
        except:
            sampless = sample
        samples = self.xgb.transform(sampless)

        samples = np.asarray(samples)
        results = self.xgb.model.predict_proba(samples)

        return results

if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)

    # python ./bt/hexp.py -I 100 --load ./bt/btmodels/q6/compas_train1_data/compas_train1_data_nbestim_25_maxdepth_3_testsplit_0.2.mod.pkl -t ../bench/cv/train/quantise/q6/other/fairml/compas/compas_train1_data.csv -T ../bench/cv/test/quantise/q6/other/fairml/compas/compas_test1_data.csv -a lime -m bnn
    # python ./bt/hexp.py -I 100 --load ./bt/btmodels/q6/zoo_train1_data/zoo_train1_data_nbestim_25_maxdepth_3_testsplit_0.2.mod.pkl -t ../bench/cv/train/quantise/q6/other/zoo/zoo_train1_data.csv -T ../bench/cv/test/quantise/q6/other/zoo/zoo_test1_data.csv -a shap -m bnn
    #X_train = pd.read_csv(args.train).iloc[:, :-1]
    #y_train = pd.read_csv(args.train).iloc[:, -1]

    #'python ./bt/hexp.py -I 100 -A lime ../bench/cv/test/quantise/q6/other/fairml/compas/compas_test1_data.csv ./bt/btmodels/q6/compas_train1_data/compas_train1_data_nbestim_25_maxdepth_3_testsplit_0.2.mod.pkl'
    #test=files[0]
    #model=files[1]
    FVMap = collections.namedtuple('FVMap', ['dir', 'opp'])
    fvmap = FVMap(dir={}, opp={})

    with open(options.files[0] + '.pkl', 'rb') as f:
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

    model = BTmodel(options)
    X_train = model.xgb.X_train

    explainer = HExplainer(options.approach, X_train, model, dinfo, fvmap)

    """

    Explaining

    """

    # reading all unique samples
    with open(options.files[0], 'r') as f:
        lines = f.readlines()[1:]
    lines = set(map(lambda l: l[: l.rfind(',')], lines))
    lines = sorted(lines)

    if options.nof_inst is not None and len(lines) > options.nof_inst:
        random.seed(1000)
        lines = random.sample(lines, options.nof_inst)

    samples = list(map(lambda l: list(map(lambda ll: float(ll), l.split(','))), lines))
    X_explain = pd.DataFrame(samples, columns=model.xgb.feature_names)

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


