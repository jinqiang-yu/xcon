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
from data import Data

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

class DLmodel(object):
    def __init__(self, args, fvmap, X_train, nof_classes):
        self.X_train = X_train
        self.init_model(args, fvmap)
        self.nof_classes = nof_classes

    def init_model(self, args, fvmap):
        columns = list(self.X_train.columns)
        with open(args.load, 'r') as f:
            lines = f.readlines()
        lines = list(filter(lambda l: 'cover: ' in l and ' => ' in l, lines))
        lines = list(map(lambda l: l.split('cover: ')[-1].strip().split(' => '), lines))
        conditions = []
        predictions = []
        for line in lines[:-1]:
            # condition
            cond = line[0].split("', ")
            cond = list(map(lambda l: l.strip("'").strip(), cond))
            cond = list(map(lambda l: tuple(l.split(': ')), cond))
            for i, c in enumerate(cond):
                if "not '" in c[0]:
                    f = c[0].split("not '", maxsplit=1)[-1].strip()
                    sign = False
                else:
                    f = c[0]
                    sign = True
                try:
                    v = int(c[1])
                except:
                    try:
                        v = float(c[1])
                    except:
                        v = c[1].strip()

                fid = columns.index(f)
                cond[i] = {'fid': fid, 'feat': f, 'value': fvmap.dir[tuple([f, v])][-1], 'sign': sign}
            conditions.append(cond)

            # prediction
            label, v = line[-1].split(':', maxsplit=1)
            try:
                v = int(v)
            except:
                try:
                    v = float(v)
                except:
                    v = v.strip()
            pred = fvmap.dir[tuple([label, v])]
            predictions.append(pred)

        # default rule
        conditions.append([])
        label, v = lines[-1][-1].split(':', maxsplit=1)
        try:
            v = int(v)
        except:
            try:
                v = float(v)
            except:
                v = v.strip()
        predictions.append(fvmap.dir[tuple([label, v])])

        self.conditions = conditions
        self.predictions = predictions

    def match(self, fid2v, cond):
        for fv in cond:
            eq = fid2v[fv['fid']] == fv['value']
            if not (eq == fv['sign']):
                return False
        else:
            return True

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

        results = []
        for s in sampless:
            fid2v = {fid: s[fid] for fid in range(len(s))}
            result = [0.0 for i in range(self.nof_classes)]
            for i, cond in enumerate(self.conditions):
                if self.match(fid2v, cond):
                    pred = self.predictions[i][-1]
                    result[pred] = 1.0
                    break

            results.append(np.asarray(result))
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

    model = DLmodel(args, fvmap, X_train, len(dinfo['class_names']))
    explainer = HExplainer(args.approach, X_train, y_train, model, dinfo, fvmap)

    """

    Explaining

    """
    # prepare test data

    # reading all unique samples
    with open(args.test, 'r') as f:
        lines = f.readlines()[1:]
    lines = set(map(lambda l: l[: l.rfind(',')], lines))
    lines = sorted(lines)

    if args.inst is not None and len(lines) > args.inst:
        random.seed(1000)
        lines = random.sample(lines, args.inst)
    samples = list(map(lambda l: list(map(lambda ll: float(ll), l.split(','))), lines))

    if args.batch is not None:
        b1, b2 = args.batch.split(',')
        b1, b2 = int(b1), int(b2)
        selected_ids = list(filter(lambda l: l % b2 == b1, range(len(samples))))

        samples = [samples[i] for i in selected_ids]

    X_explain = pd.DataFrame(samples, columns=X_train.columns)

    times = []
    nof_inst = 0

    for i in range(len(X_explain)):

        nof_inst += 1

        if i < len(X_explain) - 1:
            X = X_explain.iloc[i: i + 1, ]
        else:
            X = X_explain.iloc[i:, ]
        explainer.model.predict_proba(X)
        explainer.explain(X)

        times.append(explainer.time)

    # print(f'times: {times}\n')
    print()
    print('# of insts:', nof_inst)
    print(f'tot time: {sum(times)}')


