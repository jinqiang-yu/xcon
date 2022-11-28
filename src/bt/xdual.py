#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## xdual.py
##

from __future__ import print_function
from data import Data
from options import Options
import os
import sys
from xgbooster import XGBooster, preprocess_dataset
from enumerate import enumerate_all
import collections
import pickle
import json
import csv

def show_info():
    """
        Print info message.
    """

    print('c XDual: dual explanations for XGBoost models')
    print('')

if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    # showing head
    show_info()

    if options.explain == 'all':
        enumerate_all(options, useumcs=False, prefix='mabd')
    else:

        if (options.preprocess_categorical):
            preprocess_dataset(options.files[0], options.preprocess_categorical_files)
            exit()

        if options.files:
            xgb = None

            if options.train:
                data = Data(filename=options.files[0], mapfile=options.mapfile,
                        separator=options.separator,
                        use_categorical = options.use_categorical)

                xgb = XGBooster(options, from_data=data)
                train_accuracy, test_accuracy, model = xgb.train()

            # read a sample from options.explain
            if options.explain:
                options.explain = [float(v.strip()) for v in options.explain.split(',')]

            if options.encode:
                if not xgb:
                    xgb = XGBooster(options, from_model=options.files[0])

                # encode it and save the encoding to another file
                xgb.encode(test_on=options.explain)

            if options.explain:
                if not xgb:
                    # abduction-based approach requires an encoding
                    xgb = XGBooster(options, from_encoding=options.files[0])

                # explain using anchor or the abduction-based approach
                expl = xgb.explain(options.explain)

                # here we take only first explanation if case enumeration was done
                if options.xnum != 1:
                    expl = expl[0]

            if options.approach == 'check':
                # mapping between real feature values proxy
                FVMap = collections.namedtuple('FVMap', ['dir', 'opp'])
                fvmap = FVMap(dir={}, opp={})

                with open(options.dtinfo, 'rb') as f:
                    dtinfo = pickle.load(f)

                for fid in dtinfo['categorical_names']:
                    feature = dtinfo['feature_names'][fid]
                    for i in range(len(dtinfo['categorical_names'][fid])):
                        real_value = dtinfo['categorical_names'][fid][i]
                        fvmap.dir[tuple([feature, real_value])] = tuple([feature, i])
                        fvmap.opp[tuple([feature, i])] = tuple([feature, real_value])
                for lid in range(len(dtinfo['class_names'])):
                    real_value = dtinfo['class_names'][lid]
                    label = dtinfo['feature_names'][-1]
                    fvmap.dir[tuple([label, real_value])] = tuple([label, lid])
                    fvmap.opp[tuple([label, lid])] = tuple([label, real_value])

                files = [options.files[1], options.files[0]]
                options.files = files

                with open(options.files[0], 'r') as f:
                    expls_info = json.load(f)
                feature_names = list(dtinfo['feature_names'])

                # preparing sample and its explanation
                samples = []
                expls = []
                for inst in expls_info['stats']:
                    sample = []
                    fvs = inst.rsplit(' THEN ')[0].split('IF ', maxsplit=1)[-1].split(' AND ')

                    for fv in fvs:
                        f, v = fv.split(' = ', maxsplit=1)
                        f = f.strip()

                        try:
                            v = int(v)
                        except:
                            try:
                                v = float(v)
                            except:
                                v = v.strip()

                        sample.append(fvmap.dir[tuple([f, v])][-1])
                    samples.append(sample)

                    expl = expls_info['stats'][inst]['expl']
                    expl = sorted(map(lambda l: feature_names.index(l), expl))
                    expls.append(expl)

                # validating
                header = ['expl', 'valid?', 'minimal', 'mexpl']
                rows = []
                results = []

                for i, s in enumerate(samples):
                        # enumerate explanations only for the first 10% of samples
                        # if i % (len(lines) / 10) == 0:
                        xgb = XGBooster(options, from_model=options.files[1])

                        # encode it and save the encoding to another file
                        xgb.encode()

                        options.explain = list(map(lambda l: float(l), s))
                        expl = expls[i]

                        res, minimal, expl_ = xgb.validate(options.explain, expl)

                        row = []
                        row.append('IF {0} THEN {1}'.format(' AND '.join([xgb.x.preamble[f] for f in expl]),
                                                            xgb.target_name[xgb.x.out_id]))
                        row.append(res)
                        row.append(minimal)
                        if expl_ is None:
                            row.append(expl_)
                        else:
                            row.append('IF {0} THEN {1}'.format(' AND '.join([xgb.x.preamble[f] for f in expl_]),
                                                                xgb.target_name[xgb.x.out_id]))
                        rows.append(row)

                qdtname = files[0].rsplit('/')[-1].rsplit('.json')[0] + '_test1'

                if '/lime/' in files[0]:
                    appr = 'lime'
                elif '/shap/' in files[0]:
                    appr = 'shap'
                elif '/anchor/' in files[0]:
                    appr = 'anchor'
                else:
                    exit(1)

                if '/dl/' in files[0]:
                    model = 'dl'
                elif '/bt/' in files[0]:
                    model = 'bt'
                elif '/bnn/' in files[0]:
                    model = 'bnn'
                else:
                    exit(1)

                saved_dir = '../stats/correctness/{0}/{1}'.format(appr, 'size5' if options.knowledge else 'ori')

                if not os.path.isdir((saved_dir)):
                    os.makedirs(saved_dir)

                name = '{0}_{1}.csv'.format(model, qdtname)
                with open("{0}/{1}".format(saved_dir, name), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows)





