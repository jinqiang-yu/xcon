#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## xdl.py
##

#
#==============================================================================
from data import Data
from dlist import DecisionList, Literal, Rule
from enc import DLEncoding
from exp import DLExplainer
from options import Options
import statistics
import sys
import os
import resource
import random
import pickle
import collections
import csv
import json

#
#==============================================================================
if __name__ == '__main__':
    options = Options(sys.argv)

    # first, reading a dataset
    if options.dataset:
        with open(options.dataset, 'r') as f:
            samples = f.readlines()
        header = samples[0]
        nof_cols = len(header.split(options.separator))

        samples = set(map(lambda l: ','.join(l.split(',')[:nof_cols-1]), samples[1:]))
        samples = sorted(samples)

        if options.nof_inst is not None and len(samples) > options.nof_inst:
            random.seed(1000)
            samples = random.sample(samples, options.nof_inst)

        with open(options.dataset + '.pkl', 'rb') as f:
            dtinfo = pickle.load(f)

        lines = []
        lines.append(header)
        for i, sample in enumerate(samples):
            s = [int(feat) for feat in sample.split(',')]
            line = [str(dtinfo['categorical_names'][fid][s[fid]]) for fid in range(len(s))]
            line = ','.join(line)
            lines.append(line)

        data = Data(lines=lines, separator=options.separator, ranges=options.ranges)

    # then reading a DL model
    if options.model:
        model = DecisionList(from_file=options.model, data=data)
    else:
        model = DecisionList(from_fp=sys.stdin, data=data)

    if options.verb > 1:
        print('MODEL:')
        print(model)
        print('\nENCODINDS:')

    if options.knowledge is not None:
        model.parse_bg(bgfile=options.knowledge)

    # creating the encodings
    encoder = DLEncoding(model, options)

    if options.verb:
        print('# of classes:', len(encoder.encs))
        print('min # of vars:', min([enc.nv for enc in encoder.encs.values()]))
        print('avg # of vars: {0:.2f}'.format(statistics.mean([enc.nv for enc in encoder.encs.values()])))
        print('max # of vars:', max([enc.nv for enc in encoder.encs.values()]))
        print('min # of clauses:', min([len(enc.hard) for enc in encoder.encs.values()]))
        print('avg # of clauses: {0:.2f}'.format(statistics.mean([len(enc.hard) for enc in encoder.encs.values()])))
        print('max # of clauses:', max([len(enc.hard) for enc in encoder.encs.values()]))
        print('\nEXPLANATIONS:')


    if options.check is None:
        if options.inst:
            # creating the explainer object
            explainer = DLExplainer(model, encoder, options)

            explainer.explain(options.inst, smallest=options.smallest,
                    xtype=options.xtype, xnum=options.xnum, unit_mcs=options.unit_mcs,
                    use_cld=options.use_cld, use_mhs=options.use_mhs,
                    reduce_=options.reduce)
        else:
            # no instance is provided, hence
            # explaining all instances of the dataset
            # here are some stats
            nofex, minex, maxex, avgex, times = [], [], [], [], []
            filtertimes = []
            use_times = []
            insts = [inst for inst in data]

            if options.batch is not None:
                b1, b2 = options.batch.split(',')
                b1, b2 = int(b1), int(b2)
                selected_ids = list(filter(lambda l: l % b2 == b1, range(len(insts))))
                insts = [insts[i] for i in selected_ids]

            for inst in insts:
                # creating the explainer object
                explainer = DLExplainer(model, encoder, options)

                expls = explainer.explain(inst, smallest=options.smallest,
                        xtype=options.xtype, xnum=options.xnum,
                        unit_mcs=options.unit_mcs, use_cld=options.use_cld,
                        use_mhs=options.use_mhs, reduce_=options.reduce)

                nofex.append(len(expls))
                minex.append(min([len(e) for e in expls]))
                maxex.append(max([len(e) for e in expls]))
                avgex.append(statistics.mean([len(e) for e in expls]))
                times.append(explainer.time)
                filtertimes.append(explainer.filter_time)
                use_times.append(explainer.use_time)

            exptimes = [times[i] - filtertimes[i] for i in range(len(times))]


            if options.verb:
                print(f'exptimes: {exptimes}')

                print('# of insts:', len(nofex))
                print('tot # of expls:', sum(nofex))
                print('min # of expls:', min(nofex))
                print('avg # of expls: {0:.2f}'.format(statistics.mean(nofex)))
                print('max # of expls:', max(nofex))
                print('')
                print('Min expl sz:', min(minex))
                print('min expl sz: {0:.2f}'.format(statistics.mean(minex)))
                print('avg expl sz: {0:.2f}'.format(statistics.mean(avgex)))
                print('max expl sz: {0:.2f}'.format(statistics.mean(maxex)))
                print('Max expl sz:', max(maxex))
                print('')
                print('tot exp time: {0:.2f}'.format(sum(exptimes)))
                print('min exp time: {0:.2f}'.format(min(exptimes)))
                print('avg exp time: {0:.2f}'.format(statistics.mean(exptimes)))
                print('max exp time: {0:.2f}'.format(max(exptimes)))
                if options.xtype in ('abductive', 'abd') and options.knowledge is not None \
                    and options.approach == 'use':
                    print('tot used rules time: {0:.2f}'.format(sum(use_times)))
    else:
        #validating explanations
        #' tuple(map(lambda f:  '='.join(list(self.fvmap.opp[f])), samp))'
        # mapping between real feature values proxy
        FVMap = collections.namedtuple('FVMap', ['dir', 'opp'])
        fvmap = FVMap(dir={}, opp={})

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

        with open(options.check, 'r') as f:
            expls_info = json.load(f)
        feature_names = list(dtinfo['feature_names'])

        # preparing sample and its explanation
        insts = []
        expls = []
        for sample in expls_info['stats']:
            inst = []
            fvs = sample.rsplit(' THEN ')[0].split('IF ', maxsplit=1)[-1].split(' AND ')

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
                inst.append('{0}={1}'.format(f, v))
            insts.append(tuple(inst))

            expl = expls_info['stats'][sample]['expl']
            expl = sorted(map(lambda l: feature_names.index(l), expl))
            expls.append(expl)

        # validating
        header = ['expl', 'valid?', 'minimal', 'mexpl']
        rows = []
        results = []
        for inst_id, inst in enumerate(insts):
            # creating the explainer object
            explainer = DLExplainer(model, encoder, options)
            expl = expls[inst_id]

            res, minimal, expl_ = explainer.validate(inst, expl)

            results.append(res)

            row = []
            preamble = [str(explainer.vars.obj(explainer.hypos[i])) for i in expl]
            row.append("IF {0} THEN {1}".format(' AND '.join(preamble), explainer.label))
            row.append(res)
            row.append(minimal)
            if expl_ is None:
                row.append(expl_)
            else:
                row.append(expl_)
            rows.append(row)

        qdtname = options.check.rsplit('/')[-1].rsplit('.json')[0] + '_test1'

        if '/lime/' in options.check:
            appr = 'lime'
        elif '/shap/' in options.check:
            appr = 'shap'
        elif '/anchor/' in options.check:
            appr = 'anchor'
        else:
            exit(1)

        if '/dl/' in options.check:
            model = 'dl'
        elif '/bt/' in options.check:
            model = 'bt'
        elif '/bnn/' in options.check:
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

