#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## rextract.py
##
##

#
#==============================================================================
import collections
from ruler import Ruler
import os
import resource
import socket
import six
from check import ConsistencyChecker
from data import Data
import sys
from options import Options
import json
from apyori import apriori
import pandas as pd
from fim import eclat

#
#==============================================================================
class Rextract(object):
    """
        Individual rule-based decision set miner.
    """

    def __init__(self, data, options):
        """
            Constructor.
        """
        self.filename = options.dataset
        # saving data
        self.data_ = data
        # saving options
        self.options = options

    def next_feature(self, rextract):
        """
            rextract: float. The index of the target feature
            select the next feature as the target
        """
        self.data = Data(filename=self.filename, mapfile=self.options.mapfile,
                         separator=self.options.separator, ranges=self.options.ranges, rextract=rextract)

        if self.options.noccheck == False:
            # phase0: consistency check
            checker = ConsistencyChecker(self.data, self.options)
            if checker.status and checker.do() == False:
                checker.remove_inconsistent()
                if self.options.verb:
                    print('c0 data set is inconsistent')

                if self.options.cdump:
                    checker.dump_consistent()

            if checker.status == False:
                print('c0 not enough classes => classification makes no sense')
                return False

        # samples clustered by their label
        self.clusters = collections.defaultdict(lambda: [])
        # binarizing the data properly
        for i in range(len(self.data.samps)):
            samp_bin, out = self.data.samps[i][:-1], self.data.samps[i][-1]
            for l in samp_bin:
                if l > 0:  # negative literal means that the feature is binary
                    name, lit = self.data.fvmap.opp[l]
                    j = self.data.nm2id[name]

                    if len(self.data.feats[j]) > 2:
                        samp_bin += [-self.data.fvmap.dir[(name, l)] for l in
                                     sorted(self.data.feats[j].difference(set([lit])))]

            self.data.samps[i] = samp_bin + [out]
            self.clusters[out].append(i)

        # depending on this option, we compute either one class or all of them
        if self.options.to_compute == 'all':
            self.labels = [self.data.fvmap.dir[self.data.names[-1], c] for c in sorted(self.data.feats[-1])]
        elif self.options.to_compute == 'best':
            raise NotImplementedError('Best class computation is not supported')
        else:
            to_compute = self.options.to_compute.split(',')
            self.labels = [self.data.fvmap.dir[self.data.names[-1], c] for c in to_compute]

        return True

    def compute(self):
        """
            Rule mining
            Finding the correlations between features
        """
        if self.options.approach in ('apriori', 'eclat'):
            self.compute_itemsets()
        elif self.options.approach in ('sat', 'maxsat'):
            self.compute_maxsat()
        else:
            self.rules = {}

        return self.rules

    def compute_itemsets(self):
        """
            Rule mining
            Finding the correlations between features using Apriori or Eclat
        """
        self.rules = collections.defaultdict(lambda : [])

        samples = list(map(lambda l: l[:-1], self.data_.samps))

        if self.options.approach == 'apriori':
            if self.options.cutp > 0:
                rules = apriori(samples, min_support=1 / len(samples), max_length=self.options.cutp+1, zmin=2, min_confidence=1.0)
            else:
                rules = apriori(samples, min_support=1 / len(samples), zmin=2, min_confidence=1.0)

            for r in rules:
                for i, b in enumerate(r[2]):
                    feats = set(b[0])
                    labels_list = set(b[1])
                    for label in labels_list:
                        self.rules[label].append(feats)
        else:
            if self.options.cutp > 0:
                rules = eclat(samples, target='r', supp=0, zmax=int(self.options.cutp)+1, zmin=2,
                              conf=100, report='e', eval='c')
            else:
                rules = eclat(samples, target='r', supp=0, zmin=2,
                              conf=100, report='e', eval='c')

            for r in rules:
                label = r[0]
                feats = set(r[1])
                self.rules[label].append(feats)

        rules = collections.defaultdict(lambda: collections.defaultdict(lambda: []))

        for label_lit in self.rules:
            conditions = []
            label, lvalue = self.data_.fvmap.opp[label_lit]
            feat_lits_list = sorted(self.rules[label_lit], key=lambda l: len(l))
            for feat_lits in feat_lits_list:
                for cond in conditions:
                    if cond.issubset(feat_lits):
                        break
                else:
                    conditions.append(feat_lits)

                    conditon = []
                    for f in feat_lits:
                        feature, fvalue = self.data_.fvmap.opp[abs(f)]
                        sign = True if f > 0 else False
                        conditon.append({'feature': feature,
                                         'value': fvalue,
                                         'sign': sign})
                    rules[label][lvalue].append(conditon)

        self.rules = rules

    def compute_maxsat(self):
        """
            Rule mining
            Finding the correlations between features
        """

        # dictionary storing the rules for targetted features with all the classes
        self.rules = []
        self.rule_stats = {}

        if self.options.plimit:
            self.tg2rules = collections.defaultdict(lambda : [])

        for f in range(1, len(self.data_.names)):
            if self.options.verb:
                print('\nc0 computing feature {0}'.format(f))

            # target the next feature
            if self.next_feature(rextract=f) == False:
                continue
            nrules, cost = 0, 0

            # going over the labels to compute
            for label in self.labels:

                if self.options.verb:
                    print('c1 computing class:', self.data.fvmap.opp[label][1])

                # stage 1
                # rule mining
                ruler = Ruler(self.clusters, label, self.data, self.options)

                # block duplicate rules
                target = self.data.fvmap.opp[label]
                prules = [self.rules[i] for i in self.tg2rules[target]]

                if self.options.blk:
                    self.process_prules(prules, label, ruler)

                #compute
                rules = ruler.enumerate()

                if self.options.blk:
                    # record
                    for i, rule in enumerate(rules):
                        targets = []
                        for flit in rule.flits:
                            fnm = self.data.fvmap.opp[abs(flit)][0]
                            values = self.data.feats[self.data.nm2id[fnm]]
                            if len(values) <= 2:
                                targets.append(self.data.fvmap.opp[-flit])
                        targets.append(self.data.fvmap.opp[rule.label])

                        for t in targets:
                            self.tg2rules[t].append(len(self.rules) + i)

                self.rule_stats = {**self.rule_stats, **ruler.rule_stat}

                self.rules.extend([rule for rule in rules])
                nrules += len(rules)
                if self.options.verb:
                    if self.options.verb > 1:
                        for rule in rules:
                            print('c1 rule:', str(rule))

                    print('c1 # of rules: {0}'.format(len(rules)))
                    print('c1 rule time: {0:.4f}'.format(ruler.time))
                    print('')

            if self.options.verb:
                print('c2 total rules: {0}'.format(nrules))

                if self.options.weighted:
                    print('c2 total wght: {0}'.format(cost))

        self.data = self.data_

        rules = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
        for rule in self.rules:
            feats = rule.flits[:]
            label, lvalue = rule.fvmap.opp[rule.label]
            conditon = []
            for f in feats:
                feature, fvalue = rule.fvmap.opp[abs(f)]
                sign = True if f > 0 else False
                conditon.append({'feature': feature,
                                 'value': fvalue,
                                 'sign': sign})
            rules[label][lvalue].append(conditon)

        self.rules = rules

    def process_prules(self, prules, label, ruler):
        """
            Block the rules previously generated
        """
        fired_tvars = set()
        for pr in prules:
            pflits = [-self.data.fvmap.dir[pr.fvmap.opp[pr.label]]]
            for l in pr.flits:
                id = self.data.fvmap.dir[pr.fvmap.opp[abs(l)]]
                if abs(id) < len(self.data.samps[0]):
                    pflits.append(id if l > 0 else -id)

            # tvars fired by the current rule
            pr_tvars = set()
            match = False
            for i in ruler.clusters[label]:
                samp = set(self.data.samps[i])
                matlits = list(filter(lambda l: l in samp, pflits))
                if len(matlits) == len(pflits):
                    tvar = ruler.s2tvar[i]
                    # update the number of rules which fire each instance
                    ruler.nof_p[tvar] += 1

                    fired_tvars.add(tvar)
                    pr_tvars.add(tvar)

                    match = True

            # block the rule
            if match:
                ruler.rc2.add_clause([-ruler.vdrmap[l] for l in pflits])

                if self.options.bsymm:
                    # breaking symmetric solutions
                    symmpr = sorted(set(ruler.tvars).difference(pr_tvars))
                    ruler.rc2.add_clause(symmpr)

                if self.options.plimit:
                    i, reduced = 0, False

                    while i < len(ruler.tvars):
                        t = ruler.tvars[i]

                        if ruler.nof_p[t] < self.options.plimit:
                            i += 1
                        else:
                            ruler.tvars[i] = ruler.tvars[-1]
                            ruler.tvars.pop()
                            reduced = True

                    if reduced:
                        ruler.rc2.add_clause(ruler.tvars)

if __name__ == '__main__':
    options = Options(sys.argv)

    # parsing data
    if options.dataset:
        data = Data(filename=options.dataset, mapfile=options.mapfile,
                    separator=options.separator, ranges=options.ranges)
    else:
        data = Data(fpointer=sys.stdin, mapfile=options.mapfile,
                    separator=options.separator)

    if options.verb:
        print('c0 # of samps: {0} ({1} weighted)'.format(sum(data.wghts), len(data.samps)))
        print('c0 # of feats: {0} ({1} binary)'.format(len(data.names) - 1, len(
            list(filter(lambda x: x > 0, data.fvmap.opp.keys()))) - len(data.feats[-1])))
        print('c0 # of labls: {0}'.format(len(data.feats[-1])))

        used_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        print('c0 parse time: {0:.4f}'.format(used_time))
        print('')

    if options.noccheck == False:
        # phase0: consistency check
        checker = ConsistencyChecker(data, options)
        if checker.status and checker.do() == False:
            checker.remove_inconsistent()
            if options.verb:
                print('c0 data set is inconsistent')
                print('c0 filtering out {0} samples ({1} left)'.format(data.samps_filt, len(data.samps)))
                print('c0 filtering out {0} weights ({1} left)'.format(data.wghts_filt, sum(data.wghts)))
                print('c0 check time: {0:.4f}'.format(checker.time))
                print('')

            if options.cdump:
                checker.dump_consistent()

        if checker.status == False:
            print('c0 not enough classes => classification makes no sense')
            sys.exit(1)

    rextract = Rextract(data, options)
    knowledge = rextract.compute()

    if options.verb:
        total_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(
            resource.RUSAGE_SELF).ru_utime
        print('c3 total time: {0:.4f}\n'.format(total_time))

    if options.save_to:
        print('Rules are saved to', options.save_to)
        saved_dir = options.save_to.rsplit('/', maxsplit=1)[0]
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)

        with open(options.save_to, 'w') as f:
            json.dump(knowledge, f, indent=4)

    exit()
