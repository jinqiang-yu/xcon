#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## explain.py

#
#==============================================================================
import argparse
from itertools import chain
import json
from main_routine import main_load as load
from main_routine import main_encode_sample as encode
from main_routine import main_execute_sample as execute
import os
from pysat.examples.hitman import Hitman
from pysat.formula import IDPool
from pysat.solvers import Solver
import random
from utils import *
import resource
import statistics
import sys
import pickle
import collections
import csv

#
#==============================================================================
class BNNExplainer():
    """
        Basic BNN explainer.
    """

    def __init__(self, formula, extra, solver='g3', xtype='abd', xnum=1,
            smallest=False, reduce='lin', verbosity=1, approach=None):
        """
            Constructor.
        """

        # verbosity level
        self.verb = verbosity

        self.formula = formula
        self.extra = extra  # extra info
        self.fvmap = extra['features2vars']
        self.vpool = IDPool(start_from=formula.nv + 1)
        self.xtype = xtype
        self.xmin  = smallest
        self.xnum  = xnum
        self.reduce = reduce
        self.approach = approach

        # creating a new oracle
        self.oracle = Solver(name=solver, bootstrap_with=formula)
        self.solver = solver

        # creating feature selectors
        self.sels, self.smap = [], {}
        for fid, feat in enumerate(sorted(self.fvmap.keys())):
            item = self.fvmap[feat]
            selv = self.vpool.id(tuple(item['vars']))

            self.sels.append(selv)
            self.smap[selv] = fid

        # sample ids
        self.sids = set([])

        # number of oracle calls
        self.calls = 0

    def __del__(self):
        """
            Destructor.
        """

        self.delete()

    def __enter__(self):
        """
            'with' constructor.
        """

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            'with' destructor.
        """

        self.delete()

    def delete(self):
        """
            Explicit destructor.
        """

        if self.oracle:
            self.oracle.delete()
            self.oracle = None

    def prepare(self, sample_info):
        """
            Process the sample and include additional clauses.
        """

        # input assignment
        self.input = tuple(sample_info['input2lits'])

        # sample id is either newly created or reused
        self.sidv = self.vpool.id(self.input)

        # getting human-readable representation of feature values
        self.preamble = []
        for feat in sorted(self.fvmap.keys()):
            item = self.fvmap[feat]

            # positive feature value
            for ohv, var in enumerate(item['vars']):
                if self.input[var - 1] > 0:
                    break

            if item['name'] in item['onhotlabels'][ohv]:
                self.preamble.append(item['onhotlabels'][ohv])
            else:
                self.preamble.append('{0} = {1}'.format(item['name'], item['onhotlabels'][ohv]))

        # if this sample was already used before, do nothing
        if self.sidv in self.sids:
            return False

        # storing the new sample id
        self.sids.add(self.sidv)

        # forcing a wrong output
        self.oracle.add_clause([-self.sidv] + sample_info['makeunsat'])

        # processing features (soft clauses)
        for fid, feat in enumerate(sorted(self.fvmap.keys())):
            item = self.fvmap[feat]
            selv = self.sels[fid]

            # connecting selectors with the corresponding groups of literals
            for i, var in enumerate(item['vars']):
                self.oracle.add_clause([-self.sidv, -selv, self.input[var - 1]])

        return True  # new sample id is created

    def validate(self, xpl, sample_info, label):
        """
            Validate a given explanation for a given sample.
        """

        self.label = label

        # obtaining a sample id (to be stored in self.sidv)
        self.prepare(sample_info)

        if len(self.encoded_knowledge) > 0:
            self.filter_bg()

        if self.verb > 1:
            print('validating: IF {0} THEN {1}'.format(' AND '.join([self.preamble[f] for f in xpl]), self.label))


        # validation
        res = self.oracle.solve(assumptions=[self.sidv] + [self.sels[f] for f in xpl]) == False
        minimal = None
        expl_ = None

        if res:
            expl_ = self.extract_mus(start_from=[self.sels[f] for f in xpl])
            minimal = len(xpl) == len(expl_)
            expl_ = sorted(map(lambda s: self.smap[s], expl_))

        if self.verb > 1:
            print('explanation is', 'valid' if res else 'invalid')
            print()

        return res, minimal, expl_

    def refine(self, xpl, sample_info):
        """
            Reduce a given (valid!) explanation for a given sample.
        """

        # obtaining a sample id (to be stored in self.sidv)
        self.prepare(sample_info)
        sels = [self.sels[f] for f in xpl]

        if self.verb > 1:
            print('reducing: {0} THEN CLASS'.format(' AND '.join([self.preamble[f] for f in xpl])))

        # here, we perform standard abduction explanation extraction,
        # starting from a given explanation (instead of the instance)
        rxpl = sorted(map(lambda s: self.smap[s], self.compute_axp(start_from=sels)))

        if self.verb > 1:
            print('explanation: {0} THEN CLASS'.format(' AND '.join([self.preamble[f] for f in rxpl])))
            print('explanation size:', len(rxpl))
            print('redundant features:', len(xpl) - len(rxpl))

        return rxpl

    def repair(self, xpl, sample_info):
        """
            Repair a given (invalid!) explanation.
        """

        # obtaining a sample id (to be stored in self.sidv)
        self.prepare(sample_info)

        # the trick to sort features giving preference to the features of xpl
        sels = sorted(self.sels, key=lambda f: f in set(xpl))

        if self.verb > 1:
            print('repairing: {0} THEN CLASS'.format(' AND '.join([self.preamble[f] for f in xpl])))

        # here, we perform standard abduction explanation extraction,
        # giving a preference to the provided explanation
        rxpl = sorted(map(lambda s: self.smap[s], self.compute_axp(start_from=sels)))

        if self.verb > 1:
            print('explanation: {0} THEN CLASS'.format(' AND '.join([self.preamble[f] for f in rxpl])))
            print('explanation size:', len(rxpl))

        return rxpl

    def explain(self, sample_info, label):
        """
            Main explanation procedure.
        """
        self.label = label

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime

        if self.prepare(sample_info):

            self.filter_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime

            if len(self.encoded_knowledge) > 0:
                self.filter_bg()

            self.filter_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.filter_time

            if self.verb > 1:
                print('\nexplaining: IF {0} THEN {1}'.format(' AND '.join(self.preamble), self.label))

            if self.xtype in ('abductive', 'abd'):
                # abductive explanations => MUS computation and enumeration
                if not self.xmin and self.xnum == 1:
                    self.expls = [self.extract_mus()]
                else:
                    self.mhs_mus_enumeration()
            else:  # contrastive explanations => MCS enumeration
                self.mhs_mcs_enumeration()

            # mapping selectors back to features
            self.expls = [sorted(map(lambda s: self.smap[s], xpl)) for xpl in self.expls]

            self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                        resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

            if self.verb > 1:
                self.label = self.label.replace('=', '!=') if self.xtype in ('contrastive', 'con') \
                    else self.label

                if self.xtype in ('abductive', 'abd') and len(self.encoded_knowledge) > 0 \
                        and self.approach == 'use':
                    self.get_new_rvars()

                self.use_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                                resource.getrusage(resource.RUSAGE_SELF).ru_utime

                for xpl in self.expls:
                    print('  explanation: {0} THEN {1}'.format(' AND '.join([self.preamble[f] for f in xpl]), self.label))
                    print('  explanation size:', len(xpl))
                    #print('  redundant features:', len(self.preamble) - len(xpl))

                    if self.xtype in ('abductive', 'abd') and len(self.encoded_knowledge) > 0 \
                            and self.approach == 'use':

                        oriexpl = [self.sels[i] for i in xpl]
                        used_rules = self.extract_usedbg(oriexpl, sample_info)

                        for rvar in used_rules:
                            rule = self.rvar2rule_cl[rvar]['rule']
                            size = len(rule.split(' AND ')) if ' AND ' in rule else 1
                            print('    used rule:', rule)
                            print('    used rule size:', size)

                self.use_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                            resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.use_time

                print('  exp time: {0:.2f}'.format(self.time-self.filter_time))
                if self.xtype in ('abductive', 'abd') and len(self.encoded_knowledge) > 0 \
                        and self.approach == 'use':
                    print('  used rules time: {0:.2f}'.format(self.use_time))

            return self.expls

    def extract_mus(self, start_from=None):
        """
            Compute one abductive explanation.
        """

        def _do_linear(core):
            """
                Do linear search.
            """

            def _assump_needed(a):
                if len(to_test) > 1:
                    to_test.remove(a)
                    self.calls += 1
                    if not self.oracle.solve(assumptions=[self.sidv] + sorted(to_test)):
                        return False
                    to_test.add(a)
                    return True
                else:
                    return True

            to_test = set(core)
            return list(filter(lambda a: _assump_needed(a), core))

        def _do_quickxplain(core):
            """
                Do QuickXplain-like search.
            """

            wset = core[:]
            filt_sz = len(wset) / 2.0
            while filt_sz >= 1:
                i = 0
                while i < len(wset):
                    to_test = wset[:i] + wset[(i + int(filt_sz)):]
                    self.calls += 1
                    if to_test and not self.oracle.solve(assumptions=[self.sidv] + to_test):
                        # assumps are not needed
                        wset = to_test
                    else:
                        # assumps are needed => check the next chunk
                        i += int(filt_sz)
                # decreasing size of the set to filter
                filt_sz /= 2.0
                if filt_sz > len(wset) / 2.0:
                    # next size is too large => make it smaller
                    filt_sz = len(wset) / 2.0
            return wset

        if start_from is None:
            # this call must be unsatisfiable!
            assert self.oracle.solve(assumptions=[self.sidv] + self.sels) == False
        else:
            assert self.oracle.solve(assumptions=[self.sidv] + start_from) == False

        # this is our MUS over-approximation
        core = self.oracle.get_core()
        core.remove(self.sidv)

        self.calls = 1  # we have already made one call

        if self.reduce == 'qxp':
            return _do_quickxplain(core)
        else:  # by default, linear MUS extraction is used
            return _do_linear(core)

    def mhs_mus_enumeration(self):
        """
            Enumerate subset- and cardinality-minimal explanations.
        """

        # result
        self.expls = []

        # just in case, let's save dual (contrastive) explanations
        self.duals = []

        with Hitman(bootstrap_with=[self.sels], htype='sorted' if self.xmin else 'lbx') as hitman:
            # computing unit-size MCSes
            for i, hypo in enumerate(self.sels):
                self.calls += 1
                if self.oracle.solve(assumptions=[self.sidv] + self.sels[:i] + self.sels[(i + 1):]):
                    hitman.hit([hypo])
                    self.duals.append([hypo])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verb > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if self.oracle.solve(assumptions=[self.sidv] + hset):
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(self.sels).difference(set(hset)))

                    model = self.oracle.get_model()
                    for h in removed:
                        if model[abs(h) - 1] != h:
                            unsatisfied.append(h)
                        else:
                            hset.append(h)

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        self.calls += 1
                        if self.oracle.solve(assumptions=[self.sidv] + hset + [h]):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.verb > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)

                    self.duals.append([to_hit])
                else:
                    if self.verb > 2:
                        print('expl:', hset)

                    self.expls.append(hset)

                    if len(self.expls) != self.xnum:
                        hitman.block(hset)
                    else:
                        break

    def mhs_mcs_enumeration(self, unit_mcs=False):
        """
            Enumerate subset- and cardinality-minimal contrastive explanations.
        """

        # result
        self.expls = []

        # just in case, let's save dual (abductive) explanations
        self.duals = []

        with Hitman(bootstrap_with=[self.sels], htype='sorted' if self.xmin else 'lbx') as hitman:
            # computing unit-size MUSes
            for i, hypo in enumerate(self.sels):
                self.calls += 1

                if not self.oracle.solve(assumptions=[self.sidv] + [hypo]):
                    hitman.hit([hypo])
                    self.duals.append([hypo])
                elif unit_mcs and self.oracle.solve(assumptions=[self.sidv] + self.sels[:i] + self.sels[(i + 1):]):
                    # this is a unit-size MCS => block immediately
                    self.calls += 1
                    hitman.block([hypo])
                    self.expls.append([hypo])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verb > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if not self.oracle.solve(assumptions=[self.sidv] + sorted(set(self.sels).difference(set(hset)))):
                    to_hit = self.oracle.get_core()

                    if len(to_hit) > 1 and self.reduce != 'none':
                        to_hit = self.extract_mus(start_from=to_hit)

                    self.duals.append(to_hit)

                    if self.verb > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    if self.verb > 2:
                        print('expl:', hset)

                    self.expls.append(hset)

                    if len(self.expls) != self.xnum:
                        hitman.block(hset)
                    else:
                        break

    def filter_bg(self):
        self.oracle_knowledge = []
        oracle = Solver(name=self.solver, bootstrap_with=self.encoded_knowledge)

        for hypo in self.input:
            oracle.add_clause([hypo])

        assump = list(self.rvar2rcl.keys())
        st, prop = oracle.propagate(assumptions=assump)
        notuse = []
        while not st:
            unsat_ids = assump.index(prop[-1]) + 1 if len(prop) > 0 else 0
            notuse.append(assump[unsat_ids])

            try:
                assump = assump[unsat_ids + 1:]
                st, prop = oracle.propagate(assumptions=assump)
            except:
                st = True

        rvars = set(self.rvar2rcl.keys()).difference(set(notuse))
        for rvar in rvars:
            self.oracle.add_clause(self.rvar2rcl[rvar])
            self.oracle_knowledge.append({'cl': self.rvar2rcl[rvar],
                                          'rule': self.rvar2rule[rvar]})

    def get_new_rvars(self):
        """

        Get the new variables for knowledge for
        computing rules used

        """

        # add background knowledge with a selector
        self.rvars = []
        self.rvar2rule_cl = {}

        for cl_rule in self.oracle_knowledge:
            cl = cl_rule['cl']
            rule = cl_rule['rule']
            rvar = self.vpool.id(str(rule))
            self.rvar2rule_cl[rvar] = {'cl': cl,
                                    'rule': rule}

    def extract_usedbg(self, oriexpl, sample_info):
        # prepare
        self.oracle_ = Solver(name=self.solver, bootstrap_with=self.formula)

        # forcing a wrong output
        self.oracle_.add_clause([-self.sidv] + sample_info['makeunsat'])

        # processing features (soft clauses)
        for fid, feat in enumerate(sorted(self.fvmap.keys())):
            item = self.fvmap[feat]
            selv = self.sels[fid]

            # connecting selectors with the corresponding groups of literals
            for i, var in enumerate(item['vars']):
                self.oracle_.add_clause([-self.sidv, -selv, self.input[var - 1]])

        self.oracle_.add_clause([self.sidv])

        if self.oracle_.solve(assumptions=oriexpl) == False:
            return []

        rvars = list(self.rvar2rule_cl.keys())
        # add background knowledge with a selector
        for rvar in rvars:
            cl = self.rvar2rule_cl[rvar]['cl']
            self.oracle_.add_clause(cl + [-rvar])

        assert self.oracle_.solve(assumptions=oriexpl + rvars) == False

        core = self.oracle_.get_core()
        core = set(core).difference(set(oriexpl))
        #print(f'core: {core}')

        core = list(core)

        def _do_linear(core):
            """
                Do linear search.
            """
            self.call_ = 0

            def _assump_needed(a):
                #print('call:', self.call_)
                self.call_ += 1
                if len(to_test) > 1:
                    to_test.remove(a)
                    if not self.oracle_.solve(assumptions=oriexpl + sorted(to_test)):
                        return False
                    to_test.add(a)
                    return True
                else:
                    return True

            to_test = set(core)
            return list(filter(lambda a: _assump_needed(a), core))

        used_rules = _do_linear(core)

        return used_rules

#
#==============================================================================
def encode_bg(extra, knowledge):
    # load background knowledge
    # not sure whether sum(feature values) equal 1 is necessary or not
    with open(knowledge, 'r') as f:
        background = json.load(f)

    # feature value to variable in CNF formula
    fv2fvar = {}
    for fid in extra['features2vars']:
        fname = extra['features2vars'][fid]['name']
        fvalues = extra['features2vars'][fid]['onhotlabels']
        for i, fvalue in enumerate(fvalues):
            fvar = extra['features2vars'][fid]['vars'][i]
            fv2fvar[(fname, fvalue)] = fvar

    rvar2rcl = {}
    rvar2rule = {}
    cnfs = []
    fvars = list(fv2fvar.values())
    max_fvar = abs(max(fvars, key=lambda l: abs(l)))

    rules = []
    # rules to CNF formula
    for lname in background:
        for lvalue in background[lname]:
            for condition in background[lname][lvalue]:
                rule_formula = []
                rule_formula.append(fv2fvar[(lname, lvalue)])
                premise = []
                for fv in condition:
                    fname = fv['feature']
                    fvalue = fv['value']
                    sign = fv['sign']
                    if sign:
                        rule_formula.append(-fv2fvar[(fname, fvalue)])
                    else:
                        rule_formula.append(fv2fvar[(fname, fvalue)])

                    premise.append('{0} {1} {2}'.format(fname, '=' if sign > 0 else '!=', fvalue))

                rule = 'IF {0} THEN {1} = {2}'.format(' AND '.join(premise), lname, lvalue)
                rules.append(rule)
                cnfs.append(rule_formula)

    encoded_knowledge = []
    for i, rule in enumerate(rules):
        f = cnfs[i]
        rvar = max_fvar + 1 + i  # for the sat solver used to filter out wrong background knowledge
        # different from the formula used to compute explanations
        encoded_knowledge.append(f + [-rvar])
        rvar2rcl[rvar] = f
        rvar2rule[rvar] = rule

    return encoded_knowledge, rvar2rcl, rvar2rule

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

    parser = argparse.ArgumentParser(description='BNN explainer')
    # parser.add_argument('-c', '--config', default=None, type=str,
    #                         help='config file path (default: None)')
    parser.add_argument('-a', '--approach', type=str, default=None,
                        help='Whether extracting useful rules')
    parser.add_argument('-b', '--batch', type=str, default=None,
                        help='Batch')
    parser.add_argument('-c', '--check', type=str, default=None,
                        help='Path to explanations to be checked')
    parser.add_argument('-i', '--ids', type=str, default='-1', help='Sample index (default: -1)')
    parser.add_argument('-I', '--inst', type=int, default=None,
                        help='The number of instances being computed (default: <none>)')
    parser.add_argument('-k', '--knowledge', type=str, default=None,
                        help='knowledge file path')
    parser.add_argument('-l', '--load', default=None, type=str,
                            help='Load stored model from a dir')
    parser.add_argument('-m', '--xmin', action='store_true', help='Target smallest size explanations (default: false)')
    parser.set_defaults(xmin=False)
    parser.add_argument('-n', '--naive', action='store_true', help='Use naive cardinality constraints')
    parser.add_argument('-N', '--xnum', type=str, default=1, help='Number of explanations (default: 1)')
    parser.add_argument('-r', '--reduce', default='lin', type=str,
                            help='Core reduction (default: lin)')
    parser.add_argument('-s', '--solver', default='g3', type=str,
                            help='SAT solver to use (default: g3)')
    parser.add_argument('-t', '--xtype', type=str, default=None, help='Explanation type: abd, con or None (default: None)')
    parser.add_argument('-v', '--verbosity', type=int, default=1, help='Verbosity level')

    ret = parser.parse_args()

    # multiple samples
    ret.ids = rangeexpand(ret.ids)

    # casting xnum to integer
    ret.xnum = -1 if ret.xnum == 'all' else int(ret.xnum)

    return ret


#
#==============================================================================
if __name__ == '__main__':

    args = parse_options()

    if args.approach == 'use':
        args.xtype = 'abd'
    if args.approach == 'check':
        args.xtype = None

    # only MiniCard can handle naive cardinality constraints
    if args.naive:
        args.solver = 'mc'

    if args.load:
        # loading the config
        config = json.load(open(os.path.join(args.load, CONFIG_DEFAULT_FILE_NAME)))
        random.seed(config['manual_seed'])

        # loading the model
        model, train_loader, val_loader, aggregated = load(args, config)

        # encoding
        dummy = torch.tensor(val_loader.dataset.X[0])

        if not args.naive:
            formula, extra = encode(args, config , model, dummy, aggregated,
                    save_path=None, is_checking=False)
        else:
            # naive cardinality constraints for MiniCard
            formula, extra = encode(args, config , model, dummy, aggregated,
                    save_path=None, card_enc=CARD_ENC_NAIVE, is_checking=False)

        if args.knowledge is not None:
            encoded_knowledge, rvar2rcl, rvar2rule = encode_bg(extra, args.knowledge)
        else:
            encoded_knowledge, rvar2rcl, rvar2rule = [], {}, {}

        class_name = aggregated['feature_names'][-1]
        label_lit2lvalue = {}
        for label_ids, lvalue in enumerate(aggregated['class_names']):
            label_lit = extra['winners'][label_ids]
            label_lit2lvalue[label_lit] = lvalue

        if args.xtype:
            # looking for the right samples
            samples = []
            if args.ids[0] != -1:
                bsize = config['train']['batch_size']
                for b, (inputs, target) in enumerate(val_loader):

                    if b * bsize > max(args.ids):
                        break

                    for i, sample in enumerate(inputs):
                        if b * bsize + i > max(args.ids):
                            break
                        elif b * bsize + i in args.ids:
                            samples.append(sample)
            else:
                for b, (inputs, target) in enumerate(val_loader):
                    samples.extend(inputs)

                samples = set(map(lambda l: tuple(l.tolist()), samples))
                samples = sorted(samples)
                samples = list(map(lambda l: torch.LongTensor(l), samples))

                if args.inst is not None and len(samples) > args.inst:
                    random.seed(1000)
                    samples = random.sample(samples, args.inst)

                if args.batch is not None:
                    b1, b2 = args.batch.split(',')
                    b1, b2 = int(b1), int(b2)
                    selected_ids = list(filter(lambda l: l % b2 == b1, range(len(samples))))

                    samples = [samples[i] for i in selected_ids]

            nofex, minex, maxex, avgex, times = [], [], [], [], []
            times = []
            filtertimes = []
            use_times = []

            for sample in samples:
                with BNNExplainer(formula, extra, solver=args.solver, xtype=args.xtype,
                    xnum = args.xnum, smallest = args.xmin, reduce = args.reduce,
                    verbosity = args.verbosity, approach=args.approach) as explainer:

                    # background knowledge
                    explainer.encoded_knowledge = encoded_knowledge
                    explainer.rvar2rcl = rvar2rcl
                    explainer.rvar2rule = rvar2rule

                    # emulation of classification for the given sample
                    sinfo = execute(config, model, sample, aggregated, extra)

                    label = '{} = {}'.format(class_name, label_lit2lvalue[sinfo['winner_lit']])

                    # explanation procedure
                    expls = explainer.explain(sinfo, label)

                    nofex.append(len(expls))
                    if len(expls) > 0:
                        minex.append(min([len(e) for e in expls]))
                        maxex.append(max([len(e) for e in expls]))
                        avgex.append(statistics.mean([len(e) for e in expls]))
                    times.append(explainer.time)
                    filtertimes.append(explainer.filter_time)
                    use_times.append(explainer.use_time)

                    # validation can be done like this
                    #for xpl in expls:
                    #    assert explainer.validate(xpl, sinfo)
                    print('')

            exptimes = [times[i] - filtertimes[i] for i in range(len(times))]



            if args.verbosity > 1:
                print('\nexptimes: {0}'.format(exptimes))
                # print(f'filter times: {filtertimes}')
                #print(f'times: {times}')
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
                if args.xtype in ('abductive', 'abd') and args.approach == 'use':
                    print('tot used rules time: {0:.2f}'.format(sum(use_times)))
                #print('tot filter time: {0:.2f}'.format(sum(filtertimes)))
                #print('tot time: {0:.2f}'.format(sum(times)))
                print('min exp time: {0:.2f}'.format(min(exptimes)))
                print('avg exp time: {0:.2f}'.format(statistics.mean(exptimes)))
                print('max exp time: {0:.2f}'.format(max(exptimes)))

        if args.approach == 'check':
            # mapping between real feature values proxy
            FVMap = collections.namedtuple('FVMap', ['dir', 'opp'])
            fvmap = FVMap(dir={}, opp={})

            data_dir = config['data']['data_dir']
            data_file = config['data']['data_file']
            with open('{0}/{1}.pkl'.format(data_dir, data_file), 'rb') as f:
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

            with open(args.check, 'r') as f:
                expls_info = json.load(f)
            feature_names = list(dtinfo['feature_names'] )

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
                sample = torch.LongTensor(sample)
                samples.append(sample)

                expl = expls_info['stats'][inst]['expl']
                expl = sorted(map(lambda l: feature_names.index(l), expl))
                expls.append(expl)
            
            # validating
            header = ['expl', 'valid?', 'minimal', 'mexpl']
            rows = []
            results = []
            for i, sample in enumerate(samples):
                with BNNExplainer(formula, extra, solver=args.solver, xtype=args.xtype,
                    xnum = args.xnum, smallest = args.xmin, reduce = args.reduce,
                    verbosity = args.verbosity, approach=args.approach) as explainer:

                    # background knowledge
                    explainer.encoded_knowledge = encoded_knowledge
                    explainer.rvar2rcl = rvar2rcl
                    explainer.rvar2rule = rvar2rule

                    # emulation of classification for the given sample
                    sinfo = execute(config, model, sample, aggregated, extra)

                    label = '{} = {}'.format(class_name, label_lit2lvalue[sinfo['winner_lit']])
                    expl = expls[i]

                    # validating explanation
                    res, minimal, expl_ = explainer.validate(expl, sinfo, label)
                    results.append(res)

                    row = []
                    row.append('IF {0} THEN {1}'.format(' AND '.join([explainer.preamble[f] for f in expl]), explainer.label))
                    row.append(res)
                    row.append(minimal)
                    if expl_ is None:
                        row.append(expl_)
                    else:
                        row.append('IF {0} THEN {1}'.format(' AND '.join([explainer.preamble[f] for f in expl_]), explainer.label))
                    rows.append(row)

            quantise = config['data']['data_dir'].split('/quantise/')[-1].split('/', maxsplit=1)[0]
            dtname = config['data']['test_file'].split('_data.csv')[0]

            if '/lime/' in args.check:
                appr = 'lime'
            elif '/shap/' in args.check:
                appr = 'shap'
            elif '/anchor/' in args.check:
                appr = 'anchor'
            else:
                exit(1)

            if '/dl/' in args.check:
                model = 'dl'
            elif '/bt/' in args.check:
                model = 'bt'
            elif '/bnn/' in args.check:
                model = 'bnn'
            else:
                exit(1)

            saved_dir = '../stats/correctness/{0}/{1}'.format(appr, 'size5' if args.knowledge else 'ori')

            if not os.path.isdir((saved_dir)):
                os.makedirs(saved_dir)

            name = '{0}_{1}_{2}.csv'.format(model, quantise, dtname)
            with open("{0}/{1}".format(saved_dir, name), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

    exit()
