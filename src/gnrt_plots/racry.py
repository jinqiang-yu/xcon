#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## enc.py
##

import sys
import os
import glob
import json
import pandas as pd
import collections
import statistics

class Rule(object):
    def __init__(self, label, feats=[]):
        self.label = label
        self.feats = feats[:]

    def __str__(self):
        output = []
        for feat in self.feats:
            f = feat['feature']
            value = feat['value']
            sign = feat['sign']

            output.append('({0}, {1})'.format(f, value) if sign else
                          'NOT ({0}, {1})'.format(f, value))
        return '{0} => {1}'.format(output, (self.label['label'], self.label['lvalue']))

    def add_feat(self, feat):
        self.feats.append(feat)

    def __del__(self):
        del self

    def issat(self, data):

        for feat in self.feats:
            f = feat['feature']
            value = feat['value']
            sign = feat['sign']
            if str(data[f]) == value:
                if not sign:
                    return True
            else:
                if sign:
                    return True

        if str(data[self.label['label']]) == self.label['lvalue']:
            return True
        else:
            return False

    def lits(self):
        return len(self.feats)


class Validator(object):
    def __init__(self, bgs):

        self.parse_bgs(bgs)

    def parse_bgs(self, bgs):

        self.bgs = []
        with open(bgs, 'r') as f:
            bgs_dict = json.load(f)

        #print(f'bgs_dict: {bgs_dict}')

        for label in bgs_dict:
            for lvalue in bgs_dict[label]:
                for features in bgs_dict[label][lvalue]:
                    rule = Rule(label={'label': label, 'lvalue': lvalue})
                    #print(f'label: {label}')
                    #print(features)

                    # single rule
                    for feat in features:
                        #print(f'feat: {feat}')
                        rule.add_feat(feat)

                    self.bgs.append(rule)

    def accry(self, df):

        accry_dict = {'rules': {}, 'accry': {}}

        lit2accry = collections.defaultdict(lambda : [])

        accry = []


        high_acry_rules = {high_acry: collections.defaultdict(lambda : collections.defaultdict(lambda : []))
                           for high_acry in [1.0, 0.99]}

        high_acry_counts = {high_acry:0 for high_acry in [1.0, 0.99]}

        for rule in self.bgs:
            #print(f'lits: {rule.lits()}')
            all = len(df)
            sat = 0
            for i in range(len(df)):
                #print(f'\nrule: {rule}')
                #print(df.iloc[i])
                if rule.issat(df.iloc[i]):
                    sat += 1
                    #print('sat')
                else:
                    #print('unsat')
                    pass
            accry.append(sat / all)

            for high_acry in high_acry_rules:
                if sat / all >= high_acry:
                    high_acry_rules[high_acry][rule.label['label']][rule.label['lvalue']].append(rule.feats)
                    high_acry_counts[high_acry] += 1

            #print('accry:', round(sat / all, 3))

            accry_dict['rules'][str(rule)] = {'lits': rule.lits(),
                                              'accry': round(sat / all, 3)}

            lit2accry[rule.lits()].append(round(sat / all, 3))

        accry = sorted(accry)
        med_accry = statistics.median(accry)
        avg_accry = statistics.mean(accry)
        #print('med_accry: %.2f' % (med_accry))
        #print('avg_accry: %.2f' % (avg_accry))

        accry_dict['accry']['overall'] = {'med_accry': round(med_accry, 3),
                                          'avg_accry': round(avg_accry, 3)}

        for lit in sorted(lit2accry.keys()):
            accry = lit2accry[lit]
            med_accry = statistics.median(accry)
            avg_accry = statistics.mean(accry)

            accry_dict['accry'][lit] = {'med_accry': round(med_accry, 3),
                                              'avg_accry': round(avg_accry, 3)}

        return accry_dict

def accrydist(files):

    #for key in ['medaccry', 'avgaccry']:
    for key in ['avgaccry']:
        maxkey = 0
        minkey = 100
        for file in files:
            with open(file, 'r') as f:
                info = json.load(f)
            acrys = [info['stats'][dt][key] for dt in info['stats']]

            cur_maxk = max(acrys)
            cur_mink = min(acrys)

            if cur_maxk > maxkey:
                maxkey = cur_maxk
            if cur_mink < minkey:
                minkey = cur_mink

        cmd = 'python ./gnrt_plots/mkplot/mkplot.py --font-sz 18 -w 1.5 --ms 9 --sort ' \
              '-l --legend prog_alias --xlabel Datasets --ylabel "Accuracy(\%)" -t {0} ' \
              '--lloc "upper right" --ymin {1} -b pdf ' \
              '--save-to {2} -k {3} {4}'.format(maxkey,
                                                minkey,
                                                f'../plots/avg_accry.pdf',
                                                key,
                                                ' '.join(files))
        os.system(cmd)

def acrytable(files, key):
    r2acry = collections.defaultdict(lambda: [])

    for file in files:
        with open(file, 'r') as f:
            info = json.load(f)
        r = info["preamble"]["prog_arg"]
        for dt in info['stats']:
            acry = info['stats'][dt][key]
            r2acry[r].append(acry)
    rs = sorted(r2acry.keys())
    rs = rs[:-1] + rs[-1:]
    for r in rs:
        r2acry[r] = statistics.mean(r2acry[r])

    head = ['\\begin{table}[t!]',
            '\\centering',
            '\\caption{Average accuracy of extracted rules.}',
            '\\label{tab:racry}',
            '\\scalebox{0.84}{',
            '\\begin{tabular}{ccccccc}\\toprule']

    rows = []
    ' & rule$_1$ & rule$_2$ & rule$_3$ & rule$_4$ & rule$_5$ & rule$_{all}$ \\ \midrule'
    row = ' & ' + ' & '.join(rs) + ' \\\\ \\midrule'
    rows.append(row)
    row = ['Accuracy (\\%)'] + ['$' + str(round(r2acry[r], 2)) + '$' for r in r2acry]
    row = ' & '.join(row) + ' \\\\ \\bottomrule'
    rows.append(row)

    end = ['\\end{tabular}',
           '}',
           '\\end{table}']

    print('\n'.join(head + rows + end))

if __name__ == '__main__':

    isplot = True if 'plot' in sys.argv[1].lower() else False

    if not isplot:
        """
        
        compute accuracy
        
        """
        rule_files = glob.glob('../rules/size/*.json')

        dt2path = {}

        for root, dirs, files in os.walk('../bench/cv/test/quantise/'):
            for file in files:
                if file.endswith('csv') and '_data' not in file:
                    q = root[root.rfind('quantise/') + 1 : ].split('/')[1][1:]
                    name = file[ : file.rfind('_test')].rsplit('/')[-1]
                    train_pair = file[file.rfind('_test') + 5]
                    qn = f'q{q}_{name}_train{train_pair}'
                    test = os.path.join(root, file)
                    dt2path[qn] = test


        dt2acry = collections.defaultdict(lambda : [])
        sizes = set()

        for r_file in rule_files:
            qn = r_file[:r_file.rfind('_train') + 7].rsplit('/')[-1]
            test = dt2path[qn]

            dtname = r_file[:r_file.rfind('_train')].rsplit('/')[-1]
            df = pd.read_csv(test)
            validator = Validator(r_file)
            accry_dict = validator.accry(df)

            dt2acry[dtname].append(accry_dict['accry'])
            sizes = sizes.union(set(accry_dict['accry'].keys()))

        sizes = set(map(lambda l: str(l), sizes))
        size2acry = {size if 'all' in size else int(size):
                         {"preamble": {"benchmark": 'rule$_{0}$' if 'all' not in size else 'rule$_{all}$',
                                       "prog_arg": 'rule$_{0}$' if 'all' not in size else 'rule$_{all}$',
                                       "program": 'rule$_{0}$' if 'all' not in size else 'rule$_{all}$',
                                       "prog_alias": 'rule$_{0}$' if 'all' not in size else 'rule$_{all}$'},
                          "stats": {}}
                    for size in sizes}

        for dt in dt2acry:
            s2medacry = collections.defaultdict(lambda : [])
            s2avgacry = collections.defaultdict(lambda : [])
            for pair_acrys in dt2acry[dt]:
                for size in pair_acrys:
                    med_acry = pair_acrys[size]['med_accry']
                    avg_acry = pair_acrys[size]['avg_accry']
                    s2medacry[size].append(med_acry)
                    s2avgacry[size].append(avg_acry)
            for size in s2avgacry:
                size2acry[size]["stats"][dt] = {"status": True,
                                                "medaccry": statistics.mean(s2medacry[size])*100,
                                                "avgaccry": statistics.mean(s2avgacry[size])*100}

        saved_dir = '../stats/rule_acry'

        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)

        for size in size2acry:
            json_path = f'{saved_dir}/{size}.json'
            with open(json_path, 'w') as f:
                json.dump(size2acry[size], f, indent=4)
    else:
        """
        
        Generate accuracy table and plot
        
        """
        #print('generating plot')
        files = glob.glob('../stats/rule_acry/*.json')
        #table
        acrytable(files, key='avgaccry')
        #plot
        accrydist(files)

    exit()


