#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## enc.py
##

import sys
import os
import glob
import json
import collections
import pandas as pd
import collections
import math
import statistics
import csv

def cmptmax(files, key):
    maxkey = 0
    for file in files:
        with open(file, 'r') as f:
            keydict = json.load(f)

        for dt in keydict['stats']:
            k = keydict['stats'][dt][key]
            if k > maxkey:
                maxkey = k
    return maxkey

def parse_csv(files, appr, model, bg):
    stats_dict = {"preamble": {"program": None, "prog_args": None, "prog_alias": None, "benchmark": None},
                 "stats": {}}

    size_stats_dict =  {minimal: {"preamble": {"program": None, "prog_args": None, "prog_alias": None, "benchmark": None},
                  "stats": {}} for minimal in [True, False]}

    for e in ['program', 'prog_args', 'prog_alias']:
        stats_dict['preamble'][e] = "{0}$_{1}$".format(appr, '{' + model + '}') if bg == 'ori' \
                                    else "{0}$^r_{1}$".format(appr, '{' + model + '}')

        for minimal in [True, False]:
            size_stats_dict[minimal]['preamble'][e] = "{0}${1}_{2}$".format(appr, '^m' if minimal else '', '{' + model + '}') if bg == 'ori' \
                else "{0}$^{1}_{2}$".format(appr, '{rm}' if minimal else '{r}', '{' + model + '}')

    for file in files:
        dtname = file.rsplit('/')[-1].split('_', maxsplit=1)[-1].replace('.csv', '')
        df = pd.read_csv(file)
        correctness = sum(df['valid?']) / len(df['valid?']) * 100
        correctness = round(correctness, 2)

        stats_dict['stats'][dtname] = {'status': True, 'correctness': correctness}
        minimals = []
        for i, valid in enumerate(df['valid?']):
            if valid:
                minimal = df['minimal'][i]
                minimals.append(minimal)

                expl = df['expl'][i]
                size = len(expl.split(' AND '))
                expl_ = df['mexpl'][i]
                size_ = len(expl_.split(' AND '))

                dtname_inst = '{0}_inst{1}'.format(dtname, i)

                size_stats_dict[False]['stats'][dtname_inst] = {'status': True, 'expl':  expl, 'size': size}
                size_stats_dict[True]['stats'][dtname_inst] = {'status': True, 'expl':  expl_, 'size': size_}

    saved_dir = '../stats/correctness/summary'

    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    saved_file = '../stats/correctness/summary/{0}_{1}_{2}.json'.format(appr, model, bg)

    with open(saved_file, 'w') as f:
        json.dump(stats_dict, f, indent=4)

    for minimal in size_stats_dict:
        saved_file = '../stats/correctness/summary/dt_{0}_{1}_{2}{3}.json'.format(appr, model, bg,
                                                                          '_m' if minimal else '')
        with open(saved_file, 'w') as f:
            json.dump(size_stats_dict[minimal], f, indent=4)


def correctcactus(files, key, bg):

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

    saved_dir = '../plots/hexp/correctness'

    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    pdf_name = '{0}/correctness_cactus_{1}.pdf'.format(saved_dir, 'bg' if bg else 'nobg')

    cmd = 'python ./gnrt_plots/mkplot/mkplot.py --font-sz 18 -w 1.5 --ms 9 --sort --reverse ' \
          '-l --legend prog_alias --xlabel Datasets --ylabel "Correctness(\%)" -t {0} ' \
          '--lloc "upper right" --ymin {1} -b pdf ' \
          '--save-to {2} -k {3} {4}'.format(maxkey,
                                            minkey,
                                            pdf_name,
                                            key,
                                            ' '.join(files))

    os.system(cmd)

def grt_size_scatter(files, key, appr, model, bg):
    maxkey = cmptmax(files, key)
    maxkey = maxkey if maxkey >= 1 else 1

    saved_dir = '../plots/hexp/size'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    pdf_name = '{0}/size_scatter_{1}_{2}_{3}.pdf'.format(saved_dir, appr, model, 'bg' if bg else 'nobg')

    cmd = 'python ./gnrt_plots/mkplot/mkplot.py ' \
          '-l -p scatter -b pdf --save-to {0} -k {1} --shape squared --font-sz 25 --tol-loc none ' \
          ' -t {2} --ymax {2} {3}'.format(pdf_name,
                                          key,
                                          maxkey,
                                          ' '.join(files))
    #print('\ncmd:\n{0}'.format(cmd))
    os.system(cmd)

if __name__ == '__main__':

    apprmod2files = collections.defaultdict(lambda : [])

    for root, dirs, files in os.walk('../stats/correctness/'):
        for file in files:
            if file.endswith('csv'):
                appr = root.split('/correctness/')[-1].split('/', maxsplit=1)[0]
                model = file.split('_', maxsplit=1)[0]
                bg = 'ori' if '/ori' in root else 'size5'
                apprmod2files[tuple([appr, model, bg])].append(os.path.join(root, file))

    for appr, model, bg in apprmod2files:
        files = apprmod2files[tuple([appr, model, bg])]
        parse_csv(files, appr, model, bg)


    bgdt2files = collections.defaultdict(lambda : [])

    files = glob.glob('../stats/correctness/summary/*.json')

    for file in files:
        if file.rsplit('/')[-1].startswith('dt_'):
            dt = True
        else:
            dt = False

        bg = '_ori' not in file
        bgdt2files[tuple([bg, dt])].append(file)

    for bg, dt in bgdt2files:
        files = bgdt2files[tuple([bg, dt])]

        if dt:
            key = 'size'
            m_files = list(filter(lambda l: '_m.json' in l, files))
            for m_f in m_files:
                o_f = m_f.replace('_m.json', '.json')
                if '_dl_' in m_f:
                    model = 'dl'
                elif '_bt_' in m_f:
                    model = 'bt'
                elif '_bnn_' in m_f:
                    model = 'bnn'
                else:
                    print('model')
                    exit(1)

                if 'lime_' in m_f:
                    appr = 'lime'
                elif 'shap_' in m_f:
                    appr = 'shap'
                elif 'anchor_' in m_f:
                    appr = 'anchor'
                else:
                    print('appr')
                    exit(1)

                grt_size_scatter([m_f, o_f], key, appr, model, bg)
        else:
            key = 'correctness'
            correctcactus(files, key, bg)

    exit()



