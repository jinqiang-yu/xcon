#!/usr/bin/env python3
#-*- coding:utf-8 -*-

from __future__ import print_function
from data import Data
from options import Options
import os
import resource
import sys
from xgbooster import XGBooster
import random

def enumerate_all(options, useumcs, prefix=''):
    # setting the right preferences
    options.reduce = 'lin'
    options.useumcs = useumcs

    # reading all unique samples
    with open(options.files[0], 'r') as fp:
        lines = fp.readlines()[1:]
    lines = set(map(lambda l: l[: l.rfind(',')], lines))
    lines = sorted(lines)

    if options.nof_inst is not None and len(lines) > options.nof_inst:
        random.seed(1000)
        lines = random.sample(lines, options.nof_inst)

    if options.batch is not None:
        b1, b2 = options.batch.split(',')
        b1, b2 = int(b1), int(b2)
        selected_ids = list(filter(lambda l: l % b2 == b1, range(len(lines))))

        lines = [lines[i] for i in selected_ids]

    # timers and other variables
    times, calls = [], []
    xsize, exlen = [], []
    filtertimes = []
    use_times = []

    # doing everything incrementally is expensive;
    # let's restart the solver for every 10% of instances
    tested = set()
    for i, s in enumerate(lines):
        # enumerate explanations only for the first 10% of samples
        #if i % (len(lines) / 10) == 0:
        xgb = XGBooster(options, from_model=options.files[1])

        # encode it and save the encoding to another file
        xgb.encode()

        options.explain = [float(v.strip()) for v in s.split(',')]

        if tuple(options.explain) in tested:
            continue

        tested.add(tuple(options.explain))
        #print(prefix, 'sample {0}: {1}'.format(i, ','.join(s.split(','))))

        #timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
        #        resource.getrusage(resource.RUSAGE_SELF).ru_utime

        expls, time, filter_time, use_time = xgb.explain(options.explain)

        #timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
        #        resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer

        filtertimes.append(filter_time)
        times.append(time)
        use_times.append(use_time)

        '''
        print(prefix, 'expls:', expls)
        print(prefix, 'nof x:', len(expls))
        print(prefix, 'timex: {0:.2f}'.format(timer))
        print(prefix, 'calls:', xgb.x.calls)
        print(expls)
        print(prefix, 'Msz x:', max([len(x) for x in expls]) if expls[0] is not None else 0)
        print(prefix, 'msz x:', min([len(x) for x in expls]) if expls[0] is not None else 0)
        print(prefix, 'asz x: {0:.2f}'.format(sum([len(x) for x in expls]) / len(expls) if expls[0] is not None else 0))
        print('')
        '''

        calls.append(xgb.x.calls)
        xsize.append(sum([len(x) for x in expls]) / float(len(expls)) if expls[0] is not None else 0)
        exlen.append(len(expls))


    print('')
    exptimes = [times[i] - filtertimes[i] for i in range(len(times))]

    print(f'exptimes: {exptimes}')
    print('all samples:', len(lines))
    prefix = ''
    # reporting the time spent
    print('total exp time: {1:.2f}'.format(prefix, sum(exptimes)))
    if options.xtype == 'abductive' and options.approach == 'use':
        print('used rules time: {1:.2f}'.format(prefix, sum(use_times)))
    print('max exp time per instance: {1:.2f}'.format(prefix, max(exptimes)))
    print('min exp time per instance: {1:.2f}'.format(prefix, min(exptimes)))
    print('avg exp time per instance: {1:.2f}'.format(prefix, sum(exptimes) / len(exptimes)))
    print('total oracle calls: {1}'.format(prefix, sum(calls)))
    print('max oracle calls per instance: {1}'.format(prefix, max(calls)))
    print('min oracle calls per instance: {1}'.format(prefix, min(calls)))
    print('avg oracle calls per instance: {1:.2f}'.format(prefix, float(sum(calls)) / len(calls)))
    print('avg number of explanations per instance: {1:.2f}'.format(prefix, float(sum(exlen)) / len(exlen)))
    print('avg explanation size per instance: {1:.2f}'.format(prefix, float(sum(xsize)) / len(xsize)))
    print('')


if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    # subset-minimal
    enumerate_all(options, useumcs=False, prefix='')
