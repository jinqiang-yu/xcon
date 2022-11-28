#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## options.py
##

#
#==============================================================================
from __future__ import print_function
import decimal
import getopt
import math
import os
from pysat.card import EncType
import sys

#
#==============================================================================
encmap = {
    "pw": EncType.pairwise,
    "seqc": EncType.seqcounter,
    "cardn": EncType.cardnetwrk,
    "sortn": EncType.sortnetwrk,
    "tot": EncType.totalizer,
    "mtot": EncType.mtotalizer,
    "kmtot": EncType.kmtotalizer,
    "native": EncType.native
}


#
#==============================================================================
class Options(object):
    """
        Class for representing command-line options.
    """

    def __init__(self, command=None):
        """
            Constructor.
        """
        self.am1 = False
        self.approach = 'maxsat'
        self.blo = False
        self.bsymm = False
        self.cdump = False
        self.cenc = 'pw'
        self.dataset = None
        self.enc = 'cardn'
        self.exhaust = False
        self.inst = None
        self.mapfile = None
        self.minz = False
        self.model = None
        self.noccheck = False
        self.pdump = False
        self.plimit = 0
        self.primer = 'sortedplimit'
        self.ranges = 0
        self.reduce = 'none'
        self.separator = ','
        self.smallest = False
        self.solver = 'g3'
        self.to_compute = 'all'
        self.trim = 0
        self.unit_mcs = False
        self.use_cld = False
        self.use_mhs = False
        self.verb = 0
        self.weighted = False
        self.xnum = 1
        self.xtype = 'abd'
        self.blk = False

        self.order = 'size'
        self.cutp = 0
        self.save_to = None
        self.knowledge = None
        self.search = 'lin'

        if command:
            self.parse(command)

    def parse(self, command):
        """
            Parser.
        """

        self.command = command

        try:
            opts, args = getopt.getopt(command[1:],
                                    '1a:bBc:C:dD:e:hHk:l:m:Mn:r:o:p:R:s:S:t:uvwx:X',
                                    ['am1',
                                        'approach=',
                                     'blk',
                                     'blo',
                                        'bsymm',
                                        'cdump',
                                     'cenc=',
                                     'cut=',
                                        'enc=',
                                        'exhaust',
                                        'help',
                                     'knowledge',
                                        'map-file=',
                                        'model=',
                                        'minimum',
                                        'minz',
                                        'no-ccheck',
                                        'order=',
                                        'pdump',
                                        'plimit=',
                                        'primer=',
                                        'ranges=',
                                     'reduce=',
                                     'save-to=',
                                     'search=',
                                        'sep=',
                                        'solver=',
                                        'trim=',
                                     'unit-mcs',
                                        'verbose',
                                        'weighted',
                                     'xnum=',
                                     'xtype=',
                                     ])
        except getopt.GetoptError as err:
            sys.stderr.write(str(err).capitalize() + '\n')
            self.usage()
            sys.exit(1)

        for opt, arg in opts:
            if opt in ('-1', '--am1'):
                self.am1 = True
            elif opt in ('-a', '--approach'):
                self.approach = str(arg)
            elif opt in ('-b', '--blo'):
                self.blo = True
            elif opt in ('-B', '--bsymm'):
                self.bsymm = True
            elif opt in ('-c', '--cenc'):
                self.cenc = str(arg)
            elif opt in ('-C', '--cut'):
                self.cutp = float(arg)
            elif opt == '--cdump':
                self.cdump = True
            elif opt in ('-d', '--use-cld'):
                self.use_cld = True
            elif opt in ('-D', '--dataset'):
                self.dataset = str(arg)
            elif opt in ('-e', '--enc'):
                self.enc = str(arg)
            elif opt in ('-h', '--help'):
                self.usage()
                sys.exit(0)
            elif opt in ('-H', '--use-mhs'):
                self.use_mhs = True
            elif opt in ('-k', '--knowledge'):
                self.knowledge = str(arg)
            elif opt in ('-l', '--plimit'):
                self.plimit = int(arg)
                if self.plimit == 'best':
                    self.plimit = -1
                else:
                    self.plimit = int(self.plimit)
            elif opt in ('-m', '--model'):
                self.model = str(arg)
            elif opt in ('-M', '--minimum'):
                self.smallest = True
            elif opt == '--map-file':
                self.mapfile = str(arg)
            elif opt == '--minz':
                self.minz = True
            elif opt in ('-n', '--xnum'):
                self.xnum = str(arg)
                self.xnum = -1 if self.xnum == 'all' else int(self.xnum)
            elif opt == '--no-ccheck':
                self.noccheck = True
            elif opt in ('-o', '--order'):
                self.order = str(arg)
            elif opt in ('-p', '--primer'):
                self.primer = str(arg)
            elif opt == '--pdump':
                self.pdump = True
            elif opt in ('-r', '--ranges'):
                self.ranges = int(arg)
            elif opt in ('-R', '--reduce'):
                self.reduce = str(arg)
            elif opt == '--save-to':
                self.save_to = str(arg)
            elif opt == '--sep':
                self.separator = str(arg)
            elif opt in ('-s', '--solver'):
                self.solver = str(arg)
            elif opt in ('-S', '--search'):
                self.search = str(arg)
            elif opt in ('-t', '--trim'):
                self.trim = int(arg)
            elif opt in ('-u', '--unit-mcs'):
                self.unit_mcs = True
            elif opt in ('-v', '--verbose'):
                self.verb += 1
            elif opt in ('-w', '--weighted'):
                self.weighted = True
            elif opt in ('-x', '--xtype'):
                self.xtype = str(arg)
            elif opt in ('-X', '--exhaust'):
                self.exhaust = True
            elif opt == '--blk':
                self.blk = True
            else:
                assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

        self.enc = encmap[self.enc]

        # we expect a dataset to be given
        assert self.dataset, 'Wrong or no dataset is given'
        if args:
            self.inst = tuple(args[0].split(','))

        self.cenc = encmap[self.cenc]

    def usage(self):
        """
            Print usage message.
        """

        print('Usage: ' + os.path.basename(self.command[0]) + ' [options] training-data')
        print('Options:')
        print('        -1, --am1                  Detect AtMost1 constraints in RC2' )
        print('        -a, --approach=<string>    Whether extracting useful rules.')
        print('                                   Available values: apriori, maxsat (default = maxsat)')
        print('        -b, --blo                  Apply BLO when solving MaxSAT')
        print('        -B, --bsymm                Use symmetry breaking constraints in SAT-based approaches')
        print('        -c, --cenc=<string>        Cardinality encoding to use')
        print('                                   Available values: cardn, kmtot, mtot, sortn, tot (default = pw)')
        print('        -C, --cut=<float>          Cutting point of computing rules')
        print('                                   Available values: [0 .. 1] (default: 0)')
        print('        --cdump                    Dump largest consistent subset of input samples')
        print('        -d, --use-cld              Use CLD heuristic')
        print('        -D, --dataset=<string>     Path to dataset file (default: <none>)')
        print('        -e, --enc=<string>         Encoding to use')
        print('                                   Available values: cardn, kmtot, mtot, sortn, tot (default = cardn)')
        print('        -h, --help')
        print('        -H, --use-mhs              Use hitting set based enumeration')
        print('        -k, --knowledge=<string>   Path to background knowledge file (default: <none>)')
        print('        -l, --plimit=<int>         Compute at most this number of primes per sample')
        print('                                   Available values: [0 .. INT_MAX] (default: 0)')
        print('        -m, --model=<string>       Path to model file (default: <stdin>)')
        print('        -M, --minimum              Compute a smallest size explanation (instead of a subset-minimal one)')
        print('        --map-file=<string>        Path to a file containing a mapping to original feature values. (default: none)')
        print('        -m, --minz                 Use unsatisfiable core heuristic minimization in RC2')
        print('        -n, --xnum=<int>           Number of explanations to compute')
        print('                                   Available values: [1, INT_MAX], all (default = 1)')
        print('        --no-ccheck                Do not perform consistency check')
        print('        -o, --order=<string>       The order of computing rules.')
        print('                                   Available values: size, sup (default = size)')
        print('        -p, --primer=<string>      Prime implicant enumerator to use')
        print('                                   Available values: lbx, mcsls, sorted (default = sorted)')
        print('        --pdump                    Dump MaxSAT formula for enumerating primes')
        print('                                   (makes sense only when using an MCS-based primer)')
        print('        -r, --ranges=<int>         Try to cluster numeric features values into this number of ranges')
        print('                                   Available values: [0 .. INT_MAX] (default = 0)')
        print('        -R, --reduce=<string>      Extract an MUS from each unsatisfiable core')
        print('                                   Available values: lin, none, qxp (default = none)')
        print('        --sep=<string>             Field separator used in input file (default = \',\')')
        print('        -s, --solver=<string>      SAT solver to use')
        print('                                   Available values: g3, lgl, m22, mc, mgh (default = m22)')
        print('        --save-to=<string>         Where the computed rules should be saved')
        print('                                   Default value: None')
        print('        -t, --trim=<int>           Trim unsatisfiable core at most this number of times')
        print('                                   Available values: [0 .. INT_MAX] (default = 0)')
        print('        -u, --unit-mcs             Detect and block unit-size MCSes')
        print('        -v, --verbose              Be more verbose')
        print('        -w, --weighted             Minimize the total number of literals')
        print('        -x, --xtype=<string>       Type of explanation to compute: abductive or contrastive')
        print('                                   Available values: abd, con (default = abd)')
        print('        -X, --exhaust              Do unsatisfiable core exhaustion in RC2')
        print('        --blk                      Block duplicate rules')
