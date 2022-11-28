#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## exp.py
##

# imported modules
#==============================================================================
import collections
from dlist import FeatureValue
from pysat.examples.hitman import Hitman
from pysat.solvers import Solver
import resource


#
#==============================================================================
class DLExplainer(object):
    """
        Explainer of decision lists.
    """

    def __init__(self, dlist, encoding, options):
        """
            Constructor.
        """

        # copying the model
        self.dlist = dlist

        # copying the options
        self.options = options

        # variable ids
        self.vars = encoding.vars

        # creating the dict of oracles (for each label)
        self.oracles = {}

        for lb in encoding.encs:
            self.oracles[lb] = Solver(name=self.options.solver,
                    bootstrap_with=encoding.encs[lb].hard)

        # here is a selector for each instance
        self.svar = None

        # current label
        self.label = None

        # a reference to the current oracle
        self.oracle = None

        # number of oracle calls involved
        self.calls = 0


        self.encoding = encoding

    def __del__(self):
        """
            Destructor.
        """

        for label in self.oracles:
            self.oracles[label].delete()

    def prepare_oracle(self, instance):
        """
            Prepare the interals for dealing with the current instance.
        """

        # here we assume instance is given as 'feature_name1=value1,...'
        assert all('=' in fv for fv in instance), 'Unexpected instance format.'
        # creating the list of features
        #inst = [FeatureValue(*fv.split('=')) for fv in instance]
        inst = [FeatureValue(fv[ : fv.find('=')], fv[fv.find('=') + 1: ]) for fv in instance]

        # running the classifier to determine the class
        self.label = self.dlist.execute(inst)
        # updating the reference to the current oracle
        self.oracle = self.oracles[self.label]
        # creating assumption literals
        self.hypos = [self.vars.id(fv) for fv in inst]

        # checking if the entailment holds
        # also, this SAT oracle will be used to get an initial core
        assert self.oracle.solve(assumptions=self.hypos) == False, \
                'No entailment for instance \'{0}\''.format(','.join(instance))

    def explain(self, instance, smallest=True, xtype='abd', xnum=1,
            unit_mcs=False, use_cld=False, use_mhs=False, reduce_='none'):
        """
            Interface for computing an explanation.
        """
        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # prepare the oracle for dealing with the current instance
        self.prepare_oracle(instance)

        self.filter_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                           resource.getrusage(resource.RUSAGE_SELF).ru_utime

        if self.options.knowledge:
            self.filter_bg()

        self.filter_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                           resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.filter_time

        if self.options.verb:
            preamble = [str(self.vars.obj(i)) for i in self.hypos]
            print('  inst: "IF {0} THEN {1}"'.format(' AND '.join(preamble), self.label))

        # calling the explanation procedure
        self._explain(instance, smallest=smallest, xtype=xtype, xnum=xnum,
                unit_mcs=unit_mcs, use_cld=use_cld, use_mhs=use_mhs,
                reduce_=reduce_)

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        self.use_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                        resource.getrusage(resource.RUSAGE_SELF).ru_utime

        if self.options.verb:
            if self.expls[0] != None:
                if xtype in ('contrastive', 'con'):
                    self.label.pos = False

                if xtype in ('abductive', 'abd') and self.options.knowledge is not None \
                    and self.options.approach == 'use':
                    self.get_new_rvars()
                    self.oracle_ = Solver(name=self.options.solver,
                                         bootstrap_with=self.encoding.encs[self.label].hard)

                    for rvar in self.rvar2rule_cl:
                        self.oracle_.add_clause(self.rvar2rule_cl[rvar]['cl'] + [-rvar])

                for expl in self.expls:
                    preamble = [str(self.vars.obj(i)) for i in expl]

                    if xtype in ('contrastive', 'con'):
                        preamble = [l.replace('==', '!=') for l in preamble]

                    print('  expl: "IF {0} THEN {1}"'.format(' AND '.join(preamble), self.label))
                    print('  # hypos left:', len(expl))

                    if xtype in ('abductive', 'abd') and self.options.knowledge is not None \
                            and self.options.approach == 'use':
                        used_rules = self.extract_usedbg(expl)

                        for rvar in used_rules:
                            rule = self.rvar2rule_cl[rvar]['rule']
                            print('    used rule:', rule)
                            print('    # used rule hypos left:', len(rule))

            self.use_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                            resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.use_time
            print('  exp time: {0:.2f}\n'.format(self.time-self.filter_time))
            if xtype in ('abductive', 'abd') and self.options.knowledge is not None \
                    and self.options.approach == 'use':
                print('  used rules time: {0:.2f}\n'.format(self.use_time))

        # here we return the last computed explanation
        return self.expls

    def _explain(self, instance, smallest=True, xtype='abd', xnum=1,
            unit_mcs=False, use_cld=False, use_mhs=False, reduce_='none'):
        """
            Compute an explanation.
        """
        if xtype in ('abductive', 'abd'):
            # abductive explanations => MUS computation and enumeration
            if not smallest and xnum == 1:
                self.expls = [self.extract_mus(reduce_=reduce_)]
            else:
                self.mhs_mus_enumeration(xnum, smallest=smallest)

        else:  # contrastive explanations => MCS enumeration
            if use_mhs:
                self.mhs_mcs_enumeration(xnum, smallest, reduce_)
            else:
                if not smallest:
                    # deleting all the MCSes computed for the previous instance
                    if self.svar:
                        self.oracle.add_clause([-self.svar])

                    # creating a new selector
                    self.svar = self.vars.id(instance)

                    # MCS enumeration itself
                    self.cld_enumeration(xnum, unit_mcs, use_cld)
                else:
                    self.mhs_mcs_enumeration(xnum, smallest, reduce_)

    def extract_mus(self, reduce_='lin', start_from=None):
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
                    if not self.oracle.solve(assumptions=sorted(to_test)):
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
                    if to_test and not self.oracle.solve(assumptions=to_test):
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

        # technically, this should not be needed as we are given a core already
        if start_from is not None:
            assert self.oracle.solve(assumptions=start_from) == False

        # this is our MUS over-approximation
        core = self.oracle.get_core()
        self.calls = 1  # we have already made one call

        if reduce_ == 'qxp':
            return _do_quickxplain(core)
        else:  # by default, linear MUS extraction is used
            return _do_linear(core)

    def cld_enumeration(self, xnum, unit_mcs, use_cld):
        """
            Compute a subset-minimal contrastive explanation.
        """

        def _overapprox():
            model = self.oracle.get_model()

            for hypo in self.hypos:
                if model[abs(hypo) - 1] == hypo:
                    # soft clauses contain positive literals
                    # so if var is true then the clause is satisfied
                    self.ss_assumps.append(hypo)
                else:
                    self.setd.append(hypo)

        def _compute():
            i = 0
            while i < len(self.setd):
                if use_cld:
                    _do_cld_check(self.setd[i:])
                    i = 0

                if self.setd:
                    # it may be empty after the clause D check

                    self.calls += 1
                    self.ss_assumps.append(self.setd[i])
                    if not self.oracle.solve(assumptions=[self.svar] + self.ss_assumps + self.bb_assumps):
                        self.ss_assumps.pop()
                        self.bb_assumps.append(-self.setd[i])

                i += 1

        def _do_cld_check(cld):
            self.cldid += 1
            sel = self.vars.id('{0}_{1}'.format(self.svar, self.cldid))
            cld.append(-sel)

            # adding clause D
            self.oracle.add_clause(cld)
            self.ss_assumps.append(sel)

            self.setd = []
            st = self.oracle.solve(assumptions=[self.svar] + self.ss_assumps + self.bb_assumps)

            self.ss_assumps.pop()  # removing clause D assumption
            if st == True:
                model = self.oracle.get_model()

                for l in cld[:-1]:
                    # filtering all satisfied literals
                    if model[abs(l) - 1] == l:
                        self.ss_assumps.append(l)
                    else:
                        self.setd.append(l)
            else:
                # clause D is unsatisfiable => all literals are backbones
                self.bb_assumps.extend([-l for l in cld[:-1]])

            # deactivating clause D
            self.oracle.add_clause([-sel])

        # sets of selectors to work with
        self.cldid = 0
        self.expls = []

        # detect and block unit-size MCSes immediately
        if unit_mcs:
            for i, hypo in enumerate(self.hypos):
                if self.oracle.solve(assumptions=[self.svar] + self.hypos[:i] + self.hypos[(i + 1):]):
                    self.expls.append([hypo])
                    if len(self.expls) != xnum:
                        self.oracle.add_clause([-self.svar, hypo])
                    else:
                        break

        self.calls += 1
        while self.oracle.solve(assumptions=[self.svar]):
            self.ss_assumps, self.bb_assumps, self.setd = [], [], []
            _overapprox()
            _compute()

            expl = [-l for l in self.bb_assumps]
            self.expls.append(expl)  # here is a new CXp

            if len(self.expls) == xnum:
                break

            self.oracle.add_clause([-self.svar] + expl)
            self.calls += 1

        self.calls += self.cldid

    def mhs_mus_enumeration(self, xnum, smallest=False):
        """
            Enumerate subset- and cardinality-minimal explanations.
        """

        # result
        self.expls = []

        # just in case, let's save dual (contrastive) explanations
        self.duals = []

        with Hitman(bootstrap_with=[self.hypos], htype='sorted' if smallest else 'lbx') as hitman:

            # computing unit-size MCSes
            for i, hypo in enumerate(self.hypos):
                self.calls += 1
                if self.oracle.solve(assumptions=self.hypos[:i] + self.hypos[(i + 1):]):
                    hitman.hit([hypo])
                    self.duals.append([hypo])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.options.verb > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if self.oracle.solve(assumptions=hset):
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(self.hypos).difference(set(hset)))

                    model = self.oracle.get_model()
                    for h in removed:
                        if model[abs(h) - 1] != h:
                            unsatisfied.append(h)
                        else:
                            hset.append(h)

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        self.calls += 1
                        if self.oracle.solve(assumptions=hset + [h]):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.options.verb > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)

                    self.duals.append([to_hit])
                else:
                    if self.options.verb > 2:
                        print('expl:', hset)

                    self.expls.append(hset)

                    if len(self.expls) != xnum:
                        hitman.block(hset)
                    else:
                        break

    def mhs_mcs_enumeration(self, xnum, smallest=False, reduce_='none', unit_mcs=False):
        """
            Enumerate subset- and cardinality-minimal contrastive explanations.
        """
        # result
        self.expls = []

        # just in case, let's save dual (abductive) explanations
        self.duals = []

        with Hitman(bootstrap_with=[self.hypos], htype='sorted' if smallest else 'lbx') as hitman:
            # computing unit-size MUSes
            for i, hypo in enumerate(self.hypos):
                self.calls += 1

                if not self.oracle.solve(assumptions=[hypo]):
                    hitman.hit([hypo])
                    self.duals.append([hypo])
                elif unit_mcs and self.oracle.solve(assumptions=self.hypos[:i] + self.hypos[(i + 1):]):
                    # this is a unit-size MCS => block immediately
                    self.calls += 1
                    hitman.block([hypo])
                    self.expls.append([hypo])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.options.verb > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if not self.oracle.solve(assumptions=sorted(set(self.hypos).difference(set(hset)))):
                    to_hit = self.oracle.get_core()

                    if len(to_hit) > 1 and reduce_ != 'none':
                        to_hit = self.extract_mus(reduce_=reduce_, start_from=to_hit)

                    self.duals.append(to_hit)

                    if self.options.verb > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    if self.options.verb > 2:
                        print('expl:', hset)

                    self.expls.append(hset)

                    if len(self.expls) != xnum:
                        hitman.block(hset)
                    else:
                        break

    def filter_bg(self):


        """
        
        # Propagation
        
        """

        oracle = Solver(name=self.options.solver, bootstrap_with=self.encoding.knowledge)
        rm = list(map(lambda l: -abs(l), self.hypos))
        onehot = list(filter(lambda l: l not in rm, range(-self.encoding.max_feat_id, 0)))
        for hypo in self.hypos:
            oracle.add_clause([hypo])
        for hypo in onehot:
            oracle.add_clause([hypo])

        assump = list(self.encoding.rvar2rcl.keys())
        st, prop = oracle.propagate(assumptions=assump)
        notuse = []
        while not st:

            unsat_ids = assump.index(prop[-1]) + 1 if len(prop) > 0 else 0
            notuse.append(assump[unsat_ids])

            try:
                assump = assump[unsat_ids + 1 : ]
                st, prop = oracle.propagate(assumptions=assump)
            except:
                st = True

        rvars = set(self.encoding.rvar2rcl.keys()).difference(set(notuse))
        self.knowledge_ = []

        for rvar in rvars:
            self.oracle.add_clause(self.encoding.rvar2rcl[rvar])
            self.knowledge_.append({'cl': self.encoding.rvar2rcl[rvar],
                                   'rule': self.encoding.rvar2rule[rvar]})


    def get_new_rvars(self):
        """

        Get the new variables for knowledge for
        computing rules used

        """
        self.rvar2rule_cl = {}

        for cl_rule in self.knowledge_:
            cl = cl_rule['cl']
            rule = cl_rule['rule']
            rvar = self.vars.id(str(rule))
            self.rvar2rule_cl[rvar] = {'cl': cl,
                                    'rule': rule}

    def extract_usedbg(self, oriexpl):

        if self.oracle_.solve(assumptions=oriexpl) == False:
            return []

        rvars = list(self.rvar2rule_cl.keys())
        assert self.oracle_.solve(assumptions=rvars + oriexpl) == False

        core = self.oracle_.get_core()
        core = set(core).difference(set(oriexpl))
        #print(f'core: {core}')

        core = list(core)

        def _do_linear(core):
            """
                Do linear search.
            """

            def _assump_needed(a):
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

        #print(f'bg: {used_rules}')

        return used_rules

    def validate(self, instance, expl):
        """
            Validating explanation.
        """

        # prepare the oracle for dealing with the current instance
        self.prepare_oracle(instance)

        if self.options.knowledge:
            self.filter_bg()

        if self.options.verb:
            preamble = [str(self.vars.obj(self.hypos[i])) for i in expl]
            print('  validating: "IF {0} THEN {1}"'.format(' AND '.join(preamble), self.label))

        # validating
        res = self.oracle.solve(assumptions=[self.hypos[i] for i in expl]) == False
        minimal = None
        expl_ = None

        if res:
            expl_ = self.extract_mus(start_from=[self.hypos[i] for i in expl])
            minimal = len(expl) == len(expl_)
            preamble = [str(self.vars.obj(i)) for i in expl_]
            expl_ = "IF {0} THEN {1}".format(' AND '.join(preamble), self.label)

        if self.options.verb:
            print('explanation is', 'valid' if res else 'invalid')
            print()

        return res, minimal, expl_


