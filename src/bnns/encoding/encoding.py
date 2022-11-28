"""
CODE REUSES FROM SHAP
"""
from __future__ import print_function
import json
import numpy as np
import xgboost as xgb
import math
import resource
import torch
from pysat.card import *
from pysat.formula import WCNF
import math
import time
from pysat.formula import CNF
from pysat.formula import CNFPlus

from utils import *

from pysat.solvers import Solver  # standard way to import the library
from pysat.solvers import Minisat22, Glucose3  # more direct way
from pysat.formula import IDPool
from pysat.card import *
import operator
class EncodingBNN():
    def __init__(self, config):     
        self.vpool = 0
        self.formula = CNFPlus()
        self.var2ids = {}
        self.vars_by_layers = {}
    def inc_pool(self):
        self.vpool = self.vpool  + 1
        return int(self.vpool)

    def set_pool(self, top):
        self.vpool  = max(top, self.vpool)
        return int(self.vpool)
        
    def format_indexes(self, inds):
        return "_" + "".join(["_{}".format(x) for x in inds])

    def create_indexed_variable_name(self, name, inds):
        x_id = ("{}".format(name))
        x_id = x_id + self.format_indexes(inds)
        #x_id = x_id.replace("-", "n_")
        return x_id

    def get_varid(self, var):
        if var not in self.var2ids:
            self.var2ids[var] = self.inc_pool()
            #print(var, self.var2ids[var])
        return self.var2ids[var]

    def lookup_varid(self, var):
        if var not in self.var2ids:
             print("Requested a variable  {} that does not exist".format(var))
             exit()
        return self.var2ids[var]        
    
    def get_vars_per_layer(self, id_layer):
        vars = [ self.get_varid(x) for x in  self.vars_by_layers[id_layer]]
        #print(vars) 
        return vars
        
    def create_variables_by_layers(self, vars_layers):
        for id_layer,v in enumerate(vars_layers):
            self.vars_by_layers[id_layer] = []
            for k in range(v):
                
                var_name = self.create_indexed_variable_name(LABEL_BIN_VARS, [id_layer, k])
                var_id = self.get_varid(var_name)
                self.vars_by_layers[id_layer].append(var_name)
                #print(var_name, var_id)
    def unary_over_inputs(self, inputs):
        literals = []
        for k in inputs:
            #print(self.create_indexed_variable_name(LABEL_BIN_VARS, [0, k]))
            literals.append(self.get_varid(self.create_indexed_variable_name(LABEL_BIN_VARS, [0, k])))
        
        #print(literals)
        unary_enc_cnf = CardEnc.equals(lits =  literals, bound = 1, top_id = self.vpool)
        cls = np.concatenate(np.asarray(unary_enc_cnf.clauses))        
        #print(cls)
        self.set_pool(max(abs( cls)))        
        #print(self.vpool)
        #print(unary_enc_cnf.clauses)
        #exit()
        for clause in unary_enc_cnf.clauses:
            self.formula.append(clause)
        return literals

    def atmost_one(self, literals):
        unary_enc_cnf = CardEnc.atmost(lits =  literals, bound = 1, top_id = self.vpool)
        cls = np.concatenate(np.asarray(unary_enc_cnf.clauses))        
        self.set_pool(max(abs( cls)))        
        return unary_enc_cnf


    def atmost_k(self, literals, k = 1):
        unary_enc_cnf = CardEnc.atmost(lits =  literals, bound = k, top_id = self.vpool)
        cls = np.concatenate(np.asarray(unary_enc_cnf.clauses))        
        self.set_pool(max(abs( cls)))        
        return unary_enc_cnf


    def atleast_k(self, literals, k = 1):
        unary_enc_cnf = CardEnc.atleast(lits =  literals, bound = k, top_id = self.vpool)
        cls = np.concatenate(np.asarray(unary_enc_cnf.clauses))        
        self.set_pool(max(abs( cls)))        
        return unary_enc_cnf
            
    def append(self, formula):
        for clause in formula.clauses:
            self.formula.append(clause)

    def extend(self, formula):        
        self.formula.extend(formula)

    def solve(self, assumptions = [], name = "g3"):
        s_test = Solver(name=name)
        s_test.append_formula(self.formula)
        #print(self.formula.clauses)
        s_test.solve(assumptions = assumptions)
        solution = s_test.get_model()  
        if (solution is None):
            print("UNSAT")
            return None
                
        print("SOLVED")
        return  solution   
    def solve_formula(self, formula, assumptions = [], name = "g3"):
        #print("name", name)    
        s_test = Solver(name=name)
        s_test.append_formula(formula)
        #print(self.formula.clauses)
        s_test.solve(assumptions = assumptions)
        solution = s_test.get_model()  
        if (solution is None):
            print("UNSAT")
            return None
                
        print("SOLVED")
        return  solution           


    def eq(self, v1, v2):
        formula = CNF()
        formula.append([v1, -v2])
        formula.append([-v1, v2])        
        return formula


    def assign(self, v):
        formula = CNF()
        formula.append([v])
        return formula

    def all_solutions(self, to_enum = 10000, proj = 66):
        s = Solver(name='g3')
        s.append_formula(self.formula.clauses)
#         if (False):
#             computed = 0
#             
#             for i, model in enumerate(s.enum_models(), 1):
#                 print('v {0} 0'.format(' '.join(['{0}{1}'.format('+' if v > 0 else '', v) for v in model])))
#     
#                 computed = i
#                 if i == to_enum:
#                     break
#     
#             print('c nof mods: {0}'.format(computed))
#             print('c acc time: {0:.2f}s'.format(s.time()))
#             print('c avg time: {0:.2f}s'.format(s.time() / computed))
#         
#         
        s.solve()
        solution = s.get_model()
        count = 1
        while(not (solution is None)):
            print(count, solution[:proj])
            #print("block", [-x for x in solution[:66]])
            s.add_clause([-x for x in solution[:proj]])
            s.solve()
            solution = s.get_model()
            #print(solution)
            count = count + 1
            #exit()
        
    
    def atleast(self, constraint): 
        # TODO
        assert(False)
#         x = constraint["literals"]
#         k = int(constraint["rhs_constant"])        
#         
#         top_id = max(self.var2ids.values())
#         return CardEnc.equals(lits=x, bound = k, top_id = top_id) 
    def seqcounters_reified(self, constraint, layer_id, cons_id,  assumptions = [], testing = True, is_profile = False): #dist_output, varids2cnf_varids, coeffs, vars_ids, output_var_id, constterm, constraint_prefix):
        #print(">>>>>>>>>>>>>>>>>>> SQ <<<<<<<<<<<<<<<<<<<")
        #print("input variables:")
        #print(vars_ids)
        start = time.time()
        
        card_formula = CNF()
        var_prefix_local = "{}_{}_{}".format(LABEL_SEQ_VARS, layer_id, cons_id)
        # sum_i=0^n-1 l_i >= C <=> b
        n = len(constraint["literals"])
        verb = False
        if (verb):
            print("nb_vars", n)
      
    
        
        nb_assumptions = len(assumptions)
        
        x = constraint["literals"]
        k = int(constraint["rhs_constant"])
        rhs_y = constraint["rhs_var"]
    
        if (verb):
            print("literals", x)
            print("rhs_constant", k)
            print("rhs_var", rhs_y)
            print("n", n)
                
        if (k > n):
            #print("----> Trivial UNSAT constraint as rhs ({}) > nb vars = {} ".format(k, n))
            card_formula.append([-rhs_y])
            return card_formula, [0] * nb_assumptions     
            
        if (k <= 0):
            # constraint is trivially satisfied 
            #print("----> Trivial SAT constraint as rhs ({}) < 0".format(k))
            card_formula.append([rhs_y])
            return card_formula, [1] * nb_assumptions   
    
        if (n == 1):
            # special case
            # we only have 1 var
            # lit >= k <=> b
                    
            # if k > 1 then constraint is unsat
            if (k > 1):
                #print("---->  Trivial UNSAT constraint as rhs ({}) > nb vars = {}".format(k, n))
                card_formula.append([-rhs_y])
                return card_formula, [0]* nb_assumptions   
            else:
            # otherwsie lit <=>  b    
                #print("---->  Trivial case  of equvalnce  -- rhs ({}) ,  nb vars = {}".format(k, n))
                formula = self.eq(x[0], rhs_y)
                for c in formula.clauses:
                    card_formula.append(c)            
                return card_formula, [x[0] > 0]  * nb_assumptions 
            
        s = [[-1] * (k+1) for i in range(n+1)]
        # note that seq counters assume that we have variables from 1..n
        
        ################################################################
        # s[1][1] <=>  x[0]
        # (not x[0] V  s[1][1])
        # (x[0] V  not s[1][1])     
                
        # Intro s[1][1]
        s[1][1] = self.get_varid(self.create_indexed_variable_name(var_prefix_local, [1, 1]))
        # (not x[0] V  s[1][1])        
        card_formula.append([-x[0], s[1][1]])
        # (x[0] V  not s[1][1])     
        card_formula.append([x[0], -s[1][1]])

        # 1 < j <=k
        for j in range(2, k+1):
            #(not s[1][j])
            s[1][j] = self.get_varid(self.create_indexed_variable_name(var_prefix_local, [1, j]))
            card_formula.append([-s[1][j]])
        #print(card_formula.clauses)

        #print("after first block s = ", s)
        # 1 < i <= n    
        for i in range(2, n+1):         
            s[i][1] = self.get_varid(self.create_indexed_variable_name(var_prefix_local,[ i, 1]))
            
            #s[i][1] <=> x[i-1] \/ s[i-1][1]
            #s[i][1] \/ not x[i-1] 
            #s[i][1] \/ not s[i-1][1]
            #not s[i][1] \/ x[i-1] \/ s[i-1][1]
            
            # (s[i][1] \/ not x[i-1] )        
            #add_clause(dist_output, varids2cnf_varids, ["-" + x[i-1], s[i][1]])
            card_formula.append([-x[i-1], s[i][1]])
    
            # (s[i][1] \/ not s[i-1][1] )                
            #add_clause(dist_output, varids2cnf_varids, ["-" + s[i-1][1], s[i][1]])
            card_formula.append([-s[i-1][1], s[i][1]])
            
            # not s[i][1] \/ x[i-1] \/ s[i-1][1]     
            #add_clause(dist_output, varids2cnf_varids, ["-" + s[i][1], x[i-1], s[i-1][1]])
            card_formula.append([-s[i][1], x[i-1], s[i-1][1]])
        end  = time.time()
        if(is_profile):
            print("           (init part) seqcounters_reified :", end - start)  
 
        start = time.time()

        for j in range(2, k+1):
            for i in range(j, n+1): 
            # 1 < j <=k
                s[i][j] =  self.get_varid(self.create_indexed_variable_name(var_prefix_local, [i, j]))
                #print(i,j, s[i][j])
                
                if (i == j):
                    # corner case, e.g. we are at s_3,3 and look s_2,3 which must be false 
                    s[i-1][j] = self.get_varid(self.create_indexed_variable_name(var_prefix_local, [i-1, j]))
                    #add_clause(dist_output, varids2cnf_varids, ["-" + s[i-1][j]])
                    card_formula.append([-s[i-1][j]])
                    
            
                #s[i][j] <=> (x[i-1] /\ s[i-1][j-1]) \/ s[i-1][j]
    
                ##############################################
                #s[i][j] <= x[i-1] /\ s[i-1][j-1] \/ s[i-1][j]
                ###############################################
                #s[i][j] \/ not x[i-1] \/ not s[i-1][j-1] 
                #s[i][j] \/ not s[i-1][j]
                #not s[i][j] \/(x[i-1] /\ s[i-1][j-1] \/ s[i-1][j])
    
                #s[i][j] => x[i-1] /\ s[i-1][j-1] \/ s[i-1][j]
                # not s[i][j] \/ x[i-1]  \/ s[i-1][j]
                # not s[i][j] \/ s[i-1][j-1] \/ s[i-1][j]
                
                
                
                #s[i][j] \/ not x[i-1] \/ not s[i-1][j-1]                
                #add_clause(dist_output, varids2cnf_varids, [s[i][j], "-" + x[i-1], "-" + s[i-1][j-1]])
                card_formula.append([s[i][j], -x[i-1], -s[i-1][j-1]])
                
                #s[i][j] \/ not s[i-1][j]                
                #print(i-1,j, s[i-1][j])
                #add_clause(dist_output, varids2cnf_varids, [s[i][j], "-" + s[i-1][j]])
                card_formula.append([s[i][j], -s[i-1][j]])
    
                #not s[i][j] \/ x[i-1]  \/ s[i-1][j]   
                #add_clause(dist_output, varids2cnf_varids, ["-" + s[i][j], x[i-1],  s[i-1][j]])
                card_formula.append([- s[i][j], x[i-1],  s[i-1][j]])
    
                #not s[i][j] \/ s[i-1][j-1] \/ s[i-1][j]
                #add_clause(dist_output, varids2cnf_varids, ["-" + s[i][j], s[i-1][j-1],  s[i-1][j]])
                card_formula.append([-s[i][j], s[i-1][j-1],  s[i-1][j]])
                
        end  = time.time()
        if(is_profile):
            print("           (main part) seqcounters_reified :", end - start)  
   
        start  = time.time()
        
        # add reification constraint 
        # s[n][k] = 1 <==> output = 1    
        #print(s[n][k], output_var_id)
        formula = self.eq(s[n][k], rhs_y)
        for c in formula.clauses:
            card_formula.append(c)

        end  = time.time()
        if(is_profile):
            print("           (append part) seqcounters_reified :", end - start)  
            
        #print(card_formula.clauses)
        # testing 
        
        #print(card_formula.clauses)
     

        rhs_y_res = []
        if (testing):
            s_test = Solver(name='g3')
            s_test.append_formula(card_formula.clauses)
            if not (assumptions is None):
                for assump in assumptions:
                    s_test.solve(assumptions=assump)
                    solution = s_test.get_model()
                    rhs_y_res.append(solution[rhs_y-1] > 0)
            else:   
                for h in range(100):
                    assump = np.random.randint(2, size=n)    
                    assump_lit = []
                    #print(assump_lit)
                    # compute sum 
                    sum = 0
                    for i, a in enumerate(assump): 
                        if (a == 1):          
                            assump_lit.append(abs(x[i]))
                        else:
                            assump_lit.append(-abs(x[i]))
            
                        if (x[i] > 0) and (a == 1):
                            sum += 1
                        if (x[i] > 0) and (a == 0):
                            sum += 0
                        if (x[i] < 0) and (a == 1):
                            sum += 0
                        if (x[i] < 0) and (a == 0):
                            sum += 1
                    y_value_should_be =  sum >= k
                    #print("sum", sum, " k ", k)    
                    #print(assump_lit)
                    s_test.solve(assumptions=assump_lit)
                    solution = s_test.get_model()
                    #print(solution)
                    #print(self.var2ids)
                    #print(s[n][k], solution[s[n][k]-1])
                    #print(rhs_y,  solution[rhs_y-1])
                    if (y_value_should_be):
                        assert(solution[rhs_y-1] > 0)
                    else:
                        assert(solution[rhs_y-1] < 0)
                    rhs_y_res.append(solution[rhs_y-1] > 0)
        
        if (len(rhs_y_res) > 0):
            return card_formula, np.array(rhs_y_res).astype(int)
        else:
            return card_formula, None

        #print("final s = ", s)
            
                
    