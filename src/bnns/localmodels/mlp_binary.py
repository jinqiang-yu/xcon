import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .binarized_modules import  BinarizeLinear,  Binarize
import numpy as np
from utils import *
from .functions import *
from pysat.card import *
from pysat.solvers import Solver
REMOVED_TERM =  -2
import time

test_formula = CNFPlus()
from pysat.solvers import Solver  # standard way to import the library

__all__ = ['mlp_binary']
class MLPNetOWT_BN(nn.Module):
    def __init__(self, config):        
        super(MLPNetOWT_BN, self).__init__()
        self.config = config
        self.side = config["data"]["input_size"]
        self.is_simulation = False
        self.is_encode = False
        self.is_profile = False
        self.testing =  False
        self.card_encoding = CARD_ENC_SEQ_COUNT
        #binarize = True
        #self.BinarizeLinear_no_binarization =  BinarizeLinear(self.side, self.side, bias=False)
        #self.BinarizeLinear_no_binarization.binarize = False
        
        self.layer0 = nn.Sequential(           
            BinarizeLinear(self.side, config["model"]["layers"][0]),
            nn.BatchNorm1d(config["model"]["layers"][0]),
            nn.Hardtanh(inplace=True)
            )          
                    
        self.layer1 = nn.Sequential(
            BinarizeLinear(config["model"]["layers"][0],  config["model"]["layers"][1]),
            nn.BatchNorm1d(config["model"]["layers"][1]),
            nn.Hardtanh(inplace=True))
        self.layer2 = nn.Sequential(
            BinarizeLinear(config["model"]["layers"][1], config["model"]["layers"][2]),
            nn.BatchNorm1d(config["model"]["layers"][2]),
            nn.Hardtanh(inplace=True))
      
        self.layer3 = nn.Sequential(
                BinarizeLinear(config["model"]["layers"][2], config["model"]["layers"][3]),
                )


        self.layer0[0].small_weight = config["train"]["small_weight"]
        self.layer1[0].small_weight = config["train"]["small_weight"]
        self.layer2[0].small_weight = config["train"]["small_weight"]
        self.layer3[0].small_weight = config["train"]["small_weight"]/10

        self.layer0[0].is_simulation = self.is_simulation
        self.layer1[0].is_simulation = self.is_simulation
        self.layer2[0].is_simulation = self.is_simulation
        self.layer3[0].is_simulation = self.is_simulation
        
        self.all_layers = [
                            [self.layer0,  BinLin_BN_REIF],
                            [self.layer1,  BinLin_BN_REIF],
                            [self.layer2,  BinLin_BN_REIF],
                            [self.layer3,  BinLin_NOBN],
                            #self.layer4,
                            #self.bfc
                            ]

        lr_moving  = config["train"]["lr"]    
        self.regime = {
            0: {'optimizer': config["train"]["optimizer"], 'lr': lr_moving,  'betas': (0.9, 0.999)},
            #15: {'lr': lr_moving/2},
            20: {'lr': lr_moving/10},
            #40: {'lr': lr_moving/20},
            50: {'lr': lr_moving/100},
            75: {'lr': lr_moving/1000}
        }
        #print(self.regime)
        ##########################################3
        self.encoder = None
        #self.var2ids = {}
    def get_number_neurons(self):
        nb_neurons = []
        for layer, type in self.all_layers:
            nb_neurons_layer  =  layer[0].weight.shape[1]
            nb_neurons.append(nb_neurons_layer)
            if(type == BinLin_NOBN):
                nb_neurons.append(layer[0].weight.shape[0])
        return nb_neurons
        
#     def format_indexes(self, inds):
#         return "_" + "".join(["[{}]".format(x) for x in inds])
# 
#     def create_indexed_variable_name(self, name, inds):
#         x_id = ("{}".format(name))
#         x_id = x_id + self.format_indexes(inds)
#         x_id = x_id.replace("-", "n_")
#         return x_id
# 
#     def get_varid(self, var):
#         if var not in self.var2ids:
#             self.var2ids[var] = self.max_id
#             self.max_id += 1
#         return self.var2ids[var]
# 
#     def lookup_varid(self, var):
#         if var not in self.var2ids:
#              print("Requested a variable  {} that does not exist".format(var))
#              exit()
#         return var, self.var2ids[var]
        
    def forward(self, x):
        out = x.view(-1, self.side)
        forward_outputs = []
        forward_outputs_extra = []
        
        for layer, type in self.all_layers:
            #print(out.shape)
            #print(layer)
            out = layer(out)
            if (self.is_simulation  or self.is_encode):
                forward_outputs.append(out.clone().cpu().detach().numpy())
                #if (self.is_simulation): 
                #    forward_outputs_extra.append({"bininput": layer[0].bin_input.clone().cpu().detach().numpy(), "ax": layer[0].ax.clone().cpu().detach().numpy(), "ax_b":layer[0].ax_b.clone().cpu().detach().numpy()})
        if (self.is_simulation):
            self.forward_simulator(x, forward_outputs, forward_outputs_extra)
        #if (self.is_encode):
        #    self.forward_encoder(x, forward_outputs, forward_outputs_extra, winner)
        
        
        return out #self.logsoftmax(out)

    def forward_encode(self, x, switch = None, winner = None):
        #print(x)
        out = x.view(-1, self.side)
        forward_outputs = []
        forward_outputs_extra = []
        start = time.time()       
        for layer, type in self.all_layers:
            #print(out.shape)
            #print(layer)
            out = layer(out)
            #print(out)
            forward_outputs.append(out.clone().cpu().detach().numpy())
        end = time.time()
        if(self.is_profile):
            print("    forward :", end - start)

        start = time.time()               
        outputs_by_layers = self.forward_encoder(x, forward_outputs, forward_outputs_extra, winner)                
        end = time.time()
        if(self.is_profile):
            print("    forward_encoder :", end - start)
        return outputs_by_layers #self.logsoftmax(out)


    def forward_simulator_block_binlin_bn_ht(self, layer, x, is_ht = True):
        # take sign
        x_sign = x.copy()
        x_sign[ x_sign > 0] = 1
        x_sign[ x_sign <= 0] = -1
        
        # linear layer simulation
        bin_lin_layer  = layer[0]
        bin_weigths = binarize(bin_lin_layer.weight).cpu().detach().numpy()
        #print(bin_weigths.shape, x_sign.shape)
        y = []
        for x_s in x_sign:
            #print(bin_weigths.shape, x_s.shape)
            ax =  np.dot(bin_weigths, x_s)
            if not bin_lin_layer.bias is None:
                ax_b =  ax + bin_lin_layer.bias.cpu().detach().numpy()
            else:
                ax_b = ax
            y.append(ax_b)
        y = np.asarray(y)        
        #print(y.shape)
        
        # BN
        #\frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
        bn_layer = layer[1]
        running_mean = bn_layer.running_mean.cpu().detach().numpy()
        running_var = bn_layer.running_var.cpu().detach().numpy()        
        weights = bn_layer.weight.cpu().detach().numpy()
        bias =  bn_layer.bias.cpu().detach().numpy()
        eps = np.asarray(bn_layer.eps)
         

        
        y_t = y.copy()
        runstd = np.sqrt(running_var +  eps ) 
        invstd = (1 / runstd)
        bn_y = ((y_t - running_mean) * invstd) * weights + bias
                
        if (is_ht):            
            ht_y = bn_y.copy()
            # hardtanh
            ht_y[ ht_y > 1] = 1
            ht_y[ ht_y < -1] = -1
        else:
            ht_y = None

        return [x_sign, y, bn_y, ht_y, None]
  

    def forward_simulator_block_binlin_bn_ht_transform_four(self, layer, x, is_ht = True):
        #################################
        # move to litterals
        # WE ASSUME THAT INPUT ARE BOOLEAN
        ################################
        
        # take sign
        x_sign = x.copy()
        x_sign[ x_sign > 0] = 1
        x_sign[ x_sign <= 0] = -1
        
        # linear layer + BN simulation
        #A x + b
        bin_lin_layer  = layer[0]
        A = binarize(bin_lin_layer.weight).cpu().detach().numpy()
        if not bin_lin_layer.bias is None:            
            b = bin_lin_layer.bias.cpu().detach().numpy()
        else:
            b =  bin_lin_layer.bias.cpu().detach().numpy()
            b.fill(0)
            
        # BN
        #\frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
        bn_layer = layer[1]
        running_mean = bn_layer.running_mean.cpu().detach().numpy()
        running_var = bn_layer.running_var.cpu().detach().numpy()        
        gamma = bn_layer.weight.cpu().detach().numpy()
        beta =  bn_layer.bias.cpu().detach().numpy()
        eps = np.asarray(bn_layer.eps)        
        runstd = np.sqrt(running_var +  eps ) 
        invstd = (1 / runstd)        
        
        # ((A x +  b) - running_mean)*invstd*gamma + beta
        b_m_running_mean = b - running_mean
        invstd_times_gamma = invstd*gamma        
        # ((A x +  b_running_mean)*invstd_gamma + beta
        b_m_running_mean_times_invstd_times_gamma =  b_m_running_mean*invstd_times_gamma
        # invstd_gamma* A x +  b_running_mean_invstd_gamma + beta
        b_m_running_mean_times_invstd_times_gamma_p_beta = b_m_running_mean_times_invstd_times_gamma + beta
        # invstd_gamma* A x +  b_running_mean_invstd_gamma_beta
        
        #print(bin_weigths.shape, x_sign.shape)
        y = []
        for i, x_s in enumerate(x_sign):
            #print(bin_weigths.shape, x_s.shape)
            trans_x =  np.dot(A, x_s)*invstd_times_gamma + b_m_running_mean_times_invstd_times_gamma_p_beta
            y.append(trans_x)
            #if (i == 0):
            #    print(trans_x)
        y = np.asarray(y)        
        #print(y.shape)
        
        ax_b_bn_t = y.copy()                
        if (is_ht):            
            ht_y = ax_b_bn_t.copy()
            # hardtanh
            ht_y[ ht_y > 1] = 1
            ht_y[ ht_y < -1] = -1

            o_sign_temp = ht_y.copy()            
            o_sign_temp[ ht_y > 0] = 1
            o_sign_temp[ ht_y <= 0] = -1
            o_sign_temp = o_sign_temp.astype(int)
            
            # moving toward cardinalities             
            trans_y = []
            #remove sign of invstd_gamma
            for j, v in enumerate(invstd_times_gamma):     
                #print(A[j,:], invstd_times_gamma[j])           
                if (invstd_times_gamma[j] < 0):
                    A[j,:] =  A[j,:]*(-1)
                    invstd_times_gamma[j] = invstd_times_gamma[j]*(-1)   
            
                #trans_x_temp =  np.dot(A, x_sign[0])*invstd_times_gamma + b_m_running_mean_times_invstd_times_gamma_p_beta
                #print(trans_x_temp)
                #exit()
            
            for i, x_s in enumerate(x):
                # bin x
                
                x_s_bin = x_s.copy().astype(int)
                x_s_bin_neg = 1 - x_s_bin
                #x_s_bin[x_s_bin == -1] = 0
                x_s_ones = x_s.copy()
                x_s_ones.fill(1)

                
                #invstd_gamma* A x +  b_running_mean_invstd_gamma_beta > 0
                #invstd_gamma* A x  >  -b_running_mean_invstd_gamma_beta
                minus_b_m_running_mean_times_invstd_times_gamma_p_beta = -b_m_running_mean_times_invstd_times_gamma_p_beta
                #A (2*x_bin-I)  >  -b_running_mean_invstd_gamma_beta/invstd_gamma
                #A x_bin  >  (-b_running_mean_invstd_gamma_beta/invstd_gamma + A*I)/2
                minus_b_m_running_mean_times_invstd_times_gamma_p_beta_div_all_p_AI = (minus_b_m_running_mean_times_invstd_times_gamma_p_beta/invstd_times_gamma + np.dot(A, x_s_ones) )/2
                
                
                x_new = []
                for j, row in enumerate(A):
                    # if a[i,j] =  - 1  then we replave -x_j with neg x_j - 1
                    x_row = x_s_bin.copy()
                    x_row[row == -1] = x_s_bin_neg[row == -1]
                    num_m_one = sum(row == -1)
                    minus_b_m_running_mean_times_invstd_times_gamma_p_beta_div_all_p_AI[j] += num_m_one
                    x_new.append(x_row)
                
                x_new = np.asarray(x_new)
                #print(x_new)  
                #print(np.sum(x_new, axis=1))
                #print(minus_b_m_running_mean_times_invstd_times_gamma_p_beta_div_all_p_AI)
                #exit()
                rhs = np.ceil(minus_b_m_running_mean_times_invstd_times_gamma_p_beta_div_all_p_AI + EPSILON)            
                lhs =  np.sum(x_new, axis=1)
                if (i == 0) and False:
                    print(minus_b_m_running_mean_times_invstd_times_gamma_p_beta)
                    print(invstd_times_gamma)
                    print(rhs)
                    print(lhs)
                

                trans_x =  np.asarray(lhs >= rhs)
                #print(lhs.shape)
                #print(rhs.shape)
                #print(trans_y.shape)
                #print(trans_y)                
                trans_y.append(trans_x)
            trans_y = np.asarray(trans_y)   
                    
            o_sign = trans_y.copy().astype(int)
            o_sign_check = o_sign.copy()
            o_sign_check[ o_sign_check <=0] = -1
            #o_sign[ o_sign is False] = -1

            for i,_ in enumerate(o_sign_check):
                if (i == 0)  and False:                    
                    print(trans_y[i])
                    print(ht_y[i])
                    print(o_sign_check[i]) 
                    print(o_sign_temp[i])
                assert(np.allclose(o_sign_check[i], o_sign_temp[i]))
            #exit()
            #assert(np.allclose(o_sign, o_sign_temp))

            
        else:
            ht_y = None
            o_sign = None
            
       

        return [x_sign, None, ax_b_bn_t, ht_y, o_sign]
        
    def forward_simulator(self, x, forward_outputs, forward_outputs_extra = None):
        out = x.view(-1, self.side).clone().cpu().detach().numpy()
        for i, (layer, type) in enumerate(self.all_layers):            
            #print(layer, type, forward_outputs[i].shape)
            if (type == BinLin_BN_REIF):
                #print("Block " , BinLin_BN_REIF)
                simulated_output  = self.forward_simulator_block_binlin_bn_ht_transform_four(layer, out)
                
                # LIN
                sim_ax_b = simulated_output[1]            
                #ax_b = forward_outputs_extra[i]["ax_b"]
                #assert(np.allclose(sim_ax_b, ax_b))
                
                # BN                
                sim_bn_y = simulated_output[2]
                
                # HT          
                sim_ht_y = simulated_output[3]      
                ht_y = forward_outputs[i]

                #print(sim_ht_y.shape, ht_y.shape)
                #print(sim_ht_y[0], ht_y[0])
                assert(np.allclose(sim_ht_y, ht_y))
                if (simulated_output[4] is None):
                    out = ht_y
                else:
                    out = simulated_output[4]
                #exit()                            
            if (type == BinLin_NOBN):
                #print("Block " , BinLin_NOBN)
                simulated_output  = self.forward_simulator_block_binlin_bn_ht_transform_four(layer, out, is_ht = False)
                sim_ax_b = simulated_output[1]            
                #ax_b = forward_outputs_extra[i]["ax_b"]
                #assert(np.allclose(sim_ax_b, ax_b))
                
                # BN                
                sim_bn_y = simulated_output[2]
                bn_y = forward_outputs[i]
                assert(np.allclose(sim_bn_y, bn_y))
            #exit()

        print("Simulation is OK")
        return      
    
    
    def forward_simulator_block_binlin_bn_ht_encoding(self, id_layer, layer, x, is_ht = True, winner = None):
        #################################
        # move to litterals
        # WE ASSUME THAT INPUT ARE BOOLEAN
        ################################
        
        start = time.time()

        # take sign
        x_sign = x.copy()
        x_sign[ x_sign > 0] = 1
        x_sign[ x_sign <= 0] = -1
        
        # linear layer + BN simulation
        #A x + b
        bin_lin_layer  = layer[0]
        A = Binarize(bin_lin_layer.weight.data, small_weight = bin_lin_layer.small_weight).cpu().detach().numpy()
        if not bin_lin_layer.bias is None:            
            b = bin_lin_layer.bias.cpu().detach().numpy()
        else:
            b =  bin_lin_layer.bias.cpu().detach().numpy()
            b.fill(0)
            
        # BN
        #\frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
        bn_layer = layer[1]
        running_mean = bn_layer.running_mean.cpu().detach().numpy()
        running_var = bn_layer.running_var.cpu().detach().numpy()        
        gamma = bn_layer.weight.cpu().detach().numpy()
        beta =  bn_layer.bias.cpu().detach().numpy()
        eps = np.asarray(bn_layer.eps)        
        runstd = np.sqrt(running_var +  eps ) 
        invstd = (1 / runstd)        
        
        # ((A x +  b) - running_mean)*invstd*gamma + beta
        b_m_running_mean = b - running_mean
        invstd_times_gamma = invstd*gamma        
        # ((A x +  b_running_mean)*invstd_gamma + beta
        b_m_running_mean_times_invstd_times_gamma =  b_m_running_mean*invstd_times_gamma
        # invstd_gamma* A x +  b_running_mean_invstd_gamma + beta
        b_m_running_mean_times_invstd_times_gamma_p_beta = b_m_running_mean_times_invstd_times_gamma + beta
        # invstd_gamma* A x +  b_running_mean_invstd_gamma_beta
        
        #print(bin_weigths.shape, x_sign.shape)
        y = []
        for i, x_s in enumerate(x_sign):
            #print(bin_weigths.shape, x_s.shape)
            trans_x =  np.dot(A, x_s)*invstd_times_gamma + b_m_running_mean_times_invstd_times_gamma_p_beta
            y.append(trans_x)
            #if (i == 0):
            #    print(trans_x)
        y = np.asarray(y)        
        #print(y.shape)
        
        ax_b_bn_t = y.copy() 
        #print(ax_b_bn_t)
        
        #exit()
             # moving toward cardinalities             
        #remove sign of invstd_gamma
        for j, v in enumerate(invstd_times_gamma):     
            #print(A[j,:], invstd_times_gamma[j])           
            if (invstd_times_gamma[j] < 0):
                A[j,:] =  A[j,:]*(-1)
                invstd_times_gamma[j] = invstd_times_gamma[j]*(-1)   
        
        #invstd_gamma* A x +  b_running_mean_invstd_gamma_beta > 0
        #invstd_gamma* A x  >  -b_running_mean_invstd_gamma_beta
        minus_b_m_running_mean_times_invstd_times_gamma_p_beta = -b_m_running_mean_times_invstd_times_gamma_p_beta
        #A (2*x_bin-I)  >  -b_running_mean_invstd_gamma_beta/invstd_gamma
        #A x_bin  >  (-b_running_mean_invstd_gamma_beta/invstd_gamma + A*I)/2
        
        x_s_ones = x[0].copy()
        x_s_ones.fill(1)
        
        x_s_zeros = x[0].copy()
        x_s_zeros.fill(0)

        x_s_removed = x[0].copy()
        x_s_removed.fill(REMOVED_TERM)
                         
        minus_b_m_running_mean_times_invstd_times_gamma_p_beta_div_all_p_AI =  \
         (minus_b_m_running_mean_times_invstd_times_gamma_p_beta/invstd_times_gamma + np.dot(A, x_s_ones) )/2

        for j, row in enumerate(A):
            num_m_one = sum(row == -1)
            minus_b_m_running_mean_times_invstd_times_gamma_p_beta_div_all_p_AI[j] += num_m_one

        x_lits = []
        for j, row in enumerate(A):
              x_lit = x_s_ones.copy()
              x_lit[row == -1] = x_s_zeros[row == -1]
              x_lit[row == 0] = x_s_removed[row == 0]
              x_lits.append(x_lit)
        
        x_lits = np.asarray(x_lits)

        # generate constraints 
        rhs_constraints =  minus_b_m_running_mean_times_invstd_times_gamma_p_beta_div_all_p_AI 
        lhs_constraints = x_lits.copy() 

        #########################################
        assumptions = []
        for i, x_s in enumerate(x):            
            # bin x            
            x_s_bin = x_s.copy().astype(int)
            assump = []
            for k, v in enumerate(x_s_bin):
                var_name = self.encoder.create_indexed_variable_name(LABEL_BIN_VARS, [id_layer, k])
                var_id = self.encoder.lookup_varid(var_name)                
                if (v > 0):
                    assump.append(var_id)
                else:
                    assump.append(-var_id)                
            assumptions.append(assump)    
        ########################################################
            
        end = time.time()
        if(self.is_profile):
            print("       (init part) forward_simulator_block_binlin_bn_ht_encoding :", end - start)  
        rhs_y_signs = []
        time_create_vars = 0
        time_create_cons = 0
        time_add_cons = 0
        for j, rhs in enumerate(rhs_constraints):                   
            con_reif = {}
            con_reif["literals"] = []
            start = time.time()
            for k, l in enumerate(lhs_constraints[j]):
                # lit
                var_name = self.encoder.create_indexed_variable_name(LABEL_BIN_VARS, [id_layer, k])
                var_id = self.encoder.lookup_varid(var_name)
                if (l == REMOVED_TERM):
                    continue
                if (l == 0):
                    var_id = -var_id
                con_reif["literals"].append(var_id)
        
            ## RHS var         
            rhs_var_name = self.encoder.create_indexed_variable_name(LABEL_BIN_VARS, [id_layer+1, j])
            rhs_var_id = self.encoder.lookup_varid(rhs_var_name)
            con_reif["rhs_var"] = rhs_var_id
            
            end = time.time()
            time_create_vars = time_create_vars +end - start
              
            ## Constant                 
            start = time.time()
            con_reif["rhs_constant"] = np.ceil(rhs +  EPSILON)
            #print( con_reif)
            
            if self.card_encoding == CARD_ENC_NAIVE:
                ## greater or equal the rhs_y_res is 1
                # at most rhs - 1
                
                # not y -> sum(l_i) <= rhs -1
                # y     -> sum(l_i) >= rhs

                # l = len(l_i)
                # sum(l_i)  <= rhs -1 + (l - (rhs -1)) y
                # sum(l_i)  - (l - (rhs -1)) (y)  <= rhs -1
                # sum(l_i)  + (l - (rhs -1)) (not y - 1)  <= rhs -1 
                # sum(l_i)  + (l - (rhs -1)) not y  <= rhs -1 + (l - (rhs -1))
                # sum(l_i)  + (l - (rhs -1)) not y  <=  l
                # 9 + (18 - 8) <= 18 
                deb = 69
                nb_lit = len(con_reif["literals"])                
                rhs_atmost = int(con_reif["rhs_constant"] - 1)
                extra_lits_atmost = [-con_reif["rhs_var"]] * (nb_lit - rhs_atmost)                
                bound_atmost = nb_lit       
                
#                 if (con_reif["rhs_var"] == deb):         
#                 
#                     print(con_reif["rhs_constant"])
#                     print("nb_lit", nb_lit)
#                     print("rhs_atmost", rhs_atmost)
#                     print(con_reif["literals"] + extra_lits_atmost)
#                     print(bound_atmost)
                cnf_atmost = CardEnc.atmost(lits=con_reif["literals"] + extra_lits_atmost , bound= bound_atmost, encoding=EncType.native)
               
                #print(cnf_atmost)
                #for atm in cnf_atmost.atmosts:
                #    print(atm)
                #exit()
                self.encoder.extend(cnf_atmost)

                # y     -> sum(l_i) >= rhs
                # sum(l_i)  + rhs not y >= rhs
                rhs_atleast = int(con_reif["rhs_constant"])
                extra_lits_atleast = [-con_reif["rhs_var"]] * rhs_atleast
                bound_atleast = rhs_atleast                
#                 if (con_reif["rhs_var"] == deb):             
#                     print(con_reif["literals"] + extra_lits_atleast)
#                     print(bound_atleast)
                cnf_atleast = CardEnc.atleast(lits=con_reif["literals"] + extra_lits_atleast , bound= bound_atleast, encoding=EncType.native)
                self.encoder.extend(cnf_atleast)
#                 if (con_reif["rhs_var"] == deb) and False:    
#                     cnf1 = CNFPlus()
#                     cnf1.extend([[-1, 2], [1]])
#                     cnf1.append([[1, 2], 1], is_atmost=True)
#                     cnf2 = cnf1.copy()
#                     print(cnf1)
#                     print(cnf2)
#                     cnf1.to_file("cnf1.cnf")
#                     cnf2.to_file("cnf2.cnf")
#                     exit()
#                     
#                     test_formula.extend(cnf_atmost)
#                     #test_formula.extend(cnf_atleast)  
#                     testvars = list(abs(np.asarray([-1, -3, -4, 6, -12, 20, 26, -28, -32, -37, -38, 41, 47, -52, 53, -59, 62, 63, -69, -69, -69, -69, -69, -69, -69, -69, -69, -69])))
# 
#                     prob_sol = [-1, -3, -4, -6, -12, -20, -26, -28, -32, -37, -38, -41, -47, 52, -53, -59, -62, -63, -69]
#                     for a in prob_sol:
#                        test_formula.append([a]) 
#                  
#                  
#                     s_test = Solver(name='minicard')
#                     s_test.append_formula(test_formula)
#                     test_formula.to_file("test.cnf")
#                     test_formula2 = test_formula.copy()                 
#                     test_formula2.to_file("test02.cnf")
#                     print(test_formula)
#                     print(test_formula2)
#                     print(test_formula2.atmosts)
#                     #print(self.formula.clauses)
#                     print(assumptions)
#                     print(testvars)
#                     s_test.solve()
#                     solution = s_test.get_model()  
#                     if (solution is None):
#                         print("UNSAT")
#                         return None
#                     for q, s in enumerate(solution):
#                         if (q+1 in testvars):
#                             print(solution[q])
#                         
#                         
#                     
#                     print("SOLVED")
#                     exit()
#                 
#                                     
                
                rhs_y_res = con_reif["rhs_var"] 
                #exit()
            else:
                card_formula, rhs_y_res = self.encoder.seqcounters_reified(con_reif, id_layer, j, assumptions, testing = self.testing, is_profile= False)
                end = time.time()
                time_create_cons =  time_create_cons+ end - start
                start = time.time()
                self.encoder.append(card_formula)
                end = time.time()
                time_add_cons = time_add_cons + end - start            

            rhs_y_signs.append(rhs_y_res)

            #self.encoder.solve()
                #exit()


        if(self.is_profile):
            print("       (main part, create vars) forward_simulator_block_binlin_bn_ht_encoding :", time_create_vars)  
            print("       (main part, create cons) forward_simulator_block_binlin_bn_ht_encoding :", time_create_cons)
            print("       (main part, add cons) forward_simulator_block_binlin_bn_ht_encoding :", time_add_cons)


        start = time.time()
       
        #print(rhs_y_signs)
        trans_y = []
        lhs_all = []

        for i, x_s in enumerate(x):            
            # bin x            
            x_s_bin = x_s.copy().astype(int)
            x_s_bin_neg = 1 - x_s_bin
            #print(x_s_bin)
            #print(x_s_bin_neg)
            
            #x_s_bin[x_s_bin == -1] = 0
            x_new = []
            for j, row in enumerate(A):
                # if a[i,j] =  - 1  then we replave -x_j with neg x_j - 1  
                x_row = x_s_bin.copy()                
                x_row[x_lits[j] == 0] = x_s_bin_neg[x_lits[j] == 0]                
                x_row[row == 0] = x_s_zeros[row == 0]                

                x_new.append(x_row)

            x_new = np.asarray(x_new)
            #if( i == 0):
            #    print(x_new)  
            #    print(np.sum(x_new, axis=1))
            #    print(minus_b_m_running_mean_times_invstd_times_gamma_p_beta_div_all_p_AI_row)
            #exit()
            rhs =  np.ceil(minus_b_m_running_mean_times_invstd_times_gamma_p_beta_div_all_p_AI + EPSILON)
            lhs =  np.sum(x_new, axis=1)
            
            if( i == 0) and False:
                print(x_new, x_s_bin)
                print(rhs, lhs)

            lhs_all.append(lhs)            
            trans_x =  np.asarray(lhs >= rhs)           
            trans_y.append(trans_x)
        
        trans_y = np.asarray(trans_y)   
                
        # post constraint
        #print(rhs_constraints)
        #print(lhs_constraints)
        
        
       
        
        rhs_y_signs = np.transpose(np.asarray(rhs_y_signs))
        ht_y = ax_b_bn_t.copy()
        #print(ax_b_bn_t)
        # hardtanh
        ht_y[ ht_y > 1] = 1
        ht_y[ ht_y < -1] = -1
        #print("ht", ht_y)

        o_sign_temp = ht_y.copy()            
        o_sign_temp[ ht_y > 0] = 1
        o_sign_temp[ ht_y <= 0] = -1
        o_sign_temp = o_sign_temp.astype(int)

        o_sign = trans_y.copy().astype(int)
        o_sign_check = o_sign.copy()
        o_sign_check[ o_sign_check <=0] = -1
        #print(o_sign, rhs_y_signs)
        if (self.testing):
            assert(np.allclose(o_sign, rhs_y_signs))
        
        #o_sign[ o_sign is False] = -1

        
        for i,_ in enumerate(o_sign_check):
#             print("i=", i)
#             if (i == 0) and True:                    
#                 print(trans_y[i])
#                 print(ht_y[i])
#                 print(o_sign_check[i]) 
#                 print(o_sign_temp[i])
#             for j,_ in enumerate(o_sign_check[i]):
#                 print(o_sign_check[i][j], o_sign_temp[i][j])
#                 if (o_sign_check[i][j] != o_sign_temp[i][j]):
#                     print("j=",j)
#                     exit()
            assert(np.allclose(o_sign_check[i], o_sign_temp[i]))
        end = time.time()
        if(self.is_profile):
            print("       (last part) forward_simulator_block_binlin_bn_ht_encoding :", end - start)  
        return [x_sign, None, ax_b_bn_t, ht_y, o_sign]

    def forward_simulator_block_binlin_bn_encoding(self, id_layer, layer, x, winner = None):
        
        
        #################################
        # move to litterals
        # WE ASSUME THAT INPUT ARE BOOLEAN
        ################################
        
        # take sign
        x_sign = x.copy()
        x_sign[ x_sign > 0] = 1
        x_sign[ x_sign <= 0] = -1
        
        # linear layer + BN simulation
        #A x + b
        bin_lin_layer  = layer[0]
        A = Binarize(bin_lin_layer.weight.data, small_weight = bin_lin_layer.small_weight).cpu().detach().numpy()
        if not bin_lin_layer.bias is None:            
            b = bin_lin_layer.bias.cpu().detach().numpy()
        else:
            b =  bin_lin_layer.bias.cpu().detach().numpy()
            b.fill(0)
            

        y = []
        for i, x_s in enumerate(x_sign):
            #print(bin_weigths.shape, x_s.shape)
            trans_x =  np.dot(A, x_s) + b
            y.append(trans_x)
            #if (i == 0):
            #    print("trans_x", trans_x)
        y = np.asarray(y)        
        #print(y.shape)
        
        ax_b_bn_t = y.copy()              
        #print(ax_b_bn_t)

        num_m_one = []
        for j, row in enumerate(A):
            num_m_one.append(sum(row == -1))
        num_m_one = np.asarray(num_m_one)
        
        x_s_ones = x[0].copy()
        x_s_ones.fill(1)
        
        x_s_zeros = x[0].copy()
        x_s_zeros.fill(0)

        x_s_removed = x[0].copy()
        x_s_removed.fill(REMOVED_TERM)
                         
        x_lits = []
        for j, row in enumerate(A):
              x_lit = x_s_ones.copy()
              x_lit[row == -1] = x_s_zeros[row == -1]
              x_lit[row == 0] = x_s_removed[row == 0]
              x_lits.append(x_lit)
        
        x_lits = np.asarray(x_lits)
        

        # generate constraints 
        #print(A)
        #exit()
        #print(np.dot(A, x_s_ones))
        #exit()
        rhs_constraints =  (-b + np.dot(A, x_s_ones) ) + 2*num_m_one
        lhs_constraints =  x_lits.copy() 

        #print("lhs_constraints {}".format(lhs_constraints))
        #print("rhs_constraints {}".format(rhs_constraints))

        #print("~~~~~~ winner ~~~~~~~~", winner)
        #print(lhs_constraints)
         ## output vars mean that win >= i 
        rhs_y_signs = []
        con_reif_rhs_constant = []
        add_sum_check = []

        debug_flag = False
        assert(winner is None)
        self.encoder.winners_lits = {}         
        lits_to_win = {}

        for j, rhs in enumerate(rhs_constraints):                 
            #print(f" ------------- j {j}")
            lits_to_win[j] = {}
            # winner  -  other > =wiiner rhs  - other rhs

            
            for winnerinpair, _ in enumerate(rhs_constraints):
                if (j==winnerinpair):
                    continue
                lits_to_win[j][winnerinpair] = -1
                con_reif = {}
                con_reif["literals"] = []                
                add_sum = 0
                #print("consider {} vs {} (if winner)".format(j, winnerinpair))
                if(debug_flag):
                    print("lhs_constraints[{}]".format(winnerinpair), lhs_constraints[winnerinpair])
                    print("lhs_constraints[{}]".format(j), lhs_constraints[j])

                for k, _ in enumerate(lhs_constraints[winnerinpair]):
                    winnerinpair_l = lhs_constraints[winnerinpair][k]
                    others_l = lhs_constraints[j][k]
                    if(debug_flag):                        
                        print("-->> ", con_reif["literals"], add_sum)

                        print(f" k {k}  winnerinpair_l {winnerinpair_l} others_l {others_l}")
                    if (winnerinpair_l == 0) and (others_l == 0):
                        continue                                        
                    if (winnerinpair_l == 1) and (others_l == 1):
                        continue                                        
    
                    if (winnerinpair_l == REMOVED_TERM) and (others_l == REMOVED_TERM):
                        continue                                        
    
                    #if (A.shape[0] == 2) or True:
                    var_name = self.encoder.create_indexed_variable_name(LABEL_BIN_VARS, [id_layer, k])
                    var_id = self.encoder.lookup_varid(var_name)                       
                    #else:
                    #    var_name = self.encoder.create_indexed_variable_name(LABEL_BIN_VARS, [id_layer, k, j, winnerinpair])
                    #    var_id = self.encoder.get_varid(var_name)
                    #    var_id = self.encoder.lookup_varid(var_name)                       


                    if (winnerinpair_l == 0) and (others_l == 1):
                        var_id = -var_id
                        #add_sum+=4
                        add_sum+=2
                        con_reif["literals"].append(var_id)
                        con_reif["literals"].append(var_id)
                        con_reif["literals"].append(var_id)
                        con_reif["literals"].append(var_id)
    
           
                    if (winnerinpair_l == 0) and (others_l == REMOVED_TERM):
                        var_id = -var_id
                        #add_sum+=2
                        con_reif["literals"].append(var_id)
                        con_reif["literals"].append(var_id)
                        
    
                    if (winnerinpair_l == 1) and  (others_l == 0):
                        add_sum+=2                        
                        con_reif["literals"].append(var_id)
                        con_reif["literals"].append(var_id)
                        con_reif["literals"].append(var_id)
                        con_reif["literals"].append(var_id)
                 
                 
                    if (winnerinpair_l == 1) and  (others_l == REMOVED_TERM):
                        con_reif["literals"].append(var_id)
                        con_reif["literals"].append(var_id)
                 
                    
                    if (winnerinpair_l == REMOVED_TERM) and (others_l == 0):
                        add_sum+=2                        
                        con_reif["literals"].append(var_id)
                        con_reif["literals"].append(var_id)
    
                    if (winnerinpair_l == REMOVED_TERM) and (others_l == 1):
                        var_id = -var_id
                        add_sum+=2
                        con_reif["literals"].append(var_id)
                        con_reif["literals"].append(var_id)
    
                if(debug_flag):                                            
                    print("--<< ", con_reif["literals"], add_sum)
                add_sum_check.append(add_sum)
                        
                ## Constant   
                if(debug_flag):                  
                    print(".,.,.,--",  rhs_constraints[winnerinpair], rhs_constraints[j], add_sum)
                    print("diff", rhs_constraints[winnerinpair] - rhs_constraints[j])
                    print("add_sum", add_sum)
                con_reif["rhs_constant"] = np.ceil((rhs_constraints[winnerinpair] - rhs_constraints[j])) + add_sum
                con_reif_rhs_constant.append(con_reif["rhs_constant"])

                if (A.shape[0] == 2):
                    rhs_var_name = self.encoder.create_indexed_variable_name(LABEL_BIN_VARS, [id_layer+1, j])            
                    rhs_var_id = self.encoder.lookup_varid(rhs_var_name)
                else:

                    rhs_var_name = self.encoder.create_indexed_variable_name(LABEL_BIN_VARS, [id_layer+1, j, winnerinpair])
                    rhs_var_id = self.encoder.get_varid(rhs_var_name)
                    rhs_var_id = self.encoder.lookup_varid(rhs_var_name)
                con_reif["rhs_var"] = rhs_var_id

                #print(con_reif)
                input  = [1, 1, 1, 1 , 1]
                card_formula, rhs_y_res = self.encoder.seqcounters_reified(con_reif, id_layer, f"{j}_{winnerinpair}", assumptions= [], testing = True)
                #exit()
                #for clause in card_formula.clauses:
                #    print(clause)
                self.encoder.append(card_formula)
                if (A.shape[0] == 2):
                    self.encoder.winners_lits[winnerinpair] = rhs_var_id
                else:
                    lits_to_win[j][winnerinpair] = rhs_var_id


                #print("----------------", card_formula, rhs_y_res)
                ############################
                # Fix the winner
                ##########################
                #print(rhs_var_id)
                #print(self.encoder.formula.clauses[-10:])
                #if(j != winner):
                #    self.winner_formula = self.encoder.assign(rhs_var_id)
                #    self.loser_formula = self.encoder.assign(-rhs_var_id)                
                #print(self.encoder.formula.clauses[-10:])
            
                #print(assumptions, rhs_y_res)
                #rhs_y_signs.append(rhs_y_res)
                
                #self.encoder.solve()                
                #atleast_formula = self.encoder.atleast(con_reif)
                #print(self.encoder.solve())
                #self.encoder.append(atleast_formula)
                #print(self.encoder.solve())
            
                #print(constrait)
                #exit()

        #print(A.shape, self.encoder.winners_lits)
        if (A.shape[0] > 2):
            card_formula_fix = CNF()

            #print(lits_to_win)
            for j, rhs in enumerate(rhs_constraints):                                                                  
                rhs_var_name = self.encoder.create_indexed_variable_name(LABEL_BIN_VARS, [id_layer+1, j])            
                #print(f" rhs_var_name {rhs_var_name}")
                rhs_var_id = self.encoder.lookup_varid(rhs_var_name)
                self.encoder.winners_lits[j] = rhs_var_id

                lits = []
                for loser, winnerinpair  in lits_to_win.items():
                    #print(winnerinpair)
                    for id, var  in winnerinpair.items():
                        if (j==id):
                            lits.append(winnerinpair[id])
                            #print(f"{j},{winnerinpair}, {lits}" )

      

                # we need to ensure that this class bits others
                #print(f"self.encoder.winners_lits {self.encoder.winners_lits} ")
                # win_lit <-> win_cand_1 and win_cand2 and ..

                # win_lit -> win_cand_1 and win_cand2 and ..                
                #exit()
                for l in  lits:
                    cl1 = [-rhs_var_id]
                    cl1.append(l)
                    card_formula_fix.append(cl1)

                # win_lit <- win_cand_1 and win_cand2 and ..
                cl2 = [rhs_var_id]
                for l in  lits:
                    cl2.append(-l)
                card_formula_fix.append(cl2)
            
            self.encoder.append(card_formula_fix)
            # for cl in card_formula_fix:
            #     print(cl)
            
        #exit()
        #print(rhs_y_signs)
        trans_y = []
        lhs_all = []
        

        #print("--->", x_lits)
        for i, x_s in enumerate(x_sign):
            #print(bin_weigths.shape, x_s.shape)
            trans_x_step_1 =  np.dot(A, x_s)
            #print(trans_x_step_1, b)
            break

        for i, x_b in enumerate([x[0]]):
            #print(bin_weigths.shape, x_s.shape)
            comp_step_2 = np.dot(A, x_b)
            trans_x_step_2 =  2*np.dot(A, x_b) - np.dot(A, x_s_ones)
            #print(trans_x_step_2, b)
            #print("comp_step_2", comp_step_2)
            assert((trans_x_step_1 == trans_x_step_2).all())
            break
        

        for i, x_b in enumerate([x[0]]):
            #print(bin_weigths.shape, x_s.shape)
            #print(x_s)
            x_s_bin = x_b.copy().astype(int)
            x_s_bin_neg = 1 - x_s_bin
            #x_s_bin[x_s_bin == -1] = 0
            x_new = []
            comp_p1 = []
            comp_p2 = []
            for j, row in enumerate(A):
                # if a[i,j] =  - 1  then we replave -x_j with neg x_j - 1  
                #print(row)
                #print("x_s_bin", x_s_bin)
                x_row = x_s_bin.copy()   
                #print("x_row", x_row)             
                x_row[row == -1] = x_s_bin_neg[row == -1]         
                #print("x_row", x_row)             
                x_row[row == 0] = x_s_zeros[row == 0]                                       
                #print("x_row", x_row)                             
                x_new.append(x_row)
                comp_p1.append(sum(x_row))
                comp_p2.append(- sum(row == -1))                
            comp_p1 = np.asarray(comp_p1)
            comp_p2 = np.asarray(comp_p2)

            #print("comp_p1",comp_p1)
            assert(((comp_p1+comp_p2) == comp_step_2).all())


            #print("x_new", x_new)
            x_new = np.asarray(x_new)
        

            trans_x_step_3 =  2*comp_p1 + 2*comp_p2 - np.dot(A, x_s_ones)
            #print(trans_x_step_3, b)
            assert((trans_x_step_1 == trans_x_step_3).all())
            ((rhs_constraints == (-(b+2*comp_p2 - np.dot(A, x_s_ones)))).all())
            #print(lhs_constraints)
            #print(x_new)
            #print(A)
            
            break
        for i, x_b in enumerate([x[0]]):
            #print(bin_weigths.shape, x_s.shape)
            #print(x_s)
            x_s_bin = x_b.copy().astype(int)
            x_s_bin_neg = 1 - x_s_bin
            #x_s_bin[x_s_bin == -1] = 0
            x_new = []
            #print(lhs_constraints)
            #print("comp_p1", comp_p1)
            comp_p1_next = []
            x_row_all = []
            for j, row in enumerate(lhs_constraints):
                x_row = x_s_bin.copy()   
                #print("x_row", x_row)             
                x_row[row == 0] = x_s_bin_neg[row == 0]                    
                x_row[row == REMOVED_TERM] = x_s_zeros[row == REMOVED_TERM]   
                comp_p1_next.append(sum(x_row))
                x_row_all.append(x_row)
            assert((comp_p1 == comp_p1_next).all())   
            comp_p1_next = np.asarray(comp_p1_next)

            trans_x_step_4 =  2*comp_p1_next + 2*comp_p2 - np.dot(A, x_s_ones)
            #print(trans_x_step_4, b)
            compr_rhs = -(b+2*comp_p2 - np.dot(A, x_s_ones))

            #print(2*comp_p1_next, compr_rhs)
            
            #print(2*comp_p1_next[1] - 2*comp_p1_next[0])
            #print(2*comp_p1_next[0] - 2*comp_p1_next[1])

            assert((trans_x_step_1 == trans_x_step_4).all())
            
            # print(f"trans_x_step_1 {trans_x_step_1}")
            # print(f"trans_x_step_4 {trans_x_step_4}")


            # print(compr_rhs)
            # print(con_reif_rhs_constant)
            if   True:
                if (A.shape[0]  == 2):
                    k1 = 0
                    k2 = 1
                    p1 = 0
                    p2 = 1
                else:
                    k1 = 1
                    k2 = 2
                    cnt = 0
                    for j, rhs in enumerate(rhs_constraints):                                                                  
                        for winnerinpair, _ in enumerate(rhs_constraints):
                            if (j==winnerinpair):
                                continue                        
                            if k1 ==j and k2 == winnerinpair:
                                p1 = cnt
                            if k2 ==j and k1 == winnerinpair:
                                p2 = cnt
                            cnt = cnt + 1
                row_0 = A[k1]
                row_1 = A[k2]
                diff_01 = row_0 - row_1
                diff_10 = row_1 - row_0       
                    
                #print("----------")
                s0 = 0
                for k,d in enumerate(diff_01): 
                    if (diff_01[k] != 0 ) and (row_1[k] != 0):
                        s0 = 2 + s0

                s1 = 0
                for k,d in enumerate(diff_10): 
                    if (diff_10[k] != 0 ) and (row_0[k] != 0):
                        s1 = 2 + s1

                #print(s0)                                
                #print(s1)                                
                #print(add_sum_check)
                assert(np.ceil(compr_rhs[k2] - compr_rhs[k1] + s1 ) == con_reif_rhs_constant[p1])
                assert(np.ceil(compr_rhs[k1] - compr_rhs[k2]+ s0 ) == con_reif_rhs_constant[p2])
                
                #print(con_reif_rhs_constant)
                #print(comp_p1_next)
                #print(x_row_all)
            break
        #exit()        
#         #exit()
#         for i, x_s in enumerate([x[0]]):
#             row_0 = A[0]
#             row_1 = A[1]
#             diff_01 = row_0 - row_1
#             diff_10 = row_1 - row_0
#             rhs_01 = b[0] - b[1]
#             rhs_10 = b[1] - b[0]
#             print("row_0", row_0)
#             print("row_1", row_1)
#             print("diff_01", diff_01)
#             print("diff_10", diff_10)
# 
#             print("rhs_01", rhs_01)
#             print("rhs_10", rhs_10)
#             
#             print(sum(diff_01*x_s))
#             print(sum(diff_10*x_s))
#             
#             print(x_s)
#             x_s_bin = x_s.copy().astype(int)
#             x_s_bin_neg = 1 - x_s_bin
#             #x_s_bin[x_s_bin == -1] = 0
#             x_new = []
#             for j, row in enumerate(A):
#                 # if a[i,j] =  - 1  then we replave -x_j with neg x_j - 1  
#                 print(row)
#                 #print("x_s_bin", x_s_bin)
#                 x_row = x_s_bin.copy()   
#                 #print("x_row", x_row)             
#                 x_row[row == -1] = x_s_bin_neg[row == -1]         
#                 #print("x_row", x_row)             
#                 x_row[row == 0] = x_s_zeros[row == 0]                                       
#                 #print("x_row", x_row)                             
#                 x_new.append(x_row)
# 
#             x_new = np.asarray(x_new)
#             print("x_new", x_new)    
#             print(sum(row_0*x_s))
#             print(sum(x_new[0]))
#             exit()
               
        for i, x_s in enumerate(x):            
            # bin x            
            x_s_bin = x_s.copy().astype(int)
            x_s_bin_neg = 1 - x_s_bin
            #x_s_bin[x_s_bin == -1] = 0
            x_new = []
            for j, row in enumerate(A):
                # if a[i,j] =  - 1  then we replave -x_j with neg x_j - 1  
                #print(row)
                #print("x_s_bin", x_s_bin)
                x_row = x_s_bin.copy()   
                #print("x_row", x_row)             
                x_row[x_lits[j] == 0] = x_s_bin_neg[x_lits[j] == 0]         
                #print("x_row", x_row)             
                x_row[row == 0] = x_s_zeros[row == 0]                                       
                #print("x_row", x_row)                             
                x_new.append(x_row)

            x_new = np.asarray(x_new)
           # print("x_new", x_new)
            #if( i == 0):
            #    print(x_new)  
            #    print(np.sum(x_new, axis=1))
            #    print(minus_b_m_running_mean_times_invstd_times_gamma_p_beta_div_all_p_AI_row)
            #exit()
            rhs =   (-b + np.dot(A, x_s_ones) ) + 2*num_m_one
            lhs =  2*np.sum(x_new, axis=1)
            if(debug_flag):
                print("i {}: lhs {}, rhs {}, diff {}, x_new {}" .format(i, lhs, rhs, lhs - rhs, x_new))
            
            if False:
                print(x_new, x_s_bin)
                print(rhs, lhs)

            lhs_all.append(lhs)            
            trans_x =  np.asarray(lhs >= rhs)           
            trans_y.append(trans_x)
        
        trans_y = np.asarray(trans_y)   
        #print(trans_y) 
        # post constraint
        #print(rhs_constraints)
        #print(lhs_constraints)
        
        
       
        ht_y = None
        o_sign = None
        #print(lhs_all[0])
        #for i,_ in enumerate(lhs_all[0]):
        #    if(i == winner):
        #        continue
        #    assert(lhs_all[0][winner] > lhs_all[0][i]) 
        
        #rhs_y_signs = np.transpose(np.asarray(rhs_y_signs))
        #print(rhs_y_signs)
        #print(ax_b_bn_t)
        #assert((rhs_y_signs[0] == 1).all()) 
        #exit()
        #if not (winner is None):

        return [x_sign, None, ax_b_bn_t, ht_y, o_sign]
            
    def forward_encoder(self, x, forward_outputs, forward_outputs_extra = None, winner = None):
        self.winner_formula = None
        self.loser_formula = None

        out = x.view(-1, self.side).clone().cpu().detach().numpy()

        start = time.time()

        # check that all nessesary variable are created
        for i, (layer, type) in enumerate(self.all_layers):      
            nb_neurons_layer  =  layer[0].weight.shape[1]
            for k in range(nb_neurons_layer):
                 var_name = self.encoder.create_indexed_variable_name(LABEL_BIN_VARS, [i, k])
                 self.encoder.lookup_varid(var_name)
                 
            if(type == BinLin_NOBN):
                nb_neurons_layer  =  layer[0].weight.shape[1]
                for k in range(nb_neurons_layer):
                    var_name = self.encoder.create_indexed_variable_name(LABEL_BIN_VARS, [i, k])
                    self.encoder.lookup_varid(var_name)
       
        end = time.time()
        if(self.is_profile):
            print("    init vars :", end - start)            
        outputs_by_layers = []
        outputs_by_layers.append(out[0].astype(int))
        for i, (layer, type) in enumerate(self.all_layers):       
            start = time.time()
                 
            #print(i,  type)#, forward_outputs[i].shape)layer,
            if (type == BinLin_BN_REIF):
                #print("Block " , BinLin_BN_REIF)
                    
                simulated_output  = self.forward_simulator_block_binlin_bn_ht_encoding(i, layer, out)
                #return
                # LIN
                sim_ax_b = simulated_output[1]            
                                
                # BN                
                sim_bn_y = simulated_output[2]
                
                # HT          
                sim_ht_y = simulated_output[3]      
                ht_y = forward_outputs[i]
                
                out_sign = ht_y[0].copy()
                out_sign[ out_sign > 0] = 1
                out_sign[ out_sign <= 0] = 0
                outputs_by_layers.append(out_sign.astype(int))
                #print(outputs_by_layers)


                #print(sim_ht_y, ht_y)
               # print(sim_ht_y[0], ht_y[0])
                #for i, v  in enumerate(sim_ht_y[0]):
                #    print(v, ht_y[0][i])
                #    if abs(v - ht_y[0][i]) > 0.001:
                #        exit()
                assert(np.allclose(sim_ht_y, ht_y, atol = 0.0001))
                if (simulated_output[4] is None):
                    out = ht_y
                else:
                    out = simulated_output[4]
                #exit()                            
                #return
            

            if (type == BinLin_NOBN):
                #print("Block " , BinLin_NOBN)
                simulated_output  = self.forward_simulator_block_binlin_bn_encoding(i, layer, out, winner = winner)
                sim_ax_b = simulated_output[1]            
                
                # BN                
                sim_bn_y = simulated_output[2]
                bn_y = forward_outputs[i]
                outputs_by_layers.append(bn_y[0].copy())
                assert(np.allclose(sim_bn_y, bn_y))
                #print(sim_bn_y, bn_y)
            end = time.time()
            if(self.is_profile):
                print("    layer {}:".format(i), end - start)                
            #exit()

        print("Encoder is OK")
        return outputs_by_layers           
 
    def predict_prob(self, x):
        #print(x.shape)
        x = x[0].astype(float)
        #print(x)
        x_gpu  = np.vstack((x, x))
        x_gpu = torch.from_numpy(x_gpu).type(self.config["model"]["type_model"])
        x_gpu = x_gpu.view(2, len(x))
        
        #print("x_gpu", x_gpu)
        out = self.forward(x_gpu)    
        
        if (self.config["train"]["loss"] in "BCELoss"):
            p = out.cpu().detach().numpy()[0][0]
            return np.asarray([p, 1- p]) 
        if (self.config["train"]["loss"] in "CrossEntropyLoss"):
            return out.cpu().detach().numpy()[0]

    def predict_prob_lime(self, x):
        x = x.astype(float)
        #print(x)
        x_gpu  = x
        x_gpu = torch.from_numpy(x_gpu).type(self.config["model"]["type_model"])
        x_gpu = x_gpu.view(-1, len(x))
        
        #print("x_gpu", x_gpu)
        out = self.forward(x_gpu)    
        
        if (self.config["train"]["loss"] in "BCELoss"):
            p = out.cpu().detach().numpy()[0]
            return np.asarray([p, 1- p]) 
        if (self.config["train"]["loss"] in "CrossEntropyLoss"):
            return out.cpu().detach().numpy()
        
    def predict(self, x):
        #print("x.shape", x.shape)
        #print(x)
        x = x[0].astype(float)
        #print(x,  x.shape)
        x_gpu  = np.vstack((x, x))
        x_gpu = torch.from_numpy(x_gpu).type(self.config["model"]["type_model"])
        x_gpu = x_gpu.view(2, x.shape[0])
        
        
        #print("x_gpu", x_gpu)
        out = self.forward(x_gpu)
        if (self.config["train"]["loss"] in "BCELoss"):        
            p = out.cpu().detach().numpy()[0][0]
            #print("p.shape", p.shape)
            if (p <= 0.5):            
                return np.asarray([np.float32(0)])
            else:
                return np.asarray([np.float32(1)])
        if (self.config["train"]["loss"] in "CrossEntropyLoss"):
            _, pred = out.float().topk(1, 1, True, True)
            pred = pred.cpu().detach().numpy()[0][0]    
            #print(np.float32(pred))         
            return np.asarray([np.float32(pred)])
    
    def predict_anchor(self, x):
        #print("predict_anchor : input ", x.shape)
        x_gpu = torch.from_numpy(x).type(self.config["model"]["type_model"])    
        if len(x.shape) == 1:    
            x_gpu = x_gpu.view(-1, x.shape[0])
        
        
        #print("x_gpu", x_gpu.shape)
        out = self.forward(x_gpu)
        if (self.config["train"]["loss"] in "BCELoss"):        
            p = out.cpu().detach().numpy()[0]
            #print(p)
            if (p <= 0.5):            
                return np.asarray([np.float32(0)])
            else:
                return np.asarray([np.float32(1)])
        if (self.config["train"]["loss"] in "CrossEntropyLoss"):
            _, pred = out.float().topk(1, 1, True, True)
            if len(x.shape) == 1:  
                pred = pred[0]
            else:
                pred = pred.view(pred.shape[0]) 
            pred = pred.cpu().detach().numpy()  
            #print("return", pred.shape)         
            return np.asarray(np.float32(pred))

def mlp_binary(config):
    return MLPNetOWT_BN(config)

#                          
#                          ################ printing ################################3
#                          if (l == 0):
#                              constrait = constrait + "-"                        
#                              var_id = -var_id
#                          else:
#                              constrait = constrait + "+"                        
#                          constrait = constrait + var_name + " "
#                          ################################################3                                                  
#                 ################################################
#                 constrait = constrait  + " >= " + str(rhs) + "  <=> " + var_name
#                 ################################################

                