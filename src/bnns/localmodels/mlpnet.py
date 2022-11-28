import torch.nn as nn
import torchvision.transforms as transforms
from .binarized_modules import  BinarizeLinear,BinarizeConv2d

__all__ = ['mlpnet']
class MLPNetOWT(nn.Module):
    def __init__(self, nbclasses, input_size):        
        super(MLPNetOWT, self).__init__()
        self.side = input_size
        self.layer1 = nn.Sequential(
            nn.Linear(self.side, 50, bias=False),
            nn.BatchNorm1d(50),
           nn.Hardtanh(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Linear(50, 50, bias=False),
            nn.BatchNorm1d(50),
            nn.Hardtanh(inplace=True))
        self.layer3 = nn.Sequential(
            nn.Linear(50, 50, bias=False),
            nn.BatchNorm1d(50),
            nn.Hardtanh(inplace=True))
        self.logsoftmax = nn.LogSoftmax()
#         self.layer4 = nn.Sequential(
#             BinaryLinear(100, 100, bias=False),
#            nn.BatchNorm2d(100, momentum=args.momentum, eps=args.eps),
#             BinaryTanh())
        #self.fc = nn.Linear(100, nbclasses, bias=False)
        self.bfc = nn.Linear(50, nbclasses, bias=False)
        self.all_layers = [self.layer1,
                            self.layer2,
                            self.layer3,
                            #self.layer4,
                            self.bfc
                            ]

            

        
    def forward(self, x, switch = None):
        #print(x.size())
        out = x.view(-1, self.side)
        for layer in self.all_layers[:-1]:
            #print(out.shape)
            #print(layer)
            out = layer(out)
            #print(out.shape)
        out = out.view(out.size(0), -1)
        #last_layer = out[:]
        #if (epoch < self.splitpoint):
        #    out = self.fc(out)
        #    self.bfc.weight = self.fc.weight
        #    self.bfc.bias = self.fc.bias
        #else:
            #print("bc")
        out = self.bfc(out)    
         
        return self.logsoftmax(out)
    
    
    def forward_check(self, x):
        #print(x.size())
        out_all = {}
        nb_layer = 0
        x = x.view(-1, self.side*self.side)
        # print(x.size())
        out = x
        out_all[nb_layer] = out.cpu().data.numpy()[0]
        nb_layer +=1 
        for layer in self.all_layers[:-1]:
            #print("out ", out)                  
            out = layer(out)
            out_all[nb_layer] = out.cpu().data.numpy()[0]
            nb_layer +=1 
        
        out = out.view(out.size(0), -1)
        out = self.bfc(out)
        out_all[nb_layer] = out.cpu().data.numpy()[0]
        nb_layer +=1 

        return out_all
    
    
    def print_BinaryLinear(self, dist_point, singlelayer):
        bin_weigths = binarize(singlelayer.weight)
        nb_rows = (bin_weigths.size())[0]
        nb_cols = (bin_weigths.size())[1]
        print(nb_rows, nb_cols)
        dist_point.write("{} {} {} \n".format(SET_LAYER_ID, nb_rows, nb_cols))
        for i in range(nb_rows):
            for j in range(nb_cols):
                dist_point.write("{:3} ".format(int(bin_weigths[i][j].data[0])))
            if( not singlelayer.bias == None):
                dist_point.write("{:3} ".format(bin_weigths[i].data[0]))    
            else:    
                dist_point.write("{:3} ".format(0))
            dist_point.write("\n")

    
#     def print_BatchNorm2d(self, dist_point, singlelayer):
#         eps = singlelayer.eps
#         momentum = singlelayer.momentum
#         weight = singlelayer.weight
#         bias = singlelayer.bias
#         running_mean = singlelayer.running_mean
#         running_var = singlelayer.running_var
#         num_features = singlelayer.num_features
# 
#         dist_point.write("{} {} \n".format(SET_LAYER_BN_ID, num_features))
#         if (num_features > 85):
#             j = 0
#             print(" mean = {}, var ={}, eps ={}, weights ={}, bias={} ".format(  running_mean[j],  running_var[j], eps, weight.cpu().data.numpy()[j], bias.cpu().data.numpy()[j]))
#             #exit()  
#         
#         ##################
#         # EPS
#         ##################    
#         dist_point.write("{}\n".format(SET_BN_EPS_ID))
#         if( not eps == None):
#             dist_point.write("{0:.6f} ".format(eps))    
#         else:    
#             dist_point.write("{0:.6f} ".format(0))
#         dist_point.write("\n")
# 
#         dist_point.write("{}\n".format(SET_BN_WEIGHT_ID))        
#         for i in range(num_features):
#             dist_point.write("{0:.6f} ".format(weight[i].data[0]))    
#         dist_point.write("\n")
# 
# 
#         dist_point.write("{}\n".format(SET_BN_BIAS_ID))        
#         for i in range(num_features):
#             dist_point.write("{0:.6f} ".format(bias[i].data[0]))    
#         dist_point.write("\n")
# 
#         dist_point.write("{}\n".format(SET_BN_RUNMEAN_ID))        
#         for i in range(num_features):
#             dist_point.write("{0:.6f} ".format(running_mean[i]))    
#         dist_point.write("\n")
#                 
#         
#         dist_point.write("{}\n".format(SET_BN_RUNVAR_ID))        
#         for i in range(num_features):
#             dist_point.write("{0:.6f} ".format(running_var[i]))    
#         dist_point.write("\n")
#         
#     def forward_print(self, dist_file):
#         dist_point = open(dist_file, 'w')        
#        
#         for layer in self.all_layers:
#             #print(layer)
#             try:
#                 for singlelayer in layer:
#                     name  = str(singlelayer)
#                     print("Looking at " + name)
# 
#                     if ('BinaryLinear' in name):
#                         self.print_BinaryLinear(dist_point, singlelayer)
#                     if ('BatchNorm2d' in name):
#                         self.print_BatchNorm2d(dist_point, singlelayer)
#             except TypeError:
#                 name  = str(layer)
#                 print("Looking at " + name)
#                 if ('BinaryLinear' in name):
#                     self.print_BinaryLinear(dist_point, layer)
#                 if ('BatchNorm2d' in name):
#                     self.print_BatchNorm2d(dist_point, layer)                
#         dist_point.close()


def mlpnet(**kwargs):
    num_classes = dict(kwargs)['num_classes']
    input_size = dict(kwargs)['input_size']
    return MLPNetOWT(num_classes,input_size)
