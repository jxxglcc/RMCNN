'''
#####################################################################################################################
Description: 

This is a python implementation of Tensor-CSPNet and Graph-CSPNet. The hyperparameters in the model are 
used in the experiments. We pick the best results of several runs and put them in the paper. The performance could 
be optimized further. 

            Input Shape: 
                        (batch size, time windows * frequency bands, channel No., channel No.) ----> Tensor-CSPNet;
                        (batch size,  segment No., channel No., channel No.)                   ---->  Graph-CSPNet.   

            self.mlp: multilayer perception (1 layer, if false / 3 layers, if true).

            self.channel_num: time windows * frequency bands ----> Tensor-CSPNet;
                                                 segment No. ---->  Graph-CSPNet.

            self.dimes: This is for the shape dimension (in and out) in each BiMap layer, 

                        for example, [20, 30, 30, 20] means the first BiMap layer has the input dimension 20, 
                        and output dimension 30, and the second BiMap layer has the input dimension 30 and 
                        output dimension 20. 

            self.kernel_size: This number is the total number of temporal segments. 

            self.tcn_channels: how many output channels h are in CNNs. We suggest a relatively big number. This is
            because the small one yields a lose of discrimitive information. For example, if kernel_size = 1, 
            the tcn_channel = 16.

#######################################################################################################################
'''

import torch.nn as nn
import torch as th
from .modules import *
from utils.args import args_parser

args   = args_parser()
use_cuda = not args.no_cuda and th.cuda.is_available()
# use_cuda = False
device   = th.device("cuda" if use_cuda else "cpu")

class Tensor_CSPNet_Basic(nn.Module):

    def __init__(self, channel_num, mlp, dataset = 'KU'):
        super(Tensor_CSPNet_Basic, self).__init__()

        self._mlp             = mlp
        self.channel_in       = channel_num
 
        if dataset == 'KU':
            classes           = 2
            self.dims         = [21, 30, 30, 21]
            self.kernel_size  = 3
            self.tcn_channels = 48
        elif dataset == 'BCIC':
            classes           = 4
            self.dims         = [22, 36, 36, 22]
            self.kernel_size  = 2
            self.tcn_channels = 16
        elif dataset == 'HG':
            classes           = 4
            self.dims         = [22, 36, 36, 22]
            self.kernel_size  = 2
            self.tcn_channels = 16
            
        self.BiMap_Block      = self._make_BiMap_block(len(self.dims)//2)
        self.LogEig           = LogEig()

        '''
        We use 4Hz bandwith on 4Hz ~ 40Hz for the frequency segmentation, and the largest, i.e., 9 (=36/4), 
        performs the best usually. Hence, we pick self.tcn_width = 9.  
        '''

        self.tcn_width        =  9 
        self.Temporal_Block   = nn.Conv2d(1, self.tcn_channels, (self.kernel_size, self.tcn_width*self.dims[-1]**2), stride=(1, self.dims[-1]**2), padding=0).double()
        
        if self._mlp:
            self.Classifier = nn.Sequential(
            nn.Linear(self.tcn_channels, self.tcn_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.tcn_channels, self.tcn_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.tcn_channels, classes)
            ).double()
        else:
            self.Classifier = nn.Linear(self.tcn_channels, classes).double()
    
    def _make_BiMap_block(self, layer_num):
        layers = []

        if layer_num > 1:
          for i in range(layer_num-1):
            dim_in, dim_out = self.dims[2*i], self.dims[2*i+1]
            layers.append(BiMap(self.channel_in, dim_in, dim_out))
            layers.append(ReEig())
        
        dim_in, dim_out = self.dims[-2], self.dims[-1]
        layers.append(BiMap(self.channel_in, dim_in, dim_out))
        layers.append(BatchNormSPD(momentum = 0.1, n = dim_out))
        layers.append(ReEig())
          
        return nn.Sequential(*layers).double()

    def forward(self, x):

        window_num, band_num = x.shape[1], x.shape[2]

        x     = x.reshape(x.shape[0], window_num*band_num, x.shape[3], x.shape[4])

        x_csp = self.BiMap_Block(x)

        x_log = self.LogEig(x_csp)

        # NCHW Format: (batch_size, window_num*band_num, 4, 4) ---> (batch_size, 1, window_num, band_num * 4 * 4)
        x_vec = x_log.view(x_log.shape[0], 1, window_num, -1)

        y     = self.Classifier(self.Temporal_Block(x_vec).reshape(x.shape[0], -1))

        return y


class Graph_CSPNet_Basic(nn.Module):

    def __init__(self, channel_num, P, mlp, dataset = 'KU'):
        super(Graph_CSPNet_Basic, self).__init__()

        self._mlp       = mlp
        self.channel_in = channel_num
        self.P          = P

        if dataset   == 'KU':
            classes     = 2
            self.dims   = [21, 30, 30, 21]
        elif dataset == 'BCIC':
            classes     = 4
            self.dims   = [22, 36, 36, 22]
        elif dataset == 'HG':
            classes     = 4
            self.dims   = [22, 36, 36, 22]

        self.Graph_BiMap_Block = self._make_Graph_BiMap_block(len(self.dims)//2)
        self.LogEig            = LogEig()
        
        if self._mlp:
            self.Classifier = nn.Sequential(
            nn.Linear(channel_num*self.dims[-1]**2, channel_num),
            nn.ReLU(inplace=True),
            nn.Linear(channel_num, channel_num),
            nn.ReLU(inplace=True),
            nn.Linear(channel_num, classes)
            ).double()
        else:
            self.Classifier = nn.Linear(channel_num*self.dims[-1]**2, classes).double()

    def _make_Graph_BiMap_block(self, layer_num):

        layers = []
        _I     = th.eye(self.P.shape[0], dtype=th.double).to(device)
        
        if layer_num > 1:
          dim_in, dim_out = self.dims[0], self.dims[1]
          layers.append(Graph_BiMap(self.channel_in, dim_in, dim_out, self.P))
          layers.append(ReEig())

          for i in range(1, layer_num-1):
            dim_in, dim_out = self.dims[2*i], self.dims[2*i+1]
            layers.append(Graph_BiMap(self.channel_in, dim_in, dim_out, _I))
            layers.append(ReEig())

          dim_in, dim_out = self.dims[-2], self.dims[-1]
          layers.append(Graph_BiMap(self.channel_in, dim_in, dim_out, _I))
          layers.append(BatchNormSPD(momentum = 0.1, n = dim_out))
          layers.append(ReEig())

        else:
          dim_in, dim_out = self.dims[-2], self.dims[-1]
          layers.append(Graph_BiMap(self.channel_in, dim_in, dim_out, self.P))
          layers.append(BatchNormSPD(momentum = 0.1, n = dim_out))
          layers.append(ReEig())

        return nn.Sequential(*layers).double()


    def forward(self, x):

        x_csp = self.Graph_BiMap_Block(x)

        x_log = self.LogEig(x_csp)

        # NCHW Format (batch_size, window_num*band_num, 4, 4) ---> (batch_size, 1, window_num, band_num * 4 * 4)
        x_vec = x_log.view(x_log.shape[0], -1)

        y     = self.Classifier(x_vec)

        return y
