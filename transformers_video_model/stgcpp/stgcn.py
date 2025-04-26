import copy as cp

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(__file__))
from graph import Graph
from stgcn_block import STGCNBlock

EPS = 1e-4


class STGCN(nn.Module):

    def __init__(self,
                 hidden_size,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=1,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=(5, 8),
                 down_stages=(5, 8),
                 **kwargs):
        super().__init__()

        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
       
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.projection = nn.Linear(256,hidden_size)


    def forward_extraction(self, x):
        # x  = x.unsqueeze(1)
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
       
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])
        return x

    def head_forward(self, x):
        x  = x.squeeze()
        N, C, T, V = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.projection(x)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        
        # # Step 3: Standardization (zero mean, unit variance)
        mean = x.mean(dim=[0, 1, 3, 4], keepdim=True)  # Shape: [1, 1, C, 1, 1]
        std = x.std(dim=[0, 1, 3, 4], keepdim=True)  # Shape: [1, 1, C, 1, 1]
        std = torch.clamp(std, min=1e-6)  # Avoid division by zero
        x = (x - mean) / std
 
        x = self.forward_extraction(x)
        x = self.head_forward(x)
       
        return x
    
if __name__ == "__main__":
    model = STGCN(256).to('cuda:0')
    x = torch.randn(2,1,16,27,3).to('cuda:0')
    print(model(x).shape)
    