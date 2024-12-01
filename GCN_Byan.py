import torch
import math
import torch.nn.init as init
import torch.nn.functional as F
import pytorch_apis
import numpy as np
import graphpy
import argparse
import dgl
import scipy.sparse as sp

class GraphConv_Byan(torch.nn.Module):
    # Do similar things like torch.nn.modules.Linear
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device = None, dtype = None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, g, input):
        mat = pytorch_apis.gemm(input, self.weight, input.shape[0], self.out_features, self.device)
        result = pytorch_apis.gspmmv(g, mat, input.shape[0], self.out_features, self.device)
        if self.bias is not None:
            result = result + self.bias
        return result
    
class GCN_Byan(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device, bias=True):
        super().__init__()
        self.conv1 = GraphConv_Byan(in_features, hidden_features, bias, device=device)
        self.conv2 = GraphConv_Byan(hidden_features, out_features, bias, device=device)
        self.relu = torch.nn.ReLU()

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = self.relu(h)
        h = self.conv2(g, h)
        return h
    
if __name__ == "__main__":
    device = torch.device('cuda')
    Ptr = np.array([0, 2, 4, 5, 7, 8], dtype=np.int32)
    Dst = np.array([1, 4, 0, 3, 3, 1, 2, 0], dtype=np.int32)
    Degree = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    a = graphpy.init_graph(Ptr, Dst, Degree)
    spB = torch.Tensor([
        [3, 6, 1, 9],
        [7, 4, 2, 3],
        [8, 6, 9, 2],
        [5, 2, 8, 7],
        [6, 9, 3, 1]
    ]).requires_grad_(True)
    _spB = spB.to(device)
    model = GCN_Byan(4, 2, 4, device=device, bias=True)
    optimizer = torch.optim.SGD(model.parameters() ,lr=0.01)
    Cspmm = model(a, _spB)
    print(Cspmm)
    spz = Cspmm.sum()
    spz.backward()
    print(spz.item())
    print("GRAD"*10)
    print(model.conv1.weight.data)
    print(model.conv1.weight.grad)
    print(model.conv2.weight.data)
    print(model.conv2.weight.grad)
    optimizer.step()
    print(model.conv1.weight.data)
    print(model.conv2.weight.data)