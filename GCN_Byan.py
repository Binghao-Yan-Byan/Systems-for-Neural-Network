import torch
import math
import torch.nn.init as init
import torch.nn.functional as F
import pytorch_apis

class GraphConv_Byan(torch.nn.Module):
    # Do similar things like torch.nn.modules.Linear
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device = None, dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features), **factory_kwargs))
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
        ptrs, dsts, deg = g
        #mat = torch.mm(input, self.weight)
        mat = pytorch_apis.gemm(input, self.weight, input.shape[0], self.out_features, self.device) 
        result = pytorch_apis.spmm(ptrs, dsts, deg, mat, mat.shape[0], self.out_features, self.device)
        #torch.mm(mat, self.weight)
        if self.bias is not None:
            result += self.bias
        return result
    
class GCN_Byan(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, device):
        super().__init__()
        self.conv1 = GraphConv_Byan(in_feats, hidden_feats, device=device)
        self.conv2 = GraphConv_Byan(hidden_feats, out_feats, device=device)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
if __name__ == "__main__":
    device = torch.device('cuda')
    gb = GraphConv_Byan(2, 2, device=device)
    gb2 = GraphConv_Byan(2, 2, device=device)
    Ptr = torch.Tensor([0,2,3,5]).to(device)
    Dst = torch.Tensor([0, 2, 1, 0, 2]).to(device)
    Bs = torch.Tensor([[1, 1], [2, 2], [3, 3]]).to(device)
    Bs.requires_grad_(True)
    
    Degree = torch.Tensor([1, 1, 1]).to(device)
    g = (Ptr, Dst, Degree)
    Cs = gb(g, Bs)
    Cs = F.relu(Cs)
    Cs = gb2(g, Cs)
    print(Cs)
    print(Cs.sum())
    Cs.sum().backward()
    print("$$$$$$$$"*5)
    print(Bs.grad)