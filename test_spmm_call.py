import torch
import pytorch_apis

device = torch.device('cuda')
A = torch.Tensor([[0, 1, 1], [0, 1, 1], [1, 0, 1]]).to(device)
B = torch.Tensor([[1, 1], [2, 2], [3, 3]]).to(device)

C = pytorch_apis.gemm(A, B, 3, 2, device)
print(C)

Ptr = torch.Tensor([0,2,4,6]).to(device)
Dst = torch.Tensor([1, 2, 1, 2, 0, 2]).to(device)
Degree = torch.Tensor([1, 1, 1]).to(device)
Cspmm = pytorch_apis.spmm(Ptr, Dst, Degree, B, 3, 2, device)
print(Cspmm)