import torch
import pytorch_apis

device = torch.device('cuda')
A = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).to(device)
B = torch.Tensor([[1, 1], [2, 2], [3, 3]]).to(device)
B.requires_grad_(True)

C = pytorch_apis.gemm(A, B, 3, 2, device)
print(C.data)
C.sum().backward()
print(B.grad)


Ptr = torch.Tensor([0,2,3,5]).to(device)
Dst = torch.Tensor([0, 2, 1, 0, 2]).to(device)
Bs = torch.Tensor([[1, 1], [2, 2], [3, 3]]).to(device)
Bs.requires_grad_(True)
Degree = torch.Tensor([1, 1, 1]).to(device)
Cs = pytorch_apis.spmm(Ptr, Dst, Degree, Bs, 3, 2, device)
print(Cs)
Cs.sum().backward()
print(Bs.grad)
