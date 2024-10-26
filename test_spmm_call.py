import torch
import pytorch_apis
import graphpy
import numpy as np


device = torch.device('cuda')
A = torch.Tensor([
    [0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0]
]).to(device)
B = torch.Tensor([
    [3, 6, 1, 9],
    [7, 4, 2, 3],
    [8, 6, 9, 2],
    [5, 2, 8, 7],
    [6, 9, 3, 1]
]).to(device)
C = pytorch_apis.gemm(A, B, 5, 4, device)
print(C)


Ptr = np.array([0, 2, 4, 5, 7, 8], dtype=np.int32)
Dst = np.array([1, 4, 0, 3, 3, 1, 2, 0], dtype=np.int32)
Degree = np.array([1, 1, 1, 1, 1], dtype=np.int32)
a = graphpy.init_graph(Ptr, Dst, Degree)
print(a.get_vcount())
print(a.get_edge_count())
a.print_graph()

Bspmm = torch.Tensor([
    [3, 6, 1, 9],
    [7, 4, 2, 3],
    [8, 6, 9, 2],
    [5, 2, 8, 7],
    [6, 9, 3, 1]
])
Cspmm = pytorch_apis.gspmmv(a, Bspmm, 5, 4, device)
print(Cspmm)