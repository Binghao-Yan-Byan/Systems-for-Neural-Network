import torch as th
import torch.utils.dlpack
import graphpy as gpk
def gp_gemm(input1, input2, dim1_0, dim1_1, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.gemm(input1_dl, input2_dl, res_dl1)
    return res1
def gp_gspmmv(graph, input1, dim1_0, dim1_1, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    stream = th.cuda.current_stream().cuda_stream
    gpk.gspmmv(graph, input1_dl, res_dl1, stream)
    return res1
