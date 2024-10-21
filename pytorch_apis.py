import torch as th
import gp_apis

class gemm_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim_0, dim_1, device0):
        res = gp_apis.gp_gemm(input1, input2, dim_0, dim_1, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gemm(input1, input2, dim_0, dim_1, device0):
    return gemm_impl.apply(input1, input2, dim_0, dim_1, device0)

class spmm_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, input3, input4, dim_0, dim_1, device0):
        res = gp_apis.gp_spmm(input1, input2, input3, input4, dim_0, dim_1, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def spmm(input1, input2, input3, input4, dim_0, dim_1, device0):
    return spmm_impl.apply(input1, input2, input3, input4, dim_0, dim_1, device0)

