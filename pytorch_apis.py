import torch as th
import gp_apis

class gemm_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim_0, dim_1, device0):
        res = gp_apis.gp_gemm(input1, input2, dim_0, dim_1, device0)
        ctx.backward_cache = input1, input2, device0
        return res

    @staticmethod
    def backward(ctx, dZ):
        input1, input2, device0 = ctx.backward_cache
        dX = gp_apis.gp_gemm(dZ, input2.t().contiguous(), input1.shape[0], input1.shape[1], device0) 
        dW = gp_apis.gp_gemm(input1.t().contiguous(), dZ, input2.shape[0], input2.shape[1], device0)
        return dX, dW, None, None, None

def gemm(input1, input2, dim_0, dim_1, device0):
    return gemm_impl.apply(input1, input2, dim_0, dim_1, device0)

class spmm_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, input3, input4, dim_0, dim_1, device0):
        res = gp_apis.gp_spmm(input1, input2, input3, input4, dim_0, dim_1, device0)
        ctx.backward_cache = input1, input2, input3, device0
        return res

    # input should be symmetric
    @staticmethod
    def backward(ctx, dZ):
        input1, input2, input3, device0 = ctx.backward_cache
        res = gp_apis.gp_spmm(input1, input2, input3, dZ, dZ.shape[0], dZ.shape[1], device0)
        return None, None, None, res, None, None, None
        
def spmm(input1, input2, input3, input4, dim_0, dim_1, device0):
    return spmm_impl.apply(input1, input2, input3, input4, dim_0, dim_1, device0)

