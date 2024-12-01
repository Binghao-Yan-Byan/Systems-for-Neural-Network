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

class gspmmv_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, dim_0, dim_1, device0):
        res = gp_apis.gp_gspmmv(graph, input1, dim_0, dim_1, device0)
        ctx.backward_cache = graph, input1.shape[0], input1.shape[1], device0
        return res

    @staticmethod
    def backward(ctx, dZ):
        graph, dim_0, dim_1, device0 = ctx.backward_cache
        res = gp_apis.gp_gspmmv(graph, dZ, dim_0, dim_1, device0)
        return None, res, None, None, None

def gspmmv(graph, input1, dim_0, dim_1, device0):
    return gspmmv_impl.apply(graph, input1, dim_0, dim_1, device0)
    

