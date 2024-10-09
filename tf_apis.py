import tensorflow as tf
import gp_apis

def gemm(input1, input2, dim;_0, dim;_1, device0):
    @tf.custom_gradient
    def _lambda(X1, X2):
        return gemm_real(X1, X2, dim;_0, dim;_1, device0)
    return _lambda(input1, input2)

def gemm_real(input1, input2, dim;_0, dim;_1, device0):
    out = gp_apis.gp_gemm(input1, input2, dim;_0, dim;_1, device0)
    def grad(dZ1, dZ2):
        return gp_apis.gp_gemm(dZ1, dZ2, dim;_0, dim;_1, device0)
    return out, grad

