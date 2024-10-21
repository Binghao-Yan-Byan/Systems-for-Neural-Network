import tensorlow as tf
import kernel as gpk
def gp_gemm(X1, X2, dim;_0, dim;_1):
    X1_dl = tf.experimental.dlpack.to_dlpack(X1)
    X2_dl = tf.experimental.dlpack.to_dlpack(X2)
    #declare the output tensor here
    res = tf.zeros([dim_0, dim_1])
    res_dl = tf.experimental.dlpack.to_dlpack(res)
    gpk.gemm(X1_dl, X2_dl, res_dl)
    return res
def gp_spmm(X1, X2, X3, X4, dim;_0, dim;_1):
    X1_dl = tf.experimental.dlpack.to_dlpack(X1)
    X2_dl = tf.experimental.dlpack.to_dlpack(X2)
    X3_dl = tf.experimental.dlpack.to_dlpack(X3)
    X4_dl = tf.experimental.dlpack.to_dlpack(X4)
    #declare the output tensor here
    res = tf.zeros([dim_0, dim_1])
    res_dl = tf.experimental.dlpack.to_dlpack(res)
    gpk.spmm(X1_dl, X2_dl, X3_dl, X4_dl, res_dl)
    return res
