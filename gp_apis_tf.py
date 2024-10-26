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
def gp_gspmmv(graph, X1, dim;_0, dim;_1):
    X1_dl = tf.experimental.dlpack.to_dlpack(X1)
    #declare the output tensor here
    res = tf.zeros([dim_0, dim_1])
    res_dl = tf.experimental.dlpack.to_dlpack(res)
    gpk.gspmmv(graph, X1_dl, res_dl)
    return res
