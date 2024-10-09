#include "kernel.h"

__global__
void gemm(float *input1, float *input2, int N, int D, int M, float *output){
    int rowIdx = blockIdx.x;
    int colIdx = threadIdx.x;
    int colStep = blockDim.x;
    for(int k = 0; k < D; k ++){
        for(int j = colIdx; j < M; j += colStep){
            output[rowIdx*M + j] += input1[rowIdx*D + k] * input2[k*M + j];
        }
    }
}

void gemm(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output){
    int N = input1.row_count;
    int D = input1.col_count;
    int M = input2.col_count;
    int threadsPerBlock = 64;
    int blocks = N;
    gemm<<<blocks, threadsPerBlock>>>(input1.data_ptr, input2.data_ptr, N, D, M, output.data_ptr);
    cudaDeviceSynchronize();
}