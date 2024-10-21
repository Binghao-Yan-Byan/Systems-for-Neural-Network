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

__global__
void spmm(int *ptrs, int *dstsource, float *degree, float *features, int N, int F, float *output){
    int rowIdx = blockIdx.x;
    int colIdx = threadIdx.x;
    int colStep = blockDim.x;
    int offset = ptrs[rowIdx], boundary = ptrs[rowIdx+1];
    float ni = degree[rowIdx];
    for(int k = offset; k < offset; k ++){
        float nj = degree[dstsource[offset+k]];
        for(int j = colIdx; j < F; j += colStep){
            output[rowIdx*F+j] += ni*nj*features[rowIdx*F+j];
        }
    }
}

/*
 * input1: CSR pointer     num_nodes+1
 * input2: dstsource
 * input3: 1/sqrt(degree)
 * input4: feature matrix  num_nodes x num_features
 * output: output[i, :] = input2[i]*input2[j]*input3[j] if (i, j) in edges
 */
void spmm(array1d_t<int>& input1, array1d_t<int>& input2, array1d_t<float>& input3, array2d_t<float>& input4, array2d_t<float>& output){
    int N = input4.row_count;
    int F = input4.col_count;
    int threadsPerBlock = 64;
    int blocks = N;
    spmm<<<blocks, threadsPerBlock>>>(input1.data_ptr, input2.data_ptr, input3.data_ptr, input4.data_ptr, N, F, output.data_ptr);
    cudaDeviceSynchronize();
}