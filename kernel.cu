#include "kernel.h"
#include <iostream>
#include <math.h>
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
void gspmm(int32_t *ptrs, int32_t*dsts, int32_t *degree, float *input1, int32_t N, int32_t F, float *output){
    int rowIdx = blockIdx.x;
    int colIdx = threadIdx.x;
    int colStep = blockDim.x;
    int offset = ptrs[rowIdx], boundary = ptrs[rowIdx+1];
    for(int k = offset; k < boundary; k ++){
        float ni = 1/sqrt(1.0*degree[rowIdx]);
        int dst = dsts[k];
        float nj = degree[dst]?1/sqrt(1.0*degree[dst]):0;
        for(int j = colIdx; j < F; j += colStep){
            output[rowIdx*F+j] += ni*nj*input1[dst*F+j];
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
void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output){
    int32_t N = graph.vcount;
    int32_t F = input1.col_count;
    int32_t threadsPerBlock = 64;
    int32_t blocks = N;
    gspmm<<<blocks, threadsPerBlock>>>(graph.offset, graph.nebrs, graph.dgrs, input1.data_ptr, N, F, output.data_ptr);
    cudaDeviceSynchronize();
}
