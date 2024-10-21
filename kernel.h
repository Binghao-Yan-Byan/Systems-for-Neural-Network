#pragma once
#include "csr.h"
#include "op.h"

void gemm(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output);
void spmm(array1d_t<int>& input1, array1d_t<int>& input2, array1d_t<float>& input3, array2d_t<float>& input4, array2d_t<float>& output);