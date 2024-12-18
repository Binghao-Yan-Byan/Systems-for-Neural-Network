#pragma once
#include "csr.h"
#include "op.h"

void gemm(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output);
void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output, uintptr_t stream_handle);