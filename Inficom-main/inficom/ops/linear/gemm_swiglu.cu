#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "../../kernels/gemm_swiglu.cuh"

at::Tensor dual_linear_silu_dot_fwd(
    at::Tensor X, at::Tensor W1, at::Tensor W2){
    // X: [bs, 1, dim]
    // W1: [h_dim, dim]
    // W2: [h_dim, dim]
    if (X.size(2) != W1.size(1) || X.size(2) != W2.size(1)) {
        throw std::invalid_argument("hidden mismatch!");
    }
    
    int bs = X.size(0);
    int dim = X.size(2);
    int h_dim = W1.size(0);

    at::Tensor H = torch::empty({bs, 1, h_dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    if (bs == 1){
        dual_fast_gemv_acc_fp16_silu_dot_kernel<<<dim3(h_dim), dim3(128, 1)>>>(
            reinterpret_cast<half *>(X.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(W1.data_ptr<at::Half>()),
            reinterpret_cast<half *>(W2.data_ptr<at::Half>()), 
            bs, dim, h_dim,
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );
    }
    else if (bs <= 16){ // bs <= 16

        dual_flat_gemm_m8n32k256x8_bz1_silu_dot_kernel<8, 32, 256, 264, 40>
            <<<dim3(h_dim / 32, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(X.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(W1.data_ptr<at::Half>()),
            reinterpret_cast<half *>(W2.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(H.data_ptr<at::Half>()),
            bs, h_dim, dim
        );
    }
    else{
        throw std::invalid_argument("only support batchsize <= 16!");
    }

    return H;
}


