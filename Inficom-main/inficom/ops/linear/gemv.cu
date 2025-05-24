#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "../../kernels/fast_gemv.cuh"


at::Tensor gemv_acc_fp16(at::Tensor X, at::Tensor W){

    // X: [bs, 1, dim_in]
    // W: [dim_out, dim_in]

    if (X.size(2) != W.size(1)) {
        throw std::invalid_argument("embbed dim mismatch!");
    }
    if (X.size(2) % 128 != 0){
        throw std::invalid_argument("embbed dim must be a multiple of 128!");
    }
    
    int bs = X.size(0);
    int dim_in = X.size(2);
    int dim_out = W.size(0);
    
    at::Tensor O = torch::empty({bs, 1, dim_out}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    fast_gemv_acc_fp16_kernel<<<dim3(dim_out / 2), dim3(128)>>>(
            reinterpret_cast<half *>(W.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(X.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(O.data_ptr<at::Half>()),  
            dim_in, 
            DIV_UP(dim_in, 128));

    return O;
}


at::Tensor gemv_acc_fp32(at::Tensor X, at::Tensor W){

    // X: [bs, 1, dim_in]
    // W: [dim_out, dim_in]

    if (X.size(2) != W.size(1)) {
        throw std::invalid_argument("embbed dim mismatch!");
    }
    if (X.size(2) % 128 != 0){
        throw std::invalid_argument("embbed dim must be a multiple of 128!");
    }
    
    int bs = X.size(0);
    int dim_in = X.size(2);
    int dim_out = W.size(0);
    
    at::Tensor O = torch::empty({bs, 1, dim_out}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    fast_gemv_acc_fp32_kernel<<<dim3(dim_out), dim3(128)>>>(
            reinterpret_cast<half *>(W.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(X.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(O.data_ptr<at::Half>()),  
            dim_in, 
            DIV_UP(dim_in, 128));

    return O;
}

