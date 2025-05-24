#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "../../kernels/fast_gemv_ext.cuh"
#include "../../kernels/flat_gemm_ext.cuh"

at::Tensor attn_proj_rope_kv_cat_fwd(at::Tensor X, at::Tensor WQ, at::Tensor WK, 
            at::Tensor WV, at::Tensor K, at::Tensor V, at::Tensor freq, const int max_len, int len){

    // X: [bs, 1, dim]
    // K: [bs, max_len, hn, hs]
    // V: [bs, max_len, hn, hs]
    if (X.size(0) != K.size(0) || X.size(0) != V.size(0)) {
        throw std::invalid_argument("batchsize mismatch!");
    }
    if (X.size(2) != K.size(2) * K.size(3) || X.size(2) != V.size(2) * V.size(3)) {
        throw std::invalid_argument("embbed dim mismatch!");
    }
    if (X.size(1) != 1) {
        throw std::invalid_argument("only support decoding attention!");
    }
    if (len == K.size(1)){
        throw std::invalid_argument("KV cache is full, please adjust max_new_len to a larger number!");
    }
    
    int bs = X.size(0);
    int hn = K.size(2);
    int hs = K.size(3);
    int dim = hn * hs;

    if (hs != freq.size(-1)){
        throw std::invalid_argument("the given freqs_cis is of wrong shape!");
    }

    at::Tensor Q = torch::empty({bs, 1, hn * hs}, 
        at::device(X.device()).dtype(at::ScalarType::Half));
    
    // QKV Projection + RoPE
    if (bs == 1){
        fast_gemv_acc_fp16_with_llama2_rope_kernel<<<dim3(1, 3 * dim / 2), dim3(128, 1)>>>(
            reinterpret_cast<half *>(WQ.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WK.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WV.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(X.data_ptr<at::Half>()), 
            freq.data_ptr<float>(), 
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
            reinterpret_cast<half *>(K.data_ptr<at::Half>()),
            reinterpret_cast<half *>(V.data_ptr<at::Half>()),
            len, dim, hn, hs, dim / 128);
    }
    else if (bs <= 16){

        flat_gemm_m16n64k128x8_db_with_llama2_rope_kernel<16, 64, 128, 136, 72>
            <<<dim3(3 * dim / 64, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(X.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WQ.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WK.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WV.data_ptr<at::Half>()), 
            freq.data_ptr<float>(), 
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
            reinterpret_cast<half *>(K.data_ptr<at::Half>()),
            reinterpret_cast<half *>(V.data_ptr<at::Half>()),
            bs, dim, dim, (max_len * dim), dim, 
            len, hn, hs
        );
    
    }else{
        throw std::invalid_argument("only support batchsize <= 16!");
    }

    return Q;
}
