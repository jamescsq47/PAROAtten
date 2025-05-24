#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "../kernels/fast_gemv_ext.cuh"
#include "../kernels/flat_gemm_ext.cuh"
#include "../kernels/norm.cuh"
#include "../kernels/gemm_swiglu.cuh"

at::Tensor llama2_ffn_layer_fwd(
    at::Tensor R, at::Tensor X, at::Tensor RW, at::Tensor W1, at::Tensor W2, at::Tensor W3){

    // R: [bs, 1, dim]
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
    
    at::Tensor I = torch::empty({bs, 1, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    at::Tensor RO = torch::empty({bs, 1, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));
    
    at::Tensor O = torch::empty({bs, 1, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    residual_rmsnorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(R.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        bs, dim, 
        reinterpret_cast<half *>(I.data_ptr<at::Half>()),        
        reinterpret_cast<half *>(RO.data_ptr<at::Half>())                     
    );

    if (bs == 1){
        dual_fast_gemv_acc_fp16_silu_dot_kernel<<<dim3(h_dim), dim3(128, 1)>>>(
            reinterpret_cast<half *>(I.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(W1.data_ptr<at::Half>()),
            reinterpret_cast<half *>(W2.data_ptr<at::Half>()), 
            bs, dim, h_dim,
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );
    }
    else if (bs <= 8){ // bs <= 16

        // dual_flat_gemm_m8n32k256x8_bz1_silu_dot_kernel<8, 32, 256, 264, 40>
        //     <<<dim3(h_dim / 32, 1, 1), dim3(256)>>>(
        //     reinterpret_cast<half *>(I.data_ptr<at::Half>()), 
        //     reinterpret_cast<half *>(W1.data_ptr<at::Half>()),
        //     reinterpret_cast<half *>(W2.data_ptr<at::Half>()), 
        //     reinterpret_cast<half *>(H.data_ptr<at::Half>()),
        //     bs, h_dim, dim
        // );

        dual_flat_gemm_m16n32k256x8_bz1_silu_dot_kernel<16, 32, 256, 264, 40>
            <<<dim3(h_dim / 32, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(I.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(W1.data_ptr<at::Half>()),
            reinterpret_cast<half *>(W2.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(H.data_ptr<at::Half>()),
            bs, h_dim, dim
        );
    }
    else{
        throw std::invalid_argument("only support batchsize <= 8!");
    }

    if (bs == 1){

        fast_gemv_acc_fp16_residual_kernel<<<dim3(1, dim / 2), dim3(128, 1)>>>(
            reinterpret_cast<half *>(W3.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(RO.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(O.data_ptr<at::Half>()),  
            h_dim, 
            h_dim / 128);
    }
    else if (bs <= 8){

        // flat_gemm_m8n32k256x8_bz1_residual_kernel<8, 32, 256, 264, 40>
        //     <<<dim3(dim / 32, 1, 1), dim3(256)>>>(
        //     reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
        //     reinterpret_cast<half *>(W3.data_ptr<at::Half>()), 
        //     reinterpret_cast<half *>(RO.data_ptr<at::Half>()),  
        //     reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
        //     bs, dim, h_dim
        // );

        // if (h_dim == 11008){
        //     flat_gemm_m16n32k256x8_bz1_residual_kernel<16, 32, 256, 264, 40>
        //         <<<dim3(dim / 32, 1, 1), dim3(256)>>>(
        //         reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
        //         reinterpret_cast<half *>(W3.data_ptr<at::Half>()),  
        //         reinterpret_cast<half *>(RO.data_ptr<at::Half>()),
        //         reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
        //         bs, dim, h_dim
        //     );
        // }
        // else{

            cudaFuncSetAttribute(flat_gemm_m16n32k256x8_db_residual_kernel<16, 32, 256, 264, 40>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, 50688);
    
            unsigned int dsmem = 2 * (16 + 32) * (256 + 8) * sizeof(half);

            flat_gemm_m16n32k256x8_db_residual_kernel<16, 32, 256, 264, 40><<<dim3(dim / 32, 1, 1), dim3(256), dsmem>>>(
                reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
                reinterpret_cast<half *>(W3.data_ptr<at::Half>()),  
                reinterpret_cast<half *>(RO.data_ptr<at::Half>()),
                reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
                bs, dim, h_dim
            );
        // }
    }
    else{
        throw std::invalid_argument("only support batchsize <= 8!");
    }

    return O;
}


at::Tensor chatglm2_ffn_layer_fwd(
    at::Tensor R, at::Tensor X, at::Tensor RW, at::Tensor W1, at::Tensor W2){

    // R: [1, bs, dim]
    // X: [1, bs, dim]
    // W1: [2 * h_dim, dim]
    // W2: [dim, h_dim]
    if (X.size(2) != W1.size(1) || X.size(2) != W2.size(0)) {
        throw std::invalid_argument("hidden mismatch!");
    }
    
    int bs = X.size(1);
    int dim = X.size(2);
    int h_dim = W1.size(0) / 2;

    at::Tensor H = torch::empty({1, bs, h_dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));
    
    at::Tensor I = torch::empty({1, bs, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    at::Tensor RO = torch::empty({1, bs, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));
    
    at::Tensor O = torch::empty({1, bs, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    
    residual_rmsnorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(R.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        bs, dim, 
        reinterpret_cast<half *>(I.data_ptr<at::Half>()),        
        reinterpret_cast<half *>(RO.data_ptr<at::Half>())                     
    );

    if (bs == 1){
        fast_gemv_acc_fp16_swiglu_kernel<<<dim3(h_dim), dim3(128, 1)>>>(
            reinterpret_cast<half *>(I.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(W1.data_ptr<at::Half>()),
            bs, dim, h_dim,
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );
    }
    else if (bs <= 8){ // bs <= 16

        flat_gemm_m8n32k256x8_bz1_swiglu_kernel<8, 32, 256, 264, 40>
            <<<dim3(h_dim / 32, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(I.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(W1.data_ptr<at::Half>()),
            reinterpret_cast<half *>(H.data_ptr<at::Half>()),
            bs, h_dim, dim
        );
    }
    else{
        throw std::invalid_argument("only support batchsize <= 8!");
    }
    
    if (bs == 1){
        fast_gemv_acc_fp16_residual_kernel<<<dim3(1, dim / 2), dim3(128, 1)>>>(
            reinterpret_cast<half *>(W2.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(RO.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(O.data_ptr<at::Half>()),  
            h_dim, 
            h_dim / 128);
    }
    else if (bs <= 8){

        flat_gemm_m8n32k256x8_bz1_residual_kernel<8, 32, 256, 264, 40>
            <<<dim3(dim / 32, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(W2.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(RO.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
            bs, dim, h_dim
        );
    }
    else{
        throw std::invalid_argument("only support batchsize <= 8!");
    }

    return O;
}


at::Tensor opt_ffn_layer_fwd(
    at::Tensor R, at::Tensor X, at::Tensor RW, at::Tensor W1, at::Tensor W2, 
    at::Tensor RB, at::Tensor B1, at::Tensor B2){

    // R: [bs, 1, dim]
    // X: [bs, 1, dim]
    // W1: [h_dim, dim]
    // W2: [dim, h_dim]
    if (X.size(2) != W1.size(1)) {
        throw std::invalid_argument("hidden mismatch!");
    }
    if (X.size(0) > 16) {
        throw std::invalid_argument("only batchsize <= 16 is supported!");
    }
    
    int bs = X.size(0);
    int dim = X.size(2);
    int h_dim = W1.size(0);

    at::Tensor H = torch::empty({bs, 1, h_dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    at::Tensor RO = torch::empty({bs, 1, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));
    
    at::Tensor O = torch::empty({bs, 1, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    dim3 block_dim(128, 1);
    dim3 grid_dim(h_dim);

    residual_layernorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(R.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RB.data_ptr<at::Half>()),
        bs, dim, 
        reinterpret_cast<half *>(RO.data_ptr<at::Half>()),
        reinterpret_cast<half *>(O.data_ptr<at::Half>())                        
    );

    if (bs == 1){
        fast_gemv_acc_fp16_bias_relu_kernel<<<dim3(h_dim), dim3(128, 1)>>>(
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(W1.data_ptr<at::Half>()),
            reinterpret_cast<half *>(B1.data_ptr<at::Half>()),
            bs, dim, h_dim, 
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );
    }
    else if (bs <= 16){ // bs <= 16

        flat_gemm_m8n32k256x8_bz_1_bias_relu_kernel<8, 32, 256, 264, 40>
            <<<dim3(h_dim / 32, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(W1.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(B1.data_ptr<at::Half>()),    
            bs, h_dim, dim 
        ); 

    }
    else{
        throw std::invalid_argument("only support batchsize <= 16!");
    }

    unsigned int num_per_thread = h_dim / 128;
    assert(num_per_thread >= 8);

    if (bs == 1){
        fast_gemv_acc_fp16_bias_residual_kernel<<<dim3(1, dim / 2), dim3(128, 1)>>>(
            reinterpret_cast<half *>(W2.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(B2.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(RO.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(O.data_ptr<at::Half>()),  
            h_dim, 
            num_per_thread);
    }
    else if (bs <= 16){

        flat_gemm_m8k32n256x8_bz1_bias_residual_kernel<8, 32, 256, 264, 40>
            <<<dim3(dim / 32, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(W2.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(RO.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(B2.data_ptr<at::Half>()), 
            bs, dim, h_dim
        );
    }
    else{
        throw std::invalid_argument("only support batchsize <= 16!");
    }

    return O;
}