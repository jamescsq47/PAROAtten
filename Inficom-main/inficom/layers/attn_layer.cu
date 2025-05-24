#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "../kernels/decode_attn.cuh"
#include "../kernels/fast_gemv.cuh"
#include "../kernels/fast_gemv_ext.cuh"
#include "../kernels/norm.cuh"
#include "../kernels/flat_gemm.cuh"
#include "../kernels/flat_gemm_ext.cuh"


at::Tensor llama2_attn_layer_fwd(at::Tensor X, at::Tensor RW, at::Tensor WQ, at::Tensor WK, 
            at::Tensor WV, at::Tensor OW, at::Tensor K, at::Tensor V, at::Tensor freq, 
            const int max_len, int len, const float scale, const float attn_max){

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

    at::Tensor RX = torch::empty({bs, 1, hn * hs}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    at::Tensor Q = torch::empty({bs, 1, hn * hs}, 
        at::device(X.device()).dtype(at::ScalarType::Half));
    
    at::Tensor O = torch::zeros({bs, 1, hn * hs}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    // RMSNorm
    rmsnorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RX.data_ptr<at::Half>()),
        bs, dim  
    );

    // QKV Projection + RoPE
    if (bs == 1){
        fast_gemv_acc_fp16_with_llama2_rope_kernel<<<dim3(1, 3 * dim / 2), dim3(128, 1)>>>(
            reinterpret_cast<half *>(WQ.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WK.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WV.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(RX.data_ptr<at::Half>()), 
            freq.data_ptr<float>(), 
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
            reinterpret_cast<half *>(K.data_ptr<at::Half>()),
            reinterpret_cast<half *>(V.data_ptr<at::Half>()),
            len, dim, hn, hs, dim / 128);
    }
    else if (bs <= 8){
    
        flat_gemm_m16n64k128x8_db_with_llama2_rope_kernel<16, 64, 128, 136, 72>
            <<<dim3(3 * dim / 64, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(RX.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WQ.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WK.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WV.data_ptr<at::Half>()), 
            freq.data_ptr<float>(), 
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
            reinterpret_cast<half *>(K.data_ptr<at::Half>()),
            reinterpret_cast<half *>(V.data_ptr<at::Half>()),
            bs, dim, dim, max_len * dim, dim, len, hn, hs
        );

        // cudaFuncSetAttribute(flat_gemm_m16n32k256x8_db_with_llama2_rope_kernel<16, 32, 256, 264, 40>,
        //         cudaFuncAttributeMaxDynamicSharedMemorySize, 50688);
    
        //  int dsmem = 2 * (16 + 32) * (256 + 8) * sizeof(half);

        // flat_gemm_m16n32k256x8_db_with_llama2_rope_kernel<16, 32, 256, 264, 40>
        //     <<<dim3(3 * dim / 32, 1, 1), dim3(256), dsmem>>>(
        //     reinterpret_cast<half *>(RX.data_ptr<at::Half>()), 
        //     reinterpret_cast<half *>(WQ.data_ptr<at::Half>()), 
        //     reinterpret_cast<half *>(WK.data_ptr<at::Half>()), 
        //     reinterpret_cast<half *>(WV.data_ptr<at::Half>()), 
        //     freq.data_ptr<float>(), 
        //     reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
        //     reinterpret_cast<half *>(K.data_ptr<at::Half>()),
        //     reinterpret_cast<half *>(V.data_ptr<at::Half>()),
        //     bs, max_len, len, dim, hn, hs
        // );
    
    }else{
        throw std::invalid_argument("only support batchsize <= 8!");
    }

    len += 1; // KV cache len ++ with concat finished 

    // decoding attention
    at::Tensor H = torch::empty({bs, 1, hn * hs}, 
        at::device(Q.device()).dtype(at::ScalarType::Half));

    if (len >= 4096) {
        decode_mha_with_async_softmax_kernel<1024, 4><<<dim3(bs, hn, 1), dim3(1024)>>>(
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()), reinterpret_cast<half *>(K.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(V.data_ptr<at::Half>()), scale, attn_max, bs, hn, dim, max_len, len, hs, DIV_UP(len, 1024), 
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );
        // at::Tensor S = torch::zeros({bs, hn, 2}, 
        //     at::device(Q.device()).dtype(at::ScalarType::Float));
        // at::Tensor H_F = torch::empty({bs, 1, hn, 2, hs}, 
        //     at::device(Q.device()).dtype(at::ScalarType::Float));
        // decode_mha_with_splitKV_kernel<1024, 4, 2><<<dim3(bs, hn, 2), dim3(1024)>>>(
        //     reinterpret_cast<half *>(Q.data_ptr<at::Half>()), reinterpret_cast<half *>(K.data_ptr<at::Half>()),  
        //     reinterpret_cast<half *>(V.data_ptr<at::Half>()), scale, attn_max, bs, hn, dim, max_len, len, hs, DIV_UP(DIV_UP(len, 2), 1024), 
        //     S.data_ptr<float>(), H_F.data_ptr<float>()
        // );
        // decode_splitKV_scaling_kernel<2><<<dim3(bs, hn), dim3(hs)>>>(
        //     H_F.data_ptr<float>(), dim, hs, hn, S.data_ptr<float>(), reinterpret_cast<half *>(H.data_ptr<at::Half>())
        // );
    }else{
        decode_mha_fall_back_kernel<1024, 128><<<dim3(bs, hn, 1), dim3(1024)>>>(
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()), reinterpret_cast<half *>(K.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(V.data_ptr<at::Half>()), scale, bs, hn, dim, max_len, len, hs, DIV_UP(len, 1024), 
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );
    }

    if (bs == 1){

        fast_gemv_acc_fp16_kernel<<<dim3(1, dim / 2), dim3(128, 1)>>>(
            reinterpret_cast<half *>(OW.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(O.data_ptr<at::Half>()),  
            dim, 
            dim / 128);
    }
    else if (bs <= 8){

        // flat_gemm_m8n32k256x8_bz1_kernel<8, 32, 256, 264, 40><<<dim3(dim / 32, 1, 1), dim3(256)>>>(
        //     reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
        //     reinterpret_cast<half *>(OW.data_ptr<at::Half>()),  
        //     reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
        //     bs, dim, dim
        // );

        // flat_gemm_m16n32k256x8_bz1_kernel<16, 32, 256, 264, 40><<<dim3(dim / 32, 1, 1), dim3(256)>>>(
        //     reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
        //     reinterpret_cast<half *>(OW.data_ptr<at::Half>()),  
        //     reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
        //     bs, dim, dim
        // );

        cudaFuncSetAttribute(flat_gemm_m16n32k256x8_db_bz_kernel<16, 32, 256, 264, 40>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, 50688);
    
        unsigned int dsmem = 2 * (16 + 32) * (256 + 8) * sizeof(half);

        flat_gemm_m16n32k256x8_db_bz_kernel<16, 32, 256, 264, 40>
            <<<dim3(dim / 32, 1, 4), dim3(256), dsmem>>>(
                reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
                reinterpret_cast<half *>(OW.data_ptr<at::Half>()),  
                reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
                bs, dim, dim
        );
    }
    else{
        throw std::invalid_argument("only support batchsize <= 8!");
    }

    return O;
}


at::Tensor chatglm2_attn_layer_fwd(at::Tensor X, at::Tensor RW, at::Tensor WQKV, at::Tensor BQKV, at::Tensor OW,
            at::Tensor K, at::Tensor V, at::Tensor freq, const int max_len, int len, const float scale, const float attn_max){

    // X: [1, bs, dim=4096]
    // K: [max_len, bs, 2, hs=128]
    // V: [max_len, bs, 2, hs=128]
    // freq: [1, bs, hn=32, 2]
    if (X.size(1) != K.size(1) || X.size(1) != V.size(1)) {
        throw std::invalid_argument("batchsize mismatch!");
    }
    if (K.size(2) != 2 || V.size(2) != 2){
        throw std::invalid_argument("head num of KV cache must be 2!");
    }
    if (X.size(0) != 1) {
        throw std::invalid_argument("only support decoding attention!");
    }
    if (len == K.size(0)){
        throw std::invalid_argument("KV cache is full, please adjust max_seq_len to a larger number!");
    }
    
    int bs = X.size(1);
    int dim = X.size(2);
    int hs = K.size(3);
    int hn = dim / hs;
    int gn = K.size(2);
    int gs = gn * hs;

    if (hn != freq.size(2) || K.size(2) != freq.size(3)){
        throw std::invalid_argument("the given freqs_cis is of wrong shape!");
    }

    at::Tensor RX = torch::empty({1, bs, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    at::Tensor Q = torch::empty({1, bs, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));
    
    at::Tensor O = torch::empty({1, bs, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    // RMSNorm
    rmsnorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RX.data_ptr<at::Half>()),
        bs, dim  
    );

    // QKV Projection + RoPE
    if (bs == 1){
        fast_gemv_acc_fp16_with_chatglm2_rope_kernel<<<dim3(1, (dim + 2 * gs) / 2), dim3(128, 1)>>>(
            reinterpret_cast<half *>(WQKV.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(BQKV.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(RX.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(freq.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
            reinterpret_cast<half *>(K.data_ptr<at::Half>()),
            reinterpret_cast<half *>(V.data_ptr<at::Half>()),
            len, dim, hn, gn, hs, dim / 128);
    }
    else if (bs <= 16){
    
        flat_gemm_m8n32k256x8_bz1_with_chatglm2_rope_kernel<8, 32, 256, 264, 40>
            <<<dim3((dim + 2 * gs) / 32, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(RX.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WQKV.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(BQKV.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(freq.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
            reinterpret_cast<half *>(K.data_ptr<at::Half>()),
            reinterpret_cast<half *>(V.data_ptr<at::Half>()),
            bs, max_len, len, dim, hn, gn, hs
        );
    
    }else{
        throw std::invalid_argument("only support batchsize <= 16!");
    }

    len += 1; // KV cache len ++ with concat finished 

    // decoding attention
    at::Tensor H = torch::empty({1, bs, dim}, 
        at::device(Q.device()).dtype(at::ScalarType::Half));

    if (len >= 4096) {
        decode_mqa_with_async_softmax_kernel<1024, 4><<<dim3(bs, hn, 1), dim3(1024)>>>(
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()), reinterpret_cast<half *>(K.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(V.data_ptr<at::Half>()), scale, attn_max, bs, hn, gn, dim, max_len, len, hs, DIV_UP(len, 1024), 
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );
    }else{
        decode_mqa_fall_back_kernel<1024, 128, 4><<<dim3(bs, hn, 1), dim3(1024)>>>(
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()), reinterpret_cast<half *>(K.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(V.data_ptr<at::Half>()), scale, bs, hn, gn, dim, max_len, len, hs, DIV_UP(len, 1024), 
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );
    }

    if (bs == 1){
        fast_gemv_acc_fp16_kernel<<<dim3(1, dim / 2), dim3(128, 1)>>>(
            reinterpret_cast<half *>(OW.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(O.data_ptr<at::Half>()),  
            dim, 
            dim / 128);
    }
    else if (bs <= 16){


        flat_gemm_m8n32k256x8_bz1_kernel<8, 32, 256, 264, 40><<<dim3(dim / 32, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(OW.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
            bs, dim, dim
        );

        // please use an extra linear layer with python code for output projection computation
    }
    else{
        throw std::invalid_argument("only support batchsize <= 16!");
    }

    return O;
}


at::Tensor opt_attn_layer_fwd(at::Tensor X, at::Tensor RW, at::Tensor WQ, at::Tensor WK, at::Tensor WV, at::Tensor OW,
            at::Tensor RB, at::Tensor BQ, at::Tensor BK, at::Tensor BV, at::Tensor OB,
            at::Tensor K, at::Tensor V, const int max_len, int len, const float scale, const float attn_max){

    // X: [bs, 1, dim]
    // K: [bs, max_len, hn, hs]
    // V: [bs, max_len, hn, hs]
    if (X.size(0) > K.size(0) || X.size(0) > V.size(0)) {
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

    unsigned int num_per_thread = dim / 128;

    at::Tensor RX = torch::empty({bs, 1, hn * hs}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    at::Tensor Q = torch::empty({bs, 1, hn * hs}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    at::Tensor O = torch::empty({bs, 1, hn * hs}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

    // Norm 
    layernorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RB.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RX.data_ptr<at::Half>()),
        bs, dim  
    );

    // QKV Projection
    if (bs == 1){
        fast_gemv_acc_fp16_for_qkv_proj_kernel<<<dim3(1, 3 * dim / 2), dim3(128, 1)>>>(
            reinterpret_cast<half *>(WQ.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WK.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WV.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(BQ.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(BK.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(BV.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(RX.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
            reinterpret_cast<half *>(K.data_ptr<at::Half>()),
            reinterpret_cast<half *>(V.data_ptr<at::Half>()),
            len, dim, hn, hs, num_per_thread);
    }
    else if (bs <= 16){ 

        flat_gemm_m8n32k256x8_bz1_for_qkv_proj_kernel<8, 32, 256, 264, 40>
            <<<dim3(3 * dim / 32, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(RX.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WQ.data_ptr<at::Half>()),   
            reinterpret_cast<half *>(WK.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(WV.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
            reinterpret_cast<half *>(K.data_ptr<at::Half>()),
            reinterpret_cast<half *>(V.data_ptr<at::Half>()),
            reinterpret_cast<half *>(BQ.data_ptr<at::Half>()),   
            reinterpret_cast<half *>(BK.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(BV.data_ptr<at::Half>()), 
            bs, max_len, len, dim, hn, hs
        );
    }else{
        throw std::invalid_argument("only support batchsize <= 16!");
    }

    len += 1; // KV cache len ++ with concat finished 

    // decoding attention
    at::Tensor H = torch::empty({bs, 1, hn * hs}, 
        at::device(Q.device()).dtype(at::ScalarType::Half));

    if (len >= 4096) {
        decode_mha_with_async_softmax_kernel<1024, 4><<<dim3(bs, hn, 1), dim3(1024)>>>(
                reinterpret_cast<half *>(Q.data_ptr<at::Half>()), reinterpret_cast<half *>(K.data_ptr<at::Half>()),  
                reinterpret_cast<half *>(V.data_ptr<at::Half>()), scale, attn_max, bs, hn, dim, max_len, len, hs, DIV_UP(len, 1024), 
                reinterpret_cast<half *>(H.data_ptr<at::Half>())
            );
    }else{
        decode_mha_fall_back_kernel<1024, 128><<<dim3(bs, hn, 1), dim3(1024)>>>(
                reinterpret_cast<half *>(Q.data_ptr<at::Half>()), reinterpret_cast<half *>(K.data_ptr<at::Half>()),  
                reinterpret_cast<half *>(V.data_ptr<at::Half>()), scale, bs, hn, dim, max_len, len, hs, DIV_UP(len, 1024), 
                reinterpret_cast<half *>(H.data_ptr<at::Half>())
            );
    }

    if (bs == 1){
        fast_gemv_acc_fp16_with_bias_kernel<<<dim3(1, dim / 2), dim3(128, 1)>>>(
            reinterpret_cast<half *>(OW.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(OB.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(O.data_ptr<at::Half>()),  
            dim, 
            num_per_thread);
    }
    else if (bs <= 16){

        flat_gemm_m8n32k256x8_bz1_with_bias_kernel<8, 32, 256, 264, 40>
            <<<dim3(dim / 32, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(H.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(OW.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(OB.data_ptr<at::Half>()), 
            bs, dim, dim
        );
    }
    else{
        throw std::invalid_argument("only support batchsize <= 16!");
    }

    return O;
}