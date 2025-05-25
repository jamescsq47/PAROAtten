#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "../../kernels/decode_attn.cuh"

at::Tensor decode_mha_with_async_softmax(at::Tensor Q, at::Tensor K, at::Tensor V, 
        const int max_len, const int len, const float scale, const float attn_max){

    // Q: [bs, 1, hn, hs]
    // K: [bs, max_len, hn, hs]
    // V: [bs, max_len, hn, hs]
    if (Q.size(0) != K.size(0) || Q.size(0) != V.size(0)) {
        throw std::invalid_argument("batchsize mismatch!");
    }
    if (Q.size(2) != K.size(2) || Q.size(2) != V.size(2)) {
        throw std::invalid_argument("head number mismatch!");
    }
    if (Q.size(1) != 1) {
        throw std::invalid_argument("only support decoding attention!");
    }
    if (K.size(1) != V.size(1)) {
        throw std::invalid_argument("KV cache length mismatch!");
    }
    if (Q.size(3) != K.size(3) || Q.size(3) != V.size(3)) {
        throw std::invalid_argument("head size mismatch!");
    }
    
    int bs = Q.size(0);
    int hn = Q.size(2);
    int hs = Q.size(3);
    int dim = hn * hs;

    at::Tensor H = torch::empty({bs, 1, hn, hs}, 
        at::device(Q.device()).dtype(at::ScalarType::Half));

    decode_mha_with_async_softmax_kernel<1024, 4><<<dim3(bs, hn, 1), dim3(1024)>>>(
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()), reinterpret_cast<half *>(K.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(V.data_ptr<at::Half>()), scale, attn_max, bs, hn, dim, max_len, len, hs, DIV_UP(len, 1024), 
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );
    
    return H;
}


at::Tensor decode_mha_fall_back(at::Tensor Q, at::Tensor K, at::Tensor V, 
        const int max_len, const int len, const float scale){

    // Q: [bs, 1, hn, hs]
    // K: [bs, max_len, hn, hs]
    // V: [bs, max_len, hn, hs]
    if (Q.size(0) != K.size(0) || Q.size(0) != V.size(0)) {
        throw std::invalid_argument("batchsize mismatch!");
    }
    if (Q.size(2) != K.size(2) || Q.size(2) != V.size(2)) {
        throw std::invalid_argument("head number mismatch!");
    }
    if (Q.size(1) != 1) {
        throw std::invalid_argument("only support decoding attention!");
    }
    if (K.size(1) != V.size(1)) {
        throw std::invalid_argument("KV cache length mismatch!");
    }
    if (Q.size(3) != K.size(3) || Q.size(3) != V.size(3)) {
        throw std::invalid_argument("head size mismatch!");
    }
    
    int bs = Q.size(0);
    int hn = Q.size(2);
    int hs = Q.size(3);
    int dim = hn * hs;

    at::Tensor H = torch::empty({bs, 1, hn, hs}, 
        at::device(Q.device()).dtype(at::ScalarType::Half));

    decode_mha_fall_back_kernel<1024, 192><<<dim3(bs, hn, 1), dim3(1024)>>>(
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()), reinterpret_cast<half *>(K.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(V.data_ptr<at::Half>()), scale, bs, hn, dim, max_len, len, hs, DIV_UP(len, 1024), 
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );
    
    return H;
}


at::Tensor decode_mqa_with_async_softmax(at::Tensor Q, at::Tensor K, at::Tensor V, 
        const int max_len, int len, const double scale, const float attn_max){

    // Q: [1, bs, dim=4096]
    // K: [max_len, bs, 2, hs=128]
    // V: [max_len, bs, 2, hs=128]
    if (Q.size(1) != K.size(1) || Q.size(1) != V.size(1)) {
        throw std::invalid_argument("batchsize mismatch!");
    }
    if (K.size(2) != 2 || V.size(2) != 2){
        throw std::invalid_argument("head num of KV cache must be 2!");
    }
    if (Q.size(0) != 1) {
        throw std::invalid_argument("only support decoding attention!");
    }
    if (len == K.size(0)){
        throw std::invalid_argument("KV cache is full, please adjust max_seq_len to a larger number!");
    }
    
    int bs = Q.size(1);
    int dim = Q.size(2);
    int hs = Q.size(3);
    int hn = dim / hs;
    int gn = K.size(2);
    int gs = gn * hs;

    at::Tensor H = torch::empty({1, bs, dim}, 
        at::device(Q.device()).dtype(at::ScalarType::Half));

    decode_mqa_with_async_softmax_kernel<1024, 4><<<dim3(bs, hn, 1), dim3(1024)>>>(
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()), reinterpret_cast<half *>(K.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(V.data_ptr<at::Half>()), scale, attn_max, bs, hn, gn, dim, max_len, len, hs, DIV_UP(len, 1024), 
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );

    return H;
}


at::Tensor decode_mqa_fall_back(at::Tensor Q, at::Tensor K, at::Tensor V, 
        const int max_len, int len, const double scale){

    // Q: [1, bs, dim=4096]
    // K: [max_len, bs, 2, hs=128]
    // V: [max_len, bs, 2, hs=128]
    if (Q.size(1) != K.size(1) || Q.size(1) != V.size(1)) {
        throw std::invalid_argument("batchsize mismatch!");
    }
    if (K.size(2) != 2 || V.size(2) != 2){
        throw std::invalid_argument("head num of KV cache must be 2!");
    }
    if (Q.size(0) != 1) {
        throw std::invalid_argument("only support decoding attention!");
    }
    if (len == K.size(0)){
        throw std::invalid_argument("KV cache is full, please adjust max_seq_len to a larger number!");
    }
    
    int bs = Q.size(1);
    int dim = Q.size(2);
    int hs = Q.size(3);
    int hn = dim / hs;
    int gn = K.size(2);
    int gs = gn * hs;

    at::Tensor H = torch::empty({1, bs, dim}, 
        at::device(Q.device()).dtype(at::ScalarType::Half));

    decode_mqa_fall_back_kernel<1024, 128, 4><<<dim3(bs, hn, 1), dim3(1024)>>>(
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()), reinterpret_cast<half *>(K.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(V.data_ptr<at::Half>()), scale, bs, hn, gn, dim, max_len, len, hs, DIV_UP(len, 1024), 
            reinterpret_cast<half *>(H.data_ptr<at::Half>())
        );

    return H;
}