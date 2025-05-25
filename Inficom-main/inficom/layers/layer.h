#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor llama2_attn_layer_fwd(at::Tensor X, at::Tensor RW, at::Tensor WQ, at::Tensor WK, 
            at::Tensor WV, at::Tensor OW, at::Tensor K, at::Tensor V, at::Tensor freq, 
            const int max_len, int len, const float scale, const float attn_max);

at::Tensor chatglm2_attn_layer_fwd(at::Tensor X, at::Tensor RW, at::Tensor WQKV, at::Tensor BQKV, 
            at::Tensor OW, at::Tensor K, at::Tensor V, at::Tensor freq, 
            const int max_len, int len, const float scale, const float attn_max);

at::Tensor opt_attn_layer_fwd(at::Tensor X, at::Tensor RW, at::Tensor WQ, at::Tensor WK, at::Tensor WV, 
            at::Tensor OW, at::Tensor RB, at::Tensor BQ, at::Tensor BK, at::Tensor BV, at::Tensor OB,
            at::Tensor K, at::Tensor V, const int max_len, int len, const float scale, const float attn_max);

at::Tensor llama2_ffn_layer_fwd(
    at::Tensor R, at::Tensor X, at::Tensor RW, at::Tensor W1, at::Tensor W2, at::Tensor W3);

at::Tensor chatglm2_ffn_layer_fwd(
    at::Tensor R, at::Tensor X, at::Tensor RW, at::Tensor W1, at::Tensor W2);

at::Tensor opt_ffn_layer_fwd(
    at::Tensor R, at::Tensor X, at::Tensor RW, at::Tensor W1, at::Tensor W2, 
    at::Tensor RB, at::Tensor B1, at::Tensor B2);
