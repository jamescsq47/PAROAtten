#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor decode_mha_with_async_softmax(at::Tensor Q, at::Tensor K, at::Tensor V, 
        const int max_len, const int len, const float scale, const float attn_max);
at::Tensor decode_mha_fall_back(at::Tensor Q, at::Tensor K, at::Tensor V, 
        const int max_len, const int len, const float scale);
at::Tensor decode_mqa_with_async_softmax(at::Tensor Q, at::Tensor K, at::Tensor V, 
        const int max_len, int len, const double scale, const float attn_max);
at::Tensor decode_mqa_fall_back(at::Tensor Q, at::Tensor K, at::Tensor V, 
        const int max_len, int len, const double scale);