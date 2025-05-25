#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

void rope_permutation(at::Tensor Q, at::Tensor K, at::Tensor freq);

std::tuple<at::Tensor, at::Tensor> fused_apply_rotary_cuda(
    at::Tensor q, at::Tensor k, 
    const at::Tensor& cos,
    const at::Tensor& sin,
    int F, int H, int W,
    const std::string& pattern
);