#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor rmsnorm(at::Tensor X, at::Tensor RW);
at::Tensor layernorm(at::Tensor X, at::Tensor RW, at::Tensor RB);
at::Tensor reorder_layernorm(at::Tensor X, at::Tensor RW, at::Tensor RB, at::Tensor pattern, int F,int H,int W,int head_dim);
at::Tensor inv_reorder_layernorm(at::Tensor X,at::Tensor RW,at::Tensor RB,at::Tensor pattern,int F,int H,int W,int head_dim);
void residual_rmsnorm(at::Tensor R, at::Tensor X, at::Tensor RW);
std::tuple<at::Tensor, at::Tensor> residual_layernorm(at::Tensor R, at::Tensor X, at::Tensor RW, at::Tensor RB);
void residual_rmsnorm_test(at::Tensor R, at::Tensor X, at::Tensor RW);
