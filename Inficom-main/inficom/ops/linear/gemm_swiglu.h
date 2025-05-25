#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor dual_linear_silu_dot_fwd(at::Tensor X, at::Tensor W1, at::Tensor W2);