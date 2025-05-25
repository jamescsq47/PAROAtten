#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor add_residual(at::Tensor R, at::Tensor X);