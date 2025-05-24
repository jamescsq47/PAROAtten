#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor apply_silu(at::Tensor X);