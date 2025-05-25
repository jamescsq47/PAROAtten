#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor gemv_acc_fp16(at::Tensor X, at::Tensor W);
at::Tensor gemv_acc_fp32(at::Tensor X, at::Tensor W);