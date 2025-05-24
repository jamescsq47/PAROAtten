#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

void rope_permutation(at::Tensor Q, at::Tensor K, at::Tensor freq);