#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

std::tuple<at::Tensor, at::Tensor> rope_permutation(at::Tensor Q, at::Tensor K, at::Tensor freq, bool permutation);