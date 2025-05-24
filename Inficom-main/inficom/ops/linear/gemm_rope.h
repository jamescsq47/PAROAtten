#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor attn_proj_rope_kv_cat_fwd(at::Tensor X, at::Tensor WQ, at::Tensor WK, 
            at::Tensor WV, at::Tensor K, at::Tensor V, at::Tensor freq, const int max_len, int len);