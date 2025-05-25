#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/AccumulateType.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "../../kernels/rope.cuh"

constexpr int THREADS = 128;

void rope_permutation(at::Tensor Q, at::Tensor K, at::Tensor freq){

  // Q: [bs, H, W, dim]
  int bs = Q.size(0);
  int H = Q.size(1);
  int W = Q.size(2);
  int dim = Q.size(3);
  int hs = 128;

  rope_permutation_kernel<<<dim3(W, H, bs), dim3(128, 1)>>>(
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
            reinterpret_cast<half *>(K.data_ptr<at::Half>()),
            freq.data_ptr<float>(), 
            bs, H, W, dim, hs
        );
}

std::tuple<at::Tensor, at::Tensor> fused_apply_rotary_cuda(
    at::Tensor q, at::Tensor k, 
    const at::Tensor& cos,
    const at::Tensor& sin,
    int F, int H, int W,
    const std::string& pattern
) {
  auto B = q.size(0), S = q.size(1), D = q.size(2);
  TORCH_CHECK(S==F*H*W, "Sequence length (S) must equal to FxHxW!");
  static const std::vector<std::string> P = {
    "FHW","FWH","WFH","WHF","HFW","HWF"
  };
  int pid = -1;
  for(int i=0;i<6;i++){
    if (P[i]==pattern) { pid=i; break; }
  }
  TORCH_CHECK(pid>=0, "Unknown pattern!", pattern);

  at::Tensor q_out = at::empty_like(q);
  at::Tensor k_out = at::empty_like(k);
 
  dim3 blocks(S, B);
  dim3 threads(THREADS);

  reorder_rope_kernel<<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16 *>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16 *>(q_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16 *>(k_out.data_ptr<at::BFloat16>()),
            cos.data_ptr<float>(),
            sin.data_ptr<float>(), 
            B, F, H, W, D, S, 
            pid
        );

  return {q_out, k_out};
}