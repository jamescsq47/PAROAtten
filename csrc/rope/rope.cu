#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "rope.cuh"

std::tuple<at::Tensor, at::Tensor> rope_permutation(at::Tensor Q, at::Tensor K, at::Tensor freq, bool permutation){

  // Q: [bs, H, W, dim]
  int bs = Q.size(0);
  int H = Q.size(1);
  int W = Q.size(2);
  int dim = Q.size(3);
  int hs = 128;

  at::Tensor QO = torch::empty({bs, H*W, dim}, 
        at::device(Q.device()).dtype(at::ScalarType::Half));
  at::Tensor KO = torch::empty({bs, H*W, dim}, 
        at::device(Q.device()).dtype(at::ScalarType::Half));

  if (permutation){
    rope_permutation_kernel<<<dim3(W, H, bs), dim3(128, 1)>>>(
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
            reinterpret_cast<half *>(K.data_ptr<at::Half>()),
            reinterpret_cast<half *>(QO.data_ptr<at::Half>()),
            reinterpret_cast<half *>(KO.data_ptr<at::Half>()),
            freq.data_ptr<float>(), 
            bs, H, W, dim, hs
        );
  } else {
    rope_kernel<<<dim3(W, H, bs), dim3(128, 1)>>>(
            reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
            reinterpret_cast<half *>(K.data_ptr<at::Half>()),
            reinterpret_cast<half *>(QO.data_ptr<at::Half>()),
            reinterpret_cast<half *>(KO.data_ptr<at::Half>()),
            freq.data_ptr<float>(), 
            bs, H, W, dim, hs
        );
  }

  return {QO, KO};
}