#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "../../kernels/residual.cuh"

at::Tensor add_residual(at::Tensor R, at::Tensor X){

  // X: [bs, 1, dim]
  int bs = X.size(0);
  int dim = X.size(2);

  at::Tensor O = torch::empty({bs, 1, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

  residual_kernel<<<dim3(DIV_UP(dim, 256), bs), dim3(128, 1)>>>(
            reinterpret_cast<half *>(X.data_ptr<at::Half>()),
            reinterpret_cast<half *>(R.data_ptr<at::Half>()),
            bs, dim,
            reinterpret_cast<half *>(O.data_ptr<at::Half>())
        );

  return O;
}
