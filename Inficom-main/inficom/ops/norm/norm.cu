#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "../../kernels/norm.cuh"

at::Tensor rmsnorm(at::Tensor X, at::Tensor RW){

  // X: [bs, 1, dim]
  if (X.size(1) != 1) {
    throw std::invalid_argument("only support decoding attention!");
  }
  
  int bs = X.size(0);
  int dim = X.size(2);

  at::Tensor RX = torch::empty({bs, 1, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

  rmsnorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RX.data_ptr<at::Half>()),
        bs, dim  
    );

  return RX;
}


at::Tensor layernorm(at::Tensor X, at::Tensor RW, at::Tensor RB){

  // X: [bs, 1, dim]
  if (X.size(1) != 1) {
    throw std::invalid_argument("only support decoding attention!");
  }
  
  int bs = X.size(0);
  int dim = X.size(2);

  at::Tensor RX = torch::empty({bs, 1, dim}, 
        at::device(X.device()).dtype(at::ScalarType::Half));

  layernorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RB.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RX.data_ptr<at::Half>()),
        bs, dim  
    );

  return RX;
}


void residual_rmsnorm(
    at::Tensor R, at::Tensor X, at::Tensor RW){
  
  // X: [bs, 1, dim]
  if (X.size(1) != 1) {
    throw std::invalid_argument("only support decoding attention!");
  }
  
  int bs = X.size(0);
  int dim = X.size(2);

  // at::Tensor RO = torch::empty({bs, 1, dim}, 
  //     at::device(X.device()).dtype(at::ScalarType::Half));
  
  // at::Tensor O = torch::empty({bs, 1, dim}, 
  //     at::device(X.device()).dtype(at::ScalarType::Half));

  residual_rmsnorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(R.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        bs, dim, 
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(R.data_ptr<at::Half>())                     
    );

  // return {RO, O};
}

void residual_rmsnorm_test(
    at::Tensor R, at::Tensor X, at::Tensor RW){
  
  // X: [bs, 1, dim]
  if (X.size(1) != 1) {
    throw std::invalid_argument("only support decoding attention!");
  }
  
  int bs = X.size(0);
  int dim = X.size(2);

  // at::Tensor RO = torch::empty({bs, 1, dim}, 
  //     at::device(X.device()).dtype(at::ScalarType::Half));
  
  // at::Tensor O = torch::empty({bs, 1, dim}, 
  //     at::device(X.device()).dtype(at::ScalarType::Half));

  residual_rmsnorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(R.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        bs, dim, 
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(R.data_ptr<at::Half>())                     
    );

  // return {RO, O};
}


std::tuple<at::Tensor, at::Tensor> residual_layernorm(
    at::Tensor R, at::Tensor X, at::Tensor RW, at::Tensor RB){
  
  // X: [bs, 1, dim]
  if (X.size(1) != 1) {
    throw std::invalid_argument("only support decoding attention!");
  }
  
  int bs = X.size(0);
  int dim = X.size(2);

  at::Tensor RO = torch::empty({bs, 1, dim}, 
      at::device(X.device()).dtype(at::ScalarType::Half));
  
  at::Tensor O = torch::empty({bs, 1, dim}, 
      at::device(X.device()).dtype(at::ScalarType::Half));

  residual_layernorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<half *>(X.data_ptr<at::Half>()),
        reinterpret_cast<half *>(R.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RW.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RB.data_ptr<at::Half>()),
        bs, dim, 
        reinterpret_cast<half *>(O.data_ptr<at::Half>()),
        reinterpret_cast<half *>(RO.data_ptr<at::Half>())                        
    );

  return {RO, O};
}



at::Tensor reorder_layernorm(at::Tensor X, at::Tensor RW, at::Tensor RB,at::Tensor pattern,int F,int H,int W,int head_dim){

  // X: [bs, text_length, dim]
  
  int bs = X.size(0);
  int text_length = X.size(1);
  int dim = X.size(2);

  at::Tensor RX = torch::empty({bs, text_length, dim}, 
        at::device(X.device()).dtype(at::ScalarType::BFloat16));

  reorder_layernorm_kernel<<<dim3(bs,text_length), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<__nv_bfloat16 *>(X.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(RW.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(RB.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(RX.data_ptr<at::BFloat16>()),
        bs, text_length, dim,F,H,W,reinterpret_cast<int*>(pattern.data_ptr<int>()),head_dim  
    );

  return RX;
}

at::Tensor inv_reorder_layernorm(at::Tensor X,at::Tensor RW,at::Tensor RB,at::Tensor pattern,int F,int H,int W,int head_dim){

  // X: [bs, text_length, dim]
  
  int bs = X.size(0);
  int text_length = X.size(1);
  int dim = X.size(2);

  at::Tensor RX = torch::empty({bs, text_length, dim}, 
        at::device(X.device()).dtype(at::ScalarType::BFloat16));

  inv_reorder_layernorm_kernel<<<dim3(bs,text_length), dim3(DIV_UP(dim, 8), 1)>>>(
        reinterpret_cast<__nv_bfloat16 *>(X.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(RW.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(RB.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(RX.data_ptr<at::BFloat16>()),
        bs, text_length, dim,F,H,W,reinterpret_cast<int*>(pattern.data_ptr<int>()),head_dim  
    );

  return RX;
}


