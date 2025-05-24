#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "utils.h"

__global__ __forceinline__ void residual_kernel(half* x, half* r, 
                                int bs, int dim, half* ro){
  
  int bid = blockIdx.y;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int j = id << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 i_val[4];
  half2 r_val[4];
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&r_val[0]) = *(float4*)(&r[bid * dim + j]);
  // *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);

  // residual
  i_val[0] = __hadd2(x_val[0], r_val[0]);
  i_val[1] = __hadd2(x_val[1], r_val[1]);
  i_val[2] = __hadd2(x_val[2], r_val[2]);
  i_val[3] = __hadd2(x_val[3], r_val[3]);

  // store value
  *(float4*)(&ro[bid * dim + j]) = *(float4*)(&i_val[0]);
}
