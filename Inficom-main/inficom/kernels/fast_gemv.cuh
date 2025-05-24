#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "utils.h"

#define SHARED_MEM_MAX_ROWS 64

/*
    GEMV kernel using FP16 to accumulate. Modified by @Infinigence.
*/
__global__ __forceinline__ void fast_gemv_acc_fp16_kernel(
                          half* mat, half* vec, half* res, unsigned int n,
                          unsigned int num_per_thread) {
  
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y) << 1;
  unsigned int start_idx = threadIdx.x;
  half2 vec_val[4];
  half2 mat_val[8];

  half2 sum[2];
  sum[0] = {__float2half(0.0f), __float2half(0.0f)};
  sum[1] = {__float2half(0.0f), __float2half(0.0f)};
  half2 gsum;

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = (start_idx + iter * blockDim.x) << 3;
    if (j >= n) {break;}
    *(float4*)(&vec_val[0]) = *(float4*)(&vec[j]);
    *(float4*)(&mat_val[0]) = *(float4*)(&mat[row * n + j]);
    *(float4*)(&mat_val[4]) = *(float4*)(&mat[(row + 1) * n + j]);

    sum[0] = __hadd2(sum[0], __hmul2(vec_val[0], mat_val[0]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[1], mat_val[1]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[2], mat_val[2]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[3], mat_val[3]));

    sum[1] = __hadd2(sum[1], __hmul2(vec_val[0], mat_val[4]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[1], mat_val[5]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[2], mat_val[6]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[3], mat_val[7]));
  }

  gsum.x = __hadd(sum[0].x, sum[0].y);
  gsum.y = __hadd(sum[1].x, sum[1].y);

  static __shared__ half warpLevelSums[WARP_SIZE];

  gsum.x = blockReduceSum(gsum.x, warpLevelSums);
  gsum.y = blockReduceSum(gsum.y, warpLevelSums);

  if (tid == 0) {
    *(half2*)(&res[row]) = gsum;
  }
}

/*
    GEMV kernel using FP32 to accumulate. This implementation comes directly from
    https://github.com/wangsiping97/FastGEMV.
*/
__global__ __forceinline__ void fast_gemv_acc_fp32_kernel(
                          half* mat, half* vec, half* res, unsigned int n, 
                          unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.x;
  // if (row >= no){return;}
  unsigned int start_idx = threadIdx.x;
  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (n >> 3) + j];
      half2* vec_h1 = (half2*)&vec_val.x;
      half2* vec_h2 = (half2*)&vec_val.y;
      half2* vec_h3 = (half2*)&vec_val.z;
      half2* vec_h4 = (half2*)&vec_val.w;
      half2* mat_h1 = (half2*)&mat_val.x;
      half2* mat_h2 = (half2*)&mat_val.y;
      half2* mat_h3 = (half2*)&mat_val.z;
      half2* mat_h4 = (half2*)&mat_val.w;
      sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
      sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
      sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
      sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
      sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
      sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
      sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
      sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);
    }
  }

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}


