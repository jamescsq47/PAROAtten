#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "utils.h"

#define SHARED_MEM_MAX_ROWS 64

/*
    GEMV kernel with RoPE defined in Llama2, using FP16 to accumulate.
*/
__global__ __forceinline__ void fast_gemv_acc_fp16_with_llama2_rope_kernel(
                          half* WQ, half* WK, half* WV, half* X, 
                          float* freq, half* Q, half* K, half *V, int len, 
                          int dim, int hn, int hs, unsigned int num_per_thread) {
  
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y) << 1;
  half* weight_ptr = (row < dim) ? &WQ[0] : 
                    (row < 2 * dim) ? &WK[0] : &WV[0];
  unsigned int start_idx = threadIdx.x;
  half2 vec_val[4];
  half2 mat_val[8];

  // half2 temp_sum = {__float2half(0.0f), __float2half(0.0f)};
  half2 sum[2];
  sum[0] = {__float2half(0.0f), __float2half(0.0f)};
  sum[1] = {__float2half(0.0f), __float2half(0.0f)};
  float2 gsum;

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = (start_idx + iter * blockDim.x) << 3;
    if (j >= dim) {break;}
    *(float4*)(&vec_val[0]) = *(float4*)(&X[j]);
    *(float4*)(&mat_val[0]) = *(float4*)(weight_ptr + (row % dim) * dim + j);
    *(float4*)(&mat_val[4]) = *(float4*)(weight_ptr + ((row + 1) % dim) * dim + j);

    sum[0] = __hadd2(sum[0], __hmul2(vec_val[0], mat_val[0]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[1], mat_val[1]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[2], mat_val[2]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[3], mat_val[3]));

    sum[1] = __hadd2(sum[1], __hmul2(vec_val[0], mat_val[4]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[1], mat_val[5]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[2], mat_val[6]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[3], mat_val[7]));
  }

  gsum.x = __half2float(sum[0].x) + __half2float(sum[0].y);
  gsum.y = __half2float(sum[1].x) + __half2float(sum[1].y);

  static __shared__ float warpLevelSums[WARP_SIZE];

  gsum.x = blockReduceSum(gsum.x, warpLevelSums);
  gsum.y = blockReduceSum(gsum.y, warpLevelSums);

  if (tid > 0) {return;}
  // RoPE
  if (row < dim) {
    int idx = row % hs;
    float2 to_rotate = *(float2*)(&freq[idx]);
    float2 gres;
    gres.x = to_rotate.x * gsum.x - to_rotate.y * gsum.y;
    gres.y = to_rotate.x * gsum.y + to_rotate.y * gsum.x;
    *(half2*)(&Q[row]) = __float22half2_rn(gres);
  }
  else if (row < 2 * dim){
    int idx = row % hs;
    float2 to_rotate = *(float2*)(&freq[idx]);
    float2 gres;
    gres.x = to_rotate.x * gsum.x - to_rotate.y * gsum.y;
    gres.y = to_rotate.x * gsum.y + to_rotate.y * gsum.x;
    *(half2*)(&K[len * dim + (row % dim)]) = __float22half2_rn(gres);
  }
  else if (row < 3 * dim){
    *(half2*)(&V[len * dim + (row % dim)]) = __float22half2_rn(gsum);
  }
}


/*
    GEMV kernel with RoPE defined in ChatGLM2, using FP16 to accumulate.
*/
__global__ __forceinline__ void fast_gemv_acc_fp16_with_chatglm2_rope_kernel(
                          half* WQKV, half* BQKV, half* X, 
                          half* freq, half* Q, half* K, half *V, int len, 
                          int dim, int hn, int gn, int hs, unsigned int num_per_thread) {
  
  // [hongke @ 10.24: for GLM2-6B
  //    dim = 4096, hn = 32, gn = 2, hs = 128]
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y) << 1;
  unsigned int start_idx = threadIdx.x;
  half2 vec_val[4];
  half2 mat_val[8];

  // half2 temp_sum = {__float2half(0.0f), __float2half(0.0f)};
  half2 sum[2];
  sum[0] = {__float2half(0.0f), __float2half(0.0f)};
  sum[1] = {__float2half(0.0f), __float2half(0.0f)};
  float2 gsum;

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = (start_idx + iter * blockDim.x) << 3;
    if (j >= dim) {break;}
    *(float4*)(&vec_val[0]) = *(float4*)(&X[j]);
    *(float4*)(&mat_val[0]) = *(float4*)(&WQKV[row * dim + j]);
    *(float4*)(&mat_val[4]) = *(float4*)(&WQKV[(row + 1) * dim + j]);

    sum[0] = __hadd2(sum[0], __hmul2(vec_val[0], mat_val[0]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[1], mat_val[1]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[2], mat_val[2]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[3], mat_val[3]));

    sum[1] = __hadd2(sum[1], __hmul2(vec_val[0], mat_val[4]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[1], mat_val[5]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[2], mat_val[6]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[3], mat_val[7]));
  }

  gsum.x = __half2float(sum[0].x) + __half2float(sum[0].y);
  gsum.y = __half2float(sum[1].x) + __half2float(sum[1].y);

  static __shared__ float warpLevelSums[WARP_SIZE];

  gsum.x = blockReduceSum(gsum.x, warpLevelSums);
  gsum.y = blockReduceSum(gsum.y, warpLevelSums);

  if (tid > 0) {return;}

  // add bias
  half2 bias = *(half2*)(&BQKV[row]);
  gsum.x += __half2float(bias.x);
  gsum.y += __half2float(bias.y);
  // RoPE
  if (row < dim) {
    int idx = row % hs;
    if (idx < hs / 2){
        float2 update_sum;
        float2 to_rotate = __half22float2(*(half2*)(&freq[idx]));
        update_sum.x = to_rotate.x * gsum.x - to_rotate.y * gsum.y;
        update_sum.y = to_rotate.x * gsum.y + to_rotate.y * gsum.x;
        *(half2*)(&Q[row]) = __float22half2_rn(update_sum);
    }
    else{
      *(half2*)(&Q[row]) = __float22half2_rn(gsum);
    }
  }
  else if (row < (dim + gn * hs)){
    int idx = row % hs;
    if (idx < hs / 2){
        float2 update_sum;
        float2 to_rotate = __half22float2(*(half2*)(&freq[idx]));
        update_sum.x = to_rotate.x * gsum.x - to_rotate.y * gsum.y;
        update_sum.y = to_rotate.x * gsum.y + to_rotate.y * gsum.x;
        *(half2*)(&K[len * (gn * hs) + (row % dim)]) = __float22half2_rn(update_sum);
    }
    else{
      *(half2*)(&K[len * (gn * hs) + (row % dim)]) = __float22half2_rn(gsum);
    }
  }
  else if (row < (dim + 2 * gn * hs)){
    *(half2*)(&V[len * (gn * hs) + (row % dim % (gn * hs))]) = __float22half2_rn(gsum);
  }
}


/*
    GEMV kernel using FP16 to accumulate. With adding bias fused.
*/
__global__ __forceinline__ void fast_gemv_acc_fp16_with_bias_kernel(
            half* mat, half* b, half* vec, half* res, 
            unsigned int n, unsigned int num_per_thread) {
  
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

  half2 bias = *(half2*)(&b[row]);

  if (tid == 0) {
    *(half2*)(&res[row]) = __hadd2(gsum, bias);
  }
}


/*
    GEMV kernel using FP16 to accumulate. For QKV Proj with separate weights and bias. 
*/
__global__ __forceinline__ void fast_gemv_acc_fp16_for_qkv_proj_kernel(
                          half* WQ, half* WK, half* WV, half* BQ, half* BK, half* BV, half* X, 
                          half* Q, half* K, half *V, int len, 
                          int dim, int hn, int hs, unsigned int num_per_thread) {
  
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y) << 1;
  half* weight_ptr = (row < dim) ? &WQ[0] : 
                    (row < 2 * dim) ? &WK[0] : &WV[0];
  half* bias_ptr = (row < dim) ? &BQ[0] : 
                    (row < 2 * dim) ? &BK[0] : &BV[0];
  unsigned int start_idx = threadIdx.x;
  half2 vec_val[4];
  half2 mat_val[8];

  half2 sum[2];
  sum[0] = {__float2half(0.0f), __float2half(0.0f)};
  sum[1] = {__float2half(0.0f), __float2half(0.0f)};
  float2 gsum;

#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 8); iter++) {
    unsigned int j = (start_idx + iter * blockDim.x) << 3;
    if (j >= dim) {break;}
    *(float4*)(&vec_val[0]) = *(float4*)(&X[j]);
    *(float4*)(&mat_val[0]) = *(float4*)(weight_ptr + (row % dim) * dim + j);
    *(float4*)(&mat_val[4]) = *(float4*)(weight_ptr + ((row + 1) % dim) * dim + j);

    sum[0] = __hadd2(sum[0], __hmul2(vec_val[0], mat_val[0]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[1], mat_val[1]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[2], mat_val[2]));
    sum[0] = __hadd2(sum[0], __hmul2(vec_val[3], mat_val[3]));

    sum[1] = __hadd2(sum[1], __hmul2(vec_val[0], mat_val[4]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[1], mat_val[5]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[2], mat_val[6]));
    sum[1] = __hadd2(sum[1], __hmul2(vec_val[3], mat_val[7]));
  }

  gsum.x = __half2float(sum[0].x) + __half2float(sum[0].y);
  gsum.y = __half2float(sum[1].x) + __half2float(sum[1].y);

  static __shared__ float warpLevelSums[WARP_SIZE];

  gsum.x = blockReduceSum(gsum.x, warpLevelSums);
  gsum.y = blockReduceSum(gsum.y, warpLevelSums);

  half2 bias = *(half2*)(&bias_ptr[row % dim]);

  if (tid > 0) {return;}
  // no RoPE
  if (row < dim) {
    *(half2*)(&Q[row]) = __hadd2(__float22half2_rn(gsum), bias);
  }
  else if (row < 2 * dim){
    *(half2*)(&K[len * dim + (row % dim)]) = __hadd2(__float22half2_rn(gsum), bias);
  }
  else if (row < 3 * dim){
    *(half2*)(&V[len * dim + (row % dim)]) = __hadd2(__float22half2_rn(gsum), bias);
  }
}


__global__ __forceinline__ void fast_gemv_acc_fp16_residual_kernel(
                          half* mat, half* vec, half* r, half* res, 
                          unsigned int n, unsigned int num_per_thread) {
  
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

  half2 to_add = *(half2*)(&r[row]);

  if (tid == 0) {
    *(half2*)(&res[row]) = __hadd2(gsum, to_add);
  }
}


__global__ __forceinline__ void fast_gemv_acc_fp16_bias_relu_kernel(
                                half* x, half* w1, half* b1, 
                                int bs, int dim, int h_dim, 
                                half* res) {
  
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.x * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  half2 x_val[4];
  half2 w1_val[4];

  // half2 temp_sum = {__float2half(0.0f), __float2half(0.0f)};
  half2 temp_sum_1 = {__float2half(0.0f), __float2half(0.0f)};

#pragma unroll
  for (int iter = 0; iter < DIV_UP((dim >> 3), blockDim.x); iter++) {
    unsigned int j = (start_idx + iter * blockDim.x) << 3;
    if (j >= dim) {break;}
      
      // float4 vec_val = vec4[j];
      // float4 mat_val = mat4[row * (n >> 3) + j];
      *(float4*)(&x_val[0]) = *(float4*)(&x[j]);
      *(float4*)(&w1_val[0]) = *(float4*)(&w1[row * dim + j]);

      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[0],  w1_val[0]));  
      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[1],  w1_val[1]));
      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[2],  w1_val[2]));
      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[3],  w1_val[3])); 
  }

  float sum_1 = __half2float(__hadd(temp_sum_1.x, temp_sum_1.y));

  static __shared__ float warpLevelSums[WARP_SIZE];

  sum_1 = blockReduceSum(sum_1, warpLevelSums);

  sum_1 += __half2float(b1[row]);

  if (tid == 0) {
    if (sum_1 < 1e-5f) sum_1 = 1e-5f;
    res[row] = __float2half(sum_1);
  }
}


__global__ __forceinline__ void fast_gemv_acc_fp16_bias_residual_kernel(
                          half* mat, half* b, half* vec, half* r, half* res, 
                          unsigned int n, unsigned int num_per_thread) {
  
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

  half2 to_add = *(half2*)(&r[row]);

  half2 bias = *(half2*)(&b[row]);

  if (tid == 0) {
    *(half2*)(&res[row]) = __hadd2(__hadd2(gsum, to_add), bias);
  }
}
