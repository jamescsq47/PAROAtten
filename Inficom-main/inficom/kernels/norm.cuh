#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "utils.h"

/*
    RMSNorm kernel.
*/
__global__ __forceinline__ void rmsnorm_kernel(
                    half* x, half* rw, half* o, int bs, int dim){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 w_val[4];
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);

  // RMSNorm (float)
#pragma unroll
  for (int i = 0; i < 4; i++){
    pow_sum += __half2float(x_val[i].x) * __half2float(x_val[i].x);
    pow_sum += __half2float(x_val[i].y) * __half2float(x_val[i].y);
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  
  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + 1e-5f);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    x_val[i].x = __float2half(__half2float(x_val[i].x) * scaling);
    x_val[i].y = __float2half(__half2float(x_val[i].y) * scaling);
  }
  x_val[0] = __hmul2(x_val[0], w_val[0]);
  x_val[1] = __hmul2(x_val[1], w_val[1]);
  x_val[2] = __hmul2(x_val[2], w_val[2]);
  x_val[3] = __hmul2(x_val[3], w_val[3]);

  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
}


/*
    LayerNorm kernel.
*/
__global__ __forceinline__ void layernorm_kernel(
                    half* x, half* rw, half* rb, half* o, int bs, int dim){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 w_val[4];
  half2 b_val[4];
  float mean_sum = 0.0f;
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);
  *(float4*)(&b_val[0]) = *(float4*)(&rb[j]);

  // Mean (float)
#pragma unroll
  for (int i = 0; i < 4; i++){
    mean_sum += __half2float(x_val[i].x);
    mean_sum += __half2float(x_val[i].y);
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  mean_sum = blockReduceSum(mean_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = __fdividef(mean_sum, (float)dim);
  }
  __syncthreads();

  // Norm (float)
  float mean = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    pow_sum += (__half2float(x_val[i].x) - mean) * (__half2float(x_val[i].x) - mean);
    pow_sum += (__half2float(x_val[i].y) - mean) * (__half2float(x_val[i].y) - mean);
  }

  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + 1e-5f);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    x_val[i].x = __float2half((__half2float(x_val[i].x) - mean) * scaling);
    x_val[i].y = __float2half((__half2float(x_val[i].y) - mean) * scaling);
  }
  x_val[0] = __hadd2(__hmul2(x_val[0], w_val[0]), b_val[0]);
  x_val[1] = __hadd2(__hmul2(x_val[1], w_val[1]), b_val[1]);
  x_val[2] = __hadd2(__hmul2(x_val[2], w_val[2]), b_val[2]);
  x_val[3] = __hadd2(__hmul2(x_val[3], w_val[3]), b_val[3]);

  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
}


/*
    add residual + RMSNorm fused kernel.
*/
__global__ __forceinline__ void residual_rmsnorm_kernel(
                                half* x, half* r, half* rw, 
                                int bs, int dim, 
                                half* o, half* ro){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 i_val[4];
  half2 r_val[4];
  half2 w_val[4];
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&r_val[0]) = *(float4*)(&r[bid * dim + j]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);

  // residual
  i_val[0] = __hadd2(x_val[0], r_val[0]);
  i_val[1] = __hadd2(x_val[1], r_val[1]);
  i_val[2] = __hadd2(x_val[2], r_val[2]);
  i_val[3] = __hadd2(x_val[3], r_val[3]);

  // store intermediate value
  *(float4*)(&ro[bid * dim + j]) = *(float4*)(&i_val[0]);

  // RMSNorm (float)
#pragma unroll
  for (int i = 0; i < 4; i++){
    pow_sum += __half2float(i_val[i].x) * __half2float(i_val[i].x);
    pow_sum += __half2float(i_val[i].y) * __half2float(i_val[i].y);
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  
  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + 1e-5f);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    x_val[i].x = __float2half(__half2float(i_val[i].x) * scaling);
    x_val[i].y = __float2half(__half2float(i_val[i].y) * scaling);
  }
  x_val[0] = __hmul2(x_val[0], w_val[0]);
  x_val[1] = __hmul2(x_val[1], w_val[1]);
  x_val[2] = __hmul2(x_val[2], w_val[2]);
  x_val[3] = __hmul2(x_val[3], w_val[3]);

  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
}


/*
    add residual + LayerNorm fused kernel.
*/
__global__ __forceinline__ void residual_layernorm_kernel(
                                half* x, half* r, half* rw, half* rb, 
                                int bs, int dim, 
                                half* o, half* ro){
  
  int bid = blockIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int j = tid << 3;

  if (j >= dim) {return;}

  half2 x_val[4];
  half2 i_val[4];
  half2 r_val[4];
  half2 w_val[4];
  half2 b_val[4];
  float mean_sum = 0.0f;
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(float4*)(&x[bid * dim + j]);
  *(float4*)(&r_val[0]) = *(float4*)(&r[bid * dim + j]);
  *(float4*)(&w_val[0]) = *(float4*)(&rw[j]);
  *(float4*)(&b_val[0]) = *(float4*)(&rb[j]);

  // residual
  i_val[0] = __hadd2(x_val[0], r_val[0]);
  i_val[1] = __hadd2(x_val[1], r_val[1]);
  i_val[2] = __hadd2(x_val[2], r_val[2]);
  i_val[3] = __hadd2(x_val[3], r_val[3]);

  // store intermediate value
  *(float4*)(&ro[bid * dim + j]) = *(float4*)(&i_val[0]);

  // Mean (float)
#pragma unroll
  for (int i = 0; i < 4; i++){
    mean_sum += __half2float(i_val[i].x);
    mean_sum += __half2float(i_val[i].y);
  }

  // block reduce to get mean
  static __shared__ float warpLevelSums[WARP_SIZE];
  mean_sum = blockReduceSum(mean_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = __fdividef(mean_sum, (float)dim);
  }
  __syncthreads();

  // Norm (float)
  float mean = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    pow_sum += (__half2float(i_val[i].x) - mean) * (__half2float(i_val[i].x) - mean);
    pow_sum += (__half2float(i_val[i].y) - mean) * (__half2float(i_val[i].y) - mean);
  }
  
  pow_sum = blockReduceSum(pow_sum, warpLevelSums);
  if (tid == 0){
    warpLevelSums[0] = rsqrtf(__fdividef(pow_sum, (float)dim) + 1e-5f);
  }
  __syncthreads();

  // normalization
  float scaling = warpLevelSums[0];
#pragma unroll
  for (int i = 0; i < 4; i++){
    x_val[i].x = __float2half((__half2float(i_val[i].x) - mean) * scaling);
    x_val[i].y = __float2half((__half2float(i_val[i].y) - mean) * scaling);
  }
  x_val[0] = __hadd2(__hmul2(x_val[0], w_val[0]), b_val[0]);
  x_val[1] = __hadd2(__hmul2(x_val[1], w_val[1]), b_val[1]);
  x_val[2] = __hadd2(__hmul2(x_val[2], w_val[2]), b_val[2]);
  x_val[3] = __hadd2(__hmul2(x_val[3], w_val[3]), b_val[3]);

  // store intermediate value
  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
}
//new start

__device__ __forceinline__ int64_t flat_FHW(int64_t f,int64_t h,int64_t w,int64_t F,int64_t H,int64_t W){return f*(H*W)+h*W+w;}
__device__ __forceinline__ int64_t flat_FWH(int64_t f,int64_t h,int64_t w,int64_t F,int64_t H,int64_t W){return f*(W*H)+w*H+h;}
__device__ __forceinline__ int64_t flat_WFH(int64_t f,int64_t h,int64_t w,int64_t F,int64_t H,int64_t W){return w*(F*H)+f*H+h;}
__device__ __forceinline__ int64_t flat_WHF(int64_t f,int64_t h,int64_t w,int64_t F,int64_t H,int64_t W){return w*(H*F)+h*F+f;}
__device__ __forceinline__ int64_t flat_HFW(int64_t f,int64_t h,int64_t w,int64_t F,int64_t H,int64_t W){return h*(F*W)+f*W+w;}
__device__ __forceinline__ int64_t flat_HWF(int64_t f,int64_t h,int64_t w,int64_t F,int64_t H,int64_t W){return h*(W*F)+w*F+f;}
__device__ __forceinline__ int64_t to_flat(int pid,int64_t f,int64_t h,int64_t w,int64_t F,int64_t H,int64_t W){
    switch(pid){  case 0: return flat_FHW(f,h,w,F,H,W);
                  case 1: return flat_FWH(f,h,w,F,H,W);
                  case 2: return flat_WFH(f,h,w,F,H,W);
                  case 3: return flat_WHF(f,h,w,F,H,W);
                  case 4: return flat_HFW(f,h,w,F,H,W);
                  case 5: return flat_HWF(f,h,w,F,H,W);
    }
    return 0;
}


__global__ __forceinline__
void reorder_layernorm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ rw, //scale
    const __nv_bfloat16* __restrict__ rb, //shift
    __nv_bfloat16*       __restrict__ o,
    int bs, int text_length, int dim,
    int F, int H, int W,
    const int*  __restrict__ pattern,
    int head_dim)
{
    // 每线程处理 8 个 half
    int tid = threadIdx.x;
    int j   = tid << 3;  // j in [0, bs*text_length*dim) step 8

    // 全局元素编号 = bid*text_length*dim + j
    int bid = blockIdx.x*text_length+blockIdx.y;
    int base_idx = bid * dim + j;
    if (j >= dim) return;

    // 1) decode token idx s 和 channel c
    int s = blockIdx.y;    // 0..text_length-1
    int c = j;    // 0..dim-1

    // 2) spatial coords
    int f   = s / (H * W);
    int rem = s % (H * W);
    int h   = rem / W;
    int w   = rem % W;

    // 3) load x, rw, rb into寄存器
    __nv_bfloat162 x_val[4], w_val[4], b_val[4];
    *(float4*)(&x_val[0]) = *(const float4*)(&x[base_idx]);
    *(float4*)(&w_val[0]) = *(const float4*)(&rw[blockIdx.x*dim+c]);
    *(float4*)(&b_val[0]) = *(const float4*)(&rb[blockIdx.x*dim+c]);

    // 4) 计算 mean
    float mean_sum = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      mean_sum += __bfloat162float(x_val[i].x);
      mean_sum += __bfloat162float(x_val[i].y);
    }
    __shared__ float warpSums[WARP_SIZE];  // WARP_SIZE 元素
    mean_sum = blockReduceSum(mean_sum, warpSums);
    if (tid == 0) {
      warpSums[0] = mean_sum / float(dim);
    }
    __syncthreads();
    float mean = warpSums[0];

    // 5) 计算 variance + rsqrt
    float var_sum = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float dx = __bfloat162float(x_val[i].x) - mean;
      float dy = __bfloat162float(x_val[i].y) - mean;
      var_sum += dx*dx + dy*dy;
    }
    var_sum = blockReduceSum(var_sum, warpSums);
    if (tid == 0) {
      warpSums[0] = rsqrtf(var_sum / float(dim) + 1e-5f);
    }
    __syncthreads();
    float inv_std = warpSums[0];

    // 6) apply norm + affine
    __nv_bfloat162 one2 = __float2bfloat162_rn(1.0f);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float rx = (__bfloat162float(x_val[i].x) - mean) * inv_std;
      float ry = (__bfloat162float(x_val[i].y) - mean) * inv_std;
      x_val[i].x = __float2bfloat16(rx);
      x_val[i].y = __float2bfloat16(ry);
    }
    x_val[0] = __hadd2(__hmul2(x_val[0], __hadd2(one2, w_val[0])), b_val[0]);
    x_val[1] = __hadd2(__hmul2(x_val[1], __hadd2(one2, w_val[1])), b_val[1]);
    x_val[2] = __hadd2(__hmul2(x_val[2], __hadd2(one2, w_val[2])), b_val[2]);
    x_val[3] = __hadd2(__hmul2(x_val[3], __hadd2(one2, w_val[3])), b_val[3]);

    // 7) reorder + 写回
    int head_id      = c / head_dim;
    int lane_in_head = c % head_dim;
    int pid          = pattern[head_id];
    int new_s        = to_flat(pid, f, h, w, F, H, W);

    // 计算输出偏移（单位：half）
    int out_idx = blockIdx.x * text_length * dim
                + new_s * dim
                + head_id * head_dim
                + lane_in_head;
    __nv_bfloat16* out_ptr = o + out_idx;
    __nv_bfloat162* dst    = reinterpret_cast<__nv_bfloat162*>(out_ptr);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      dst[i] = x_val[i];
    }
}

__global__ __forceinline__
void inv_reorder_layernorm_kernel(
    const __nv_bfloat16* __restrict__ x,          // 经过重排的输入  [bs, T, C]
    const __nv_bfloat16* __restrict__ rw,         // scale (γ-1)    [bs, C]
    const __nv_bfloat16* __restrict__ rb,         // shift (β)      [bs, C]
    __nv_bfloat16*       __restrict__ o,          // 输出 (FHW 顺序) same shape as x
    int   bs, int text_length, int dim,
    int   F,  int H, int W,
    const int* __restrict__ pattern,     
    int   head_dim)
{
    const int tid   = threadIdx.x;
    const int j     = tid << 3;                 // 每线程操 8 half (=4 half2)
    if (j >= dim) return;

    const int token_id = blockIdx.y;            // 0…T-1，对应 FHW 顺序
    const int batch_id = blockIdx.x;

    /* ---- token 的 (f,h,w) 坐标 ---- */
    const int f = token_id / (H * W);
    const int rem = token_id % (H * W);
    const int h = rem / W;
    const int w = rem % W;

    /* ---- channel → head ---- */
    const int c        = j;                     // 0…C-1
    const int head_id  = c / head_dim;
    const int lane_in_head = c % head_dim;
    const int pid      = pattern[head_id];
    const int src_s = to_flat(pid, f, h, w, F, H, W);
    const int src_idx =
        batch_id * text_length * dim + src_s * dim + j;

    /* ---- 读 x, γ, β ---- */
    __nv_bfloat162 x_val[4], g_val[4], b_val[4];
    *(float4*)&x_val[0] = *(const float4*)&x [src_idx];
    *(float4*)&g_val[0] = *(const float4*)&rw[batch_id * dim + j];
    *(float4*)&b_val[0] = *(const float4*)&rb[batch_id * dim + j];

    /* ---- LayerNorm（同上一版） ---- */
    float mean_part = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        mean_part += __bfloat162float(x_val[i].x) + __bfloat162float(x_val[i].y);

    __shared__ float smem[WARP_SIZE];            
    float mean = blockReduceSum(mean_part, smem) / float(dim);

    float var_part = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float dx = __bfloat162float(x_val[i].x) - mean;
        float dy = __bfloat162float(x_val[i].y) - mean;
        var_part += dx*dx + dy*dy;
    }
    float inv_std = rsqrtf(blockReduceSum(var_part, smem) /
                           float(dim) + 1e-5f);

    const __nv_bfloat162 one2 = __float2bfloat162_rn(1.0f);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float nx = (__bfloat162float(x_val[i].x) - mean) * inv_std;
        float ny = (__bfloat162float(x_val[i].y) - mean) * inv_std;
        x_val[i].x = __float2bfloat16(nx);
        x_val[i].y = __float2bfloat16(ny);
        __nv_bfloat162 gamma = __hadd2(one2, g_val[i]);     // γ = 1 + scale
        x_val[i]    = __hadd2(__hmul2(x_val[i], gamma), b_val[i]);
    }

    /* ---- 写回到 “原 FHW 顺序” ---- */
    const int dst_idx =
         batch_id * text_length * dim + token_id * dim + j;
    *(float4*)&o[dst_idx] = *(float4*)&x_val[0];
}