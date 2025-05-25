#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "utils.h"

/*
    A Rope kernel with permutation for PAROAttention.
*/
__global__ __forceinline__ void rope_permutation_kernel(
                          half* K, // in: [bs, H, W, dim], out: [bs, W, H, dim]
                          half* Q, // in: [bs, H, W, dim], out: [bs, W, H, dim]
                          float* freq, 
                          int bs, int H, int W, int dim, int hs) {
  
    // a block deals with a token
    int b_id = blockIdx.z;
    int h_id = blockIdx.y;
    int w_id = blockIdx.x;
    int tid = threadIdx.x;

    int old_token_id = b_id * (H * W) + (h_id * W + w_id);
    int new_token_id = b_id * (H * W) + (w_id * H + h_id);

    half2 reg_q[4];
    half2 reg_k[4];

#pragma unroll
    for (int iter = 0; iter < DIV_UP(dim, 8 * blockDim.x); iter++) {
        unsigned int j = (tid + iter * blockDim.x) << 3;
        if (j >= dim) {break;}
        // load data
        *(float4*)(reg_q) = *(float4*)(&Q[old_token_id * dim + j]);
        *(float4*)(reg_k) = *(float4*)(&K[old_token_id * dim + j]);
        // RoPE
        int idx = j % hs;
#pragma unroll   
        for (int i = 0; i < 4; i++) {
            float2 to_rotate = *(float2*)(&freq[idx + (i << 1)]);
            float2 gres;
            gres.x = to_rotate.x * __half2float(reg_q[i].x) - to_rotate.y * __half2float(reg_q[i].y);
            gres.y = to_rotate.x * __half2float(reg_q[i].y) + to_rotate.y * __half2float(reg_q[i].x);
            reg_q[i] = __float22half2_rn(gres);
            
            gres.x = to_rotate.x * __half2float(reg_k[i].x) - to_rotate.y * __half2float(reg_k[i].y);
            gres.y = to_rotate.x * __half2float(reg_k[i].y) + to_rotate.y * __half2float(reg_k[i].x);
            reg_k[i] = __float22half2_rn(gres);
        }    
        *(float4*)(&Q[new_token_id * dim + j]) = *(float4*)(reg_q);
        *(float4*)(&K[new_token_id * dim + j]) = *(float4*)(reg_k);
    }
}

// FHW 解码
__device__ inline void decode_FHW(int idx, int H, int W,
                                  int &f, int &h, int &w) {
  f = idx / (H*W);
  int rem = idx % (H*W);
  h = rem / W;
  w = rem % W;
}

__device__ inline int to_flat(int pid,
    int f, int h, int w,
    int F, int H, int W) {
  switch(pid) {
    case 0: return f*(H*W) + h*W + w;              // FHW
    case 1: return f*(W*H) + w*H + h;              // FWH 
    case 2: return w*(F*H) + f*H + h;              // WFH 
    case 3: return w*(H*F) + h*F + f;              // WHF 
    case 4: return h*(F*W) + f*W + w;              // HFW 
    case 5: return h*(W*F) + w*F + f;              // HWF 
  }
  return 0;
}

/*
    A Rope kernel with permutation for PAROAttention demo.
*/
__global__ __forceinline__ void reorder_rope_kernel(
                          __nv_bfloat16* Q, // in: [B, (FxHxW), D], out: [B, (FxHxW), D]
                          __nv_bfloat16* K, // in: [B, (FxHxW), D], out: [B, (FxHxW), D]
                          __nv_bfloat16* QO, 
                          __nv_bfloat16* KO, 
                          float* cos, float* sin,  
                          int B, int F, int H, int W, int D, int S, 
                          int pid) {
  
    // a block deals with a token
    int b_id = blockIdx.y;
    int s_id = blockIdx.x;
    // a thread deals with 8 hiddens
    int tid = threadIdx.x;

    int f,h,w;
    decode_FHW(s_id, H, W, f, h, w);
    int token_id = to_flat(pid, f, h, w, F, H, W);

    __nv_bfloat162 reg_q[4];
    __nv_bfloat162 reg_k[4];

    float* cos_ptr = cos + s_id * D;
    float* sin_ptr = sin + s_id * D;

#pragma unroll
    for (int iter = 0; iter < DIV_UP(D, 8 * blockDim.x); iter++) {
        unsigned int j = (tid + iter * blockDim.x) << 3;
        if (j >= D) {break;}
        // load data
        *(float4*)(reg_q) = *(float4*)(&Q[(b_id * S + token_id) * D + j]);
        *(float4*)(reg_k) = *(float4*)(&K[(b_id * S + token_id) * D + j]);
        // RoPE
        // int idx = j % hs;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            // float2 to_rotate = *(float2*)(&freq[idx + (i << 1)]);
            float2 this_cos = *(float2*)(&cos_ptr[j + (i << 1)]);
            float2 this_sin = *(float2*)(&sin_ptr[j + (i << 1)]);
            float2 gres;
            gres.x = this_cos.x * __bfloat162float(reg_q[i].x) - this_sin.x * __bfloat162float(reg_q[i].y);
            gres.y = this_cos.y * __bfloat162float(reg_q[i].y) + this_sin.y * __bfloat162float(reg_q[i].x);
            reg_q[i] = __float22bfloat162_rn(gres);
            
            gres.x = this_cos.x * __bfloat162float(reg_k[i].x) - this_sin.x * __bfloat162float(reg_k[i].y);
            gres.y = this_cos.y * __bfloat162float(reg_k[i].y) + this_sin.y * __bfloat162float(reg_k[i].x);
            reg_k[i] = __float22bfloat162_rn(gres);
        }    
        *(float4*)(&QO[(b_id * S + token_id) * D + j]) = *(float4*)(reg_q);
        *(float4*)(&KO[(b_id * S + token_id) * D + j]) = *(float4*)(reg_k);
    }
}