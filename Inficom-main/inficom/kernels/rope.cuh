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