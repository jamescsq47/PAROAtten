#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "utils.h"

__global__ __forceinline__ void silu_kernel(half * __restrict__ c, 
                const int bs, const int h_dim) {

    int bid = blockIdx.y;   
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    if ((tid << 1) >= h_dim) {return;}
    
    float2 data;

    data = __half22float2(*(half2*)(&c[bid * h_dim + (tid << 1)]));

    data.x = data.x * __fdividef(1.0f, __expf(-1 * data.x) + 1.0f);
    data.y = data.y * __fdividef(1.0f, __expf(-1 * data.y) + 1.0f);

    *(half2*)(&c[bid * h_dim + (tid << 1)]) = __float22half2_rn(data);
}


__global__ __forceinline__ void relu_kernel(half * __restrict__ c, 
                const int bs, const int h_dim) {

    int bid = blockIdx.y;   
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    if ((tid << 1) >= h_dim) {return;}
    
    float2 data;

    data = __half22float2(*(half2*)(&c[bid * h_dim + (tid << 1)]));

    data.x = max(0, data.x);
    data.y = max(0, data.y);

    *(half2*)(&c[bid * h_dim + (tid << 1)]) = __float22half2_rn(data);
}


__global__ __forceinline__ void gelu_kernel(half * __restrict__ c, 
                const int bs, const int h_dim) {

    int bid = blockIdx.y;   
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    if ((tid << 1) >= h_dim) {return;}
    
    float2 data;

    data = __half22float2(*(half2*)(&c[bid * h_dim + (tid << 1)]));

    data.x = 0.5f * (1.0f + __tanhf((0.7978845608028654f * (data.x + 0.044715f * __powf(data.x, 3)))));
    data.y = 0.5f * (1.0f + __tanhf((0.7978845608028654f * (data.y + 0.044715f * __powf(data.y, 3)))));

    *(half2*)(&c[bid * h_dim + (tid << 1)]) = __float22half2_rn(data);
}


