/*
    Utility functions. 
*/

#pragma once

#include <cuda_fp16.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define DIV_UP(x, y) ((x) + (y) - 1) / (y)
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

__device__ __forceinline__ float warpReduceSum(float sum_val,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, 1);  // 0-1, 2-3, 4-5, etc.
  return sum_val;
}

__device__ __forceinline__ half warpReduceSum(half result,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    result = __hadd(result, __shfl_down_sync(0xffffffff, result, 1));  // 0-1, 2-3, 4-5, etc.
  return result;
}

__device__ __forceinline__ half2 warpReduceSum(half2 result,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    result = __hadd2(result, __shfl_down_sync(0xffffffff, result, 1));  // 0-1, 2-3, 4-5, etc.
  return result;
}

__device__ __forceinline__ float warpReduceMax(float max_val,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 16));  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 8));  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 4));  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, 1));  // 0-1, 2-3, 4-5, etc.
  return max_val;
}

// __device__ __forceinline__
// float blockReduceSum(float v, float *shared_mem)
// {
//     // --- ① 每个 warp 内求和 ---
//     #pragma unroll
//     for (int mask = 16; mask > 0; mask >>= 1)
//         v += __shfl_down_sync(0xffffffff, v, mask);

//     const int lane_id = threadIdx.x & 31;     // 0-31
//     const int warp_id = threadIdx.x >> 5;     // 0 .. warpCnt-1

//     // --- ② 写入共享内存 ---
//     if (lane_id == 0) shared_mem[warp_id] = v;
//     __syncthreads();

//     // --- ③ 第 0 warp 做跨-warp reduce ---
//     const int warp_cnt = (blockDim.x + 31) >> 5;   // 实际 warp 数
//     v = (threadIdx.x < warp_cnt) ? shared_mem[lane_id] : 0.f;

//     if (warp_id == 0) {
//         #pragma unroll
//         for (int mask = 16; mask > 0; mask >>= 1)
//             v += __shfl_down_sync(0xffffffff, v, mask);
//     }

//     // --- ④ 广播最终结果 ---
//     v = __shfl_sync(0xffffffff, v, 0);
//     return v;
// }
__device__ __forceinline__ float blockReduceSum(float reducing, float *shared_mem)
{
    // Helper function for reduce softmax exp sum.
    int32_t WPT = blockDim.x / 32;
    int32_t WPTB = 32 / (32 / WPT);
    const int32_t lane_id = threadIdx.x % 32;
    const int32_t warp_id = threadIdx.x / 32;

# pragma unroll
    for (int32_t mask = 16; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPTB) reducing = lane_id < WPT ? shared_mem[lane_id] : 0.0f;

# pragma unroll
    for (int32_t mask = WPTB / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

// __device__ __forceinline__ float blockReduceSum(float reducing, float *shared_mem)
// {
//     // Helper function for reduce softmax exp sum.
//     const int32_t WPT = blockDim.x / 32;
//     int32_t WPTB = WPT == 20 ? 32 : WPT;
//     const int32_t lane_id = threadIdx.x % 32;
//     const int32_t warp_id = threadIdx.x / 32;

// # pragma unroll
//     for (int32_t mask = 16; mask >= 1; mask /= 2) {
//         reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
//     }

//     if (lane_id == 0) shared_mem[warp_id] = reducing;
//     __syncthreads();

//     if (lane_id < WPTB) reducing = lane_id < WPT ? shared_mem[lane_id] : 0.0f;

// # pragma unroll
//     for (int32_t mask = WPTB / 2; mask >= 1; mask /= 2) {
//         reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
//     }
//     reducing = __shfl_sync(uint32_t(-1), reducing, 0);
//     return reducing;
// }

__device__ __forceinline__ half blockReduceSum(half reducing, half *shared_mem)
{
    // Helper function for reduce softmax exp sum.
    const int32_t WPT = blockDim.x / 32;
    const int32_t lane_id = threadIdx.x % 32;
    const int32_t warp_id = threadIdx.x / 32;

# pragma unroll
    for (int32_t mask = 16; mask >= 1; mask /= 2) {
        reducing = __hadd(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing = __hadd(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

__device__ __forceinline__ float4 _12bits_dequant(uint8_t *input){

    half output[8];
    output[0] = static_cast<half>(input[0] << 4 + ((input[1] & 0xf0) >> 4));
    output[1] = static_cast<half>((input[1] & 0x0f) << 8 + input[2]);
    output[2] = static_cast<half>(input[3] << 4 + ((input[4] & 0xf0) >> 4));
    output[3] = static_cast<half>((input[4] & 0x0f) << 8 + input[5]);
    output[4] = static_cast<half>(input[6] << 4 + ((input[7] & 0xf0) >> 4));
    output[5] = static_cast<half>((input[7] & 0x0f) << 8 + input[8]);
    output[6] = static_cast<half>(input[9] << 4 + ((input[10] & 0xf0) >> 4));
    output[7] = static_cast<half>((input[10] & 0x0f) << 8 + input[11]);

    return *((float4*)&output[0]);
}

__device__ __forceinline__ float4 _4bits_dequant(uint8_t *input){

    half output[8];
    output[0] = static_cast<half>(input[0] >> 4);
    output[1] = static_cast<half>(input[0] & 0x0f);
    output[2] = static_cast<half>(input[1] >> 4);
    output[3] = static_cast<half>(input[1] & 0x0f);
    output[4] = static_cast<half>(input[2] >> 4);
    output[5] = static_cast<half>(input[2] & 0x0f);
    output[6] = static_cast<half>(input[3] >> 4);
    output[7] = static_cast<half>(input[3] & 0x0f);

    return *((float4*)&output[0]);
}