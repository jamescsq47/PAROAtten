#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <cub/cub.cuh>

#include "utils.h"

#define FLOAT_BANK_SIZE 32
#define HALF_BANK_SIZE 64
#define MAX_HEAD_SIZE 128
#define MAX_LEN_GROUP 64
#define MAX_LOOP_SPACE 2

/*
    Decoding multi-head attention kernel with async softmax.
    FLOOP = DIV_UP(len_group, FLOAT_BANK_SIZE),
    loop = DIV_UP(len, BLOCKSIZE), useless in this kernel.
    TODO: speed loss needs to be fixed. [0.175 ms --> 0.245 ms]
    Layout:
        Q   [bs, 1, hn, hs]
        K/V [bs, max_len, hn, hs]
*/
template <int BLOCK_SIZE, int FLOOP>
__global__ __forceinline__ void decode_mha_with_async_softmax_kernel(
                half* Q, half* K, half* V, const float p_scale, const float max_val, 
                int bs, int hn, int dim, int max_len, int len, int hs, int loop, half* H) {

  unsigned int q_offset = blockIdx.x * dim + blockIdx.y * hs;
  unsigned int kv_offset = blockIdx.x * max_len * dim + blockIdx.y * hs;
  
  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  
  int len_group = DIV_UP(BLOCK_SIZE, (hs >> 3)); // make sure len_group is a power-of-two number
  int pad_len = DIV_UP(len, len_group) * len_group; 

  unsigned int kv_row = tid / (hs >> 3);
  unsigned int kv_col = tid % (hs >> 3);
  unsigned int j = kv_col << 3;

  half2 q_val[4], k_val[4], v_val[4], s_temp;
  half temp_sum;
  half PD[8] = {__float2half(0.0f)};

  // 4KB
  __shared__ float s_shmem[FLOOP * (FLOAT_BANK_SIZE + 8)];
  // 35KB
  __shared__ half av_shmem[MAX_LEN_GROUP][MAX_HEAD_SIZE + 8]; 
  __shared__ float block_sum[4];

  // load Q
  *(float4*)(&q_val[0]) = *(float4*)(&Q[q_offset + j]);

  float temp_sum_val = 0.0f;
  float a;
  float4 pv[2];

#pragma unroll
  for (int i = 0; i < 2; i++){
    pv[i].x = 0.0f;
    pv[i].y = 0.0f;
    pv[i].z = 0.0f;
    pv[i].w = 0.0f;
  }

#pragma unroll
  for (int row = kv_row; row < pad_len; row += len_group){

    // Q x K
    *(float4*)(&k_val[0]) = row < len ? 
      *(float4*)(&K[kv_offset + row * dim + j]) : 
      *(float4*)(&PD[0]);
    *(float4*)(&v_val[0]) = row < len ? 
      *(float4*)(&V[kv_offset + row * dim + j]) : 
      *(float4*)(&PD[0]);

    s_temp = __hmul2(q_val[0], k_val[0]);
    s_temp = __hadd2(s_temp, __hmul2(q_val[1], k_val[1]));
    s_temp = __hadd2(s_temp, __hmul2(q_val[2], k_val[2]));
    s_temp = __hadd2(s_temp, __hmul2(q_val[3], k_val[3]));

    temp_sum = __hadd(s_temp.x, s_temp.y);
    temp_sum = warpReduceSum(temp_sum, (hs >> 3));

    // now for all k_col == 0, a temp_s is stored in the register
    // push into share memory
    unsigned int s_row = kv_row / FLOAT_BANK_SIZE;
    unsigned int s_col = kv_row % FLOAT_BANK_SIZE;

    __syncthreads();

    // softmax without scaling
    if (kv_col == 0){
      s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col] = row < len ? 
        __expf(__half2float(temp_sum) * p_scale - max_val) : 0.0f;
    }

    __syncthreads();
    // calculate the intermediate sum
    unsigned int t_row = tid / FLOAT_BANK_SIZE;
    unsigned int t_col = tid % FLOAT_BANK_SIZE;
    temp_sum_val += (tid < len_group) ? 
        s_shmem[t_row * (FLOAT_BANK_SIZE + 8) + t_col] : 0.0f;

    // AxV
    a = s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col];

    pv[0].x = pv[0].x + a * __half2float(v_val[0].x);
    pv[0].y = pv[0].y + a * __half2float(v_val[0].y);
    pv[0].z = pv[0].z + a * __half2float(v_val[1].x);
    pv[0].w = pv[0].w + a * __half2float(v_val[1].y);
    pv[1].x = pv[1].x + a * __half2float(v_val[2].x);
    pv[1].y = pv[1].y + a * __half2float(v_val[2].y);
    pv[1].z = pv[1].z + a * __half2float(v_val[3].x);
    pv[1].w = pv[1].w + a * __half2float(v_val[3].y);
  }

  unsigned int real_len_group = min(len_group, 32);

  unsigned int o_row = tid % len_group;
  unsigned int o_col = tid / len_group;

  // calculate the global sum
  temp_sum_val = warpReduceSum(temp_sum_val, real_len_group);

  if (tid % real_len_group == 0 && tid < len_group){
    block_sum[(tid >> 5)] = temp_sum_val; 
  }
  __syncthreads();

  if (tid == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      block_sum[0] += block_sum[r];
    }
  }

  __syncthreads();

  float scaling = __fdividef(1.0f, block_sum[0] + 1e-6f);

  half2 pv_half[4];

  pv_half[0] = {__float2half(pv[0].x * scaling), __float2half(pv[0].y * scaling)};
  pv_half[1] = {__float2half(pv[0].z * scaling), __float2half(pv[0].w * scaling)};
  pv_half[2] = {__float2half(pv[1].x * scaling), __float2half(pv[1].y * scaling)};
  pv_half[3] = {__float2half(pv[1].z * scaling), __float2half(pv[1].w * scaling)};

  // transpoisition allows efficient mem access of V matrix
  *(float4*)(&av_shmem[kv_row][(kv_col << 3)]) = *(float4*)(&pv_half[0]);
  __syncthreads();

  *(float4*)(&pv_half[0]) = *(float4*)(&av_shmem[o_row][(o_col << 3)]);

  pv_half[0] = warpReduceSum(pv_half[0], real_len_group);
  pv_half[1] = warpReduceSum(pv_half[1], real_len_group);
  pv_half[2] = warpReduceSum(pv_half[2], real_len_group);
  pv_half[3] = warpReduceSum(pv_half[3], real_len_group);

  if (o_row % 32 == 0){
    *(float4*)(&av_shmem[(o_row >> 5)][(o_col << 3)]) = *(float4*)(&pv_half[0]);
  }
  __syncthreads();

  if (o_row == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      pv_half[0] = __hadd2(pv_half[0], *(half2*)(&av_shmem[r][(o_col << 3) + 0]));
      pv_half[1] = __hadd2(pv_half[1], *(half2*)(&av_shmem[r][(o_col << 3) + 2]));
      pv_half[2] = __hadd2(pv_half[2], *(half2*)(&av_shmem[r][(o_col << 3) + 4]));
      pv_half[3] = __hadd2(pv_half[3], *(half2*)(&av_shmem[r][(o_col << 3) + 6]));
    } 
    *(float4*)(&H[q_offset + (o_col << 3)]) = *(float4*)(&pv_half[0]);
  }
}


/*
    Decoding multi-head attention kernel without async softmax.
    FLOOP = DIV_UP(len, FLOAT_BANK_SIZE),
    loop = DIV_UP(len, BLOCKSIZE).
    The kernel only supports KV cache len <= 8k due to the shared memory limitation.
    Layout:
        Q   [bs, 1, hn, hs]
        K/V [bs, max_len, hn, hs]
*/
template <int BLOCK_SIZE, int FLOOP>
__global__ __forceinline__ void decode_mha_fall_back_kernel(
                half* Q, half* K, half* V, const float p_scale, 
                int bs, int hn, int dim, int max_len, int len, int hs, int loop, half* H) {

  unsigned int q_offset = blockIdx.x * dim + blockIdx.y * hs;
  unsigned int kv_offset = blockIdx.x * max_len * dim + blockIdx.y * hs;
  
  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  
  int len_group = DIV_UP(BLOCK_SIZE, (hs >> 3)); // make sure len_group is a power-of-two number
  int pad_len = DIV_UP(len, len_group) * len_group;

  unsigned int k_row = tid / (hs >> 3);
  unsigned int k_col = tid % (hs >> 3);
  unsigned int j = k_col << 3;

  half2 q_val[4], k_val[4];
  float s_temp;
  half temp_sum;
  half PD[8] = {__float2half(0.0f)};

  // 4KB
  __shared__ float s_shmem[FLOOP * (FLOAT_BANK_SIZE + 8)];
  // 512B
  __shared__ float temp_max_val[MAX_LEN_GROUP];
  // 35KB
  __shared__ half av_shmem[MAX_LEN_GROUP][MAX_HEAD_SIZE + 8]; 
  // 1KB
  __shared__ half half_space[MAX_LOOP_SPACE][MAX_HEAD_SIZE + 8];
  __shared__ float float_space[MAX_LOOP_SPACE];
  __shared__ float block_max, block_sum;

  // load Q
  *(float4*)(&q_val[0]) = *(float4*)(&Q[q_offset + j]);

  // to execute reduction in the loop
  float qxk_result = 0.0f;
  float sum_val = 0.0f, temp_sum_val = 0.0f;
  float max_val = -1e20f, temp_max_val_thread = -1e20f;

  // Q x K
#pragma unroll
  for (int row = k_row; row < pad_len; row += len_group){
    // if (row >= len) {s_shmem[row] = 0; break;}
    
    *(float4*)(&k_val[0]) = row < len ? 
      *(float4*)(&K[kv_offset + row * dim + j]) : 
      *(float4*)(&PD[0]);

    s_temp = __half2float(q_val[0].x) * __half2float(k_val[0].x);
    s_temp += __half2float(q_val[0].y) * __half2float(k_val[0].y);
    s_temp += __half2float(q_val[1].x) * __half2float(k_val[1].x);
    s_temp += __half2float(q_val[1].y) * __half2float(k_val[1].y);
    s_temp += __half2float(q_val[2].x) * __half2float(k_val[2].x);
    s_temp += __half2float(q_val[2].y) * __half2float(k_val[2].y);
    s_temp += __half2float(q_val[3].x) * __half2float(k_val[3].x);
    s_temp += __half2float(q_val[3].y) * __half2float(k_val[3].y);
    s_temp = warpReduceSum(s_temp, (hs >> 3));

    // now for all k_col == 0, a temp_s is stored in the register
    // push into share memory
    unsigned int s_row = row / FLOAT_BANK_SIZE;
    unsigned int s_col = row % FLOAT_BANK_SIZE;

    if (k_col == 0){
      qxk_result = s_temp * p_scale;
      temp_max_val_thread = max(temp_max_val_thread, qxk_result);
      s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col] = qxk_result;
    }
  }

  if (k_col == 0){
    temp_max_val[k_row] = temp_max_val_thread;
  }

  // make sure QxK and temporary max and sum are stored into shmem
  __syncthreads();
  
  max_val = tid < len_group ? temp_max_val[tid] : -1e20f;
  max_val = warpReduceMax(max_val, min(len_group, 32));
  
  if (tid % 32 == 0){
    float_space[(tid / 32)] = max_val;
  }
  __syncthreads();

  if (tid == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      float_space[0] = max(float_space[0], float_space[r]);
    } 
  }
  __syncthreads();

  __shared__ typename cub::BlockReduce<float, BLOCK_SIZE>::TempStorage temp_storage;
  
#pragma unroll
  for (int l = 0; l < loop; l += 1){
    unsigned int lid = l * BLOCK_SIZE + tid;
    unsigned int l_row = lid / FLOAT_BANK_SIZE;
    unsigned int l_col = lid % FLOAT_BANK_SIZE;
    s_shmem[l_row * (FLOAT_BANK_SIZE + 8) + l_col] = lid < len ? 
      __expf(s_shmem[l_row * (FLOAT_BANK_SIZE + 8) + l_col] - float_space[0]) : 0.0f;
    temp_sum_val += s_shmem[l_row * (FLOAT_BANK_SIZE + 8) + l_col];
  }
  temp_sum_val = cub::BlockReduce<float, BLOCK_SIZE>(temp_storage).Reduce(temp_sum_val, cub::Sum());
  if (tid == 0){block_sum = temp_sum_val;}
  __syncthreads();
  
  float a;
  half result;
  float2 pv[4];

#pragma unroll
  for (int i = 0; i < 4; i++){
    pv[i].x = 0.0f;
    pv[i].y = 0.0f;
  }

  // V partition
  unsigned int v_row = tid / (hs >> 3);
  unsigned int v_col = tid % (hs >> 3);

#pragma unroll
  for (int row = v_row; row < pad_len; row += len_group){

    *(float4*)(&k_val[0]) = row < len ? 
      *(float4*)(&V[kv_offset + row * dim + (v_col << 3)]) : 
      *(float4*)(&PD[0]);
    
    unsigned int a_row = row / FLOAT_BANK_SIZE;
    unsigned int a_col = row % FLOAT_BANK_SIZE;

    a = s_shmem[a_row * (FLOAT_BANK_SIZE + 8) + a_col] * 
      __fdividef(1.0f, block_sum + 1e-6f);

    pv[0].x = pv[0].x + a * __half2float(k_val[0].x);
    pv[0].y = pv[0].y + a * __half2float(k_val[0].y);
    pv[1].x = pv[1].x + a * __half2float(k_val[1].x);
    pv[1].y = pv[1].y + a * __half2float(k_val[1].y);
    pv[2].x = pv[2].x + a * __half2float(k_val[2].x);
    pv[2].y = pv[2].y + a * __half2float(k_val[2].y);
    pv[3].x = pv[3].x + a * __half2float(k_val[3].x);
    pv[3].y = pv[3].y + a * __half2float(k_val[3].y);
  }

  half2 pv_half[4];

  pv_half[0] = __float22half2_rn(pv[0]);
  pv_half[1] = __float22half2_rn(pv[1]);
  pv_half[2] = __float22half2_rn(pv[2]);
  pv_half[3] = __float22half2_rn(pv[3]);

  // transpoisition allows efficient mem access of V matrix
  *(float4*)(&av_shmem[v_row][(v_col << 3)]) = *(float4*)(&pv_half[0]);
  __syncthreads();

  unsigned int o_row = tid % len_group;
  unsigned int o_col = tid / len_group;

  *(float4*)(&pv_half[0]) = *(float4*)(&av_shmem[o_row][(o_col << 3)]);

  unsigned int real_len_group = min(len_group, 32);

  pv_half[0] = warpReduceSum(pv_half[0], real_len_group);
  pv_half[1] = warpReduceSum(pv_half[1], real_len_group);
  pv_half[2] = warpReduceSum(pv_half[2], real_len_group);
  pv_half[3] = warpReduceSum(pv_half[3], real_len_group);

  if (o_row % 32 == 0){
    *(float4*)(&av_shmem[(o_row / 32)][(o_col << 3)]) = *(float4*)(&pv_half[0]);
  }
  __syncthreads();

  if (o_row == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      pv_half[0] = __hadd2(pv_half[0], *(half2*)(&av_shmem[r][(o_col << 3) + 0]));
      pv_half[1] = __hadd2(pv_half[1], *(half2*)(&av_shmem[r][(o_col << 3) + 2]));
      pv_half[2] = __hadd2(pv_half[2], *(half2*)(&av_shmem[r][(o_col << 3) + 4]));
      pv_half[3] = __hadd2(pv_half[3], *(half2*)(&av_shmem[r][(o_col << 3) + 6]));
    } 
    *(float4*)(&H[q_offset + (o_col << 3)]) = *(float4*)(&pv_half[0]);
  }
}


/*
    Decoding multi-query attention kernel with async softmax.
    FLOOP = DIV_UP(len_group, FLOAT_BANK_SIZE),
    loop = DIV_UP(len, BLOCKSIZE), useless in this kernel.
    Layout:
        Q   [1, bs, hn, hs]
        K/V [max_len, bs, hn, hs]
*/
template <int BLOCK_SIZE, int FLOOP>
__global__ __forceinline__ void decode_mqa_with_async_softmax_kernel(
                half* Q, half* K, half* V, const float p_scale, const float max_val, 
                int bs, int hn, int gn, int dim, int max_len, int len, int hs, int loop, half* H) {
  
  // [hongke @ 10.25: for GLM2-6B
  //    dim = 4096, hn = 32, gn = 2, hs = 128]

  unsigned int q_offset = blockIdx.x * dim + blockIdx.y * hs;
  unsigned int kv_offset = blockIdx.x * (gn * hs) + blockIdx.y / (hn / gn) * hs;
  
  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  
  int len_group = DIV_UP(BLOCK_SIZE, (hs >> 3)); // make sure len_group is a power-of-two number
  int pad_len = DIV_UP(len, len_group) * len_group; 

  unsigned int kv_row = tid / (hs >> 3);
  unsigned int kv_col = tid % (hs >> 3);
  unsigned int j = kv_col << 3;

  half2 q_val[4], k_val[4], v_val[4], s_temp;
  half temp_sum;
  half PD[8] = {__float2half(0.0f)};

  // 4KB
  __shared__ float s_shmem[FLOOP * (FLOAT_BANK_SIZE + 8)];
  // 35KB
  __shared__ half av_shmem[MAX_LEN_GROUP][MAX_HEAD_SIZE + 8]; 
  // 1KB
  __shared__ float block_sum[4];

  // load Q
  *(float4*)(&q_val[0]) = *(float4*)(&Q[q_offset + j]);

  float temp_sum_val = 0.0f;
  float a;
  float4 pv[2];

#pragma unroll
  for (int i = 0; i < 2; i++){
    pv[i].x = 0.0f;
    pv[i].y = 0.0f;
    pv[i].z = 0.0f;
    pv[i].w = 0.0f;
  }

  unsigned int s_row = kv_row / FLOAT_BANK_SIZE;
  unsigned int s_col = kv_row % FLOAT_BANK_SIZE;
  unsigned int t_row = tid / FLOAT_BANK_SIZE;
  unsigned int t_col = tid % FLOAT_BANK_SIZE;

#pragma unroll
  for (int row = kv_row; row < pad_len; row += len_group){

    // Q x K
    *(float4*)(&k_val[0]) = row < len ? 
      *(float4*)(&K[kv_offset + row * (bs * gn * hs) + j]) : 
      *(float4*)(&PD[0]);
    *(float4*)(&v_val[0]) = row < len ? 
      *(float4*)(&V[kv_offset + row * (bs * gn * hs) + j]) : 
      *(float4*)(&PD[0]);

    s_temp = __hmul2(q_val[0], k_val[0]);
    s_temp = __hadd2(s_temp, __hmul2(q_val[1], k_val[1]));
    s_temp = __hadd2(s_temp, __hmul2(q_val[2], k_val[2]));
    s_temp = __hadd2(s_temp, __hmul2(q_val[3], k_val[3]));

    temp_sum = __hadd(s_temp.x, s_temp.y);
    temp_sum = warpReduceSum(temp_sum, (hs >> 3));

    // softmax without scaling
    if (kv_col == 0){
      s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col] = row < len ? 
        __expf(__half2float(temp_sum) * p_scale - max_val) : 0.0f;
    }

    __syncthreads();
    temp_sum_val += (tid < len_group) ? 
        s_shmem[t_row * (FLOAT_BANK_SIZE + 8) + t_col] : 0.0f;

    // AxV
    a = s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col];

    pv[0].x = pv[0].x + a * __half2float(v_val[0].x);
    pv[0].y = pv[0].y + a * __half2float(v_val[0].y);
    pv[0].z = pv[0].z + a * __half2float(v_val[1].x);
    pv[0].w = pv[0].w + a * __half2float(v_val[1].y);
    pv[1].x = pv[1].x + a * __half2float(v_val[2].x);
    pv[1].y = pv[1].y + a * __half2float(v_val[2].y);
    pv[1].z = pv[1].z + a * __half2float(v_val[3].x);
    pv[1].w = pv[1].w + a * __half2float(v_val[3].y);
  }

  unsigned int real_len_group = min(len_group, 32);

  unsigned int o_row = tid % len_group;
  unsigned int o_col = tid / len_group;

  // calculate the global sum
  temp_sum_val = warpReduceSum(temp_sum_val, real_len_group);

  if (tid % real_len_group == 0 && tid < len_group){
    block_sum[(tid >> 5)] = temp_sum_val; 
  }
  __syncthreads();

  if (tid == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      block_sum[0] += block_sum[r];
    }
  }

  __syncthreads();

  float scaling = __fdividef(1.0f, block_sum[0] + 1e-6f);

  half2 pv_half[4];

  pv_half[0] = {__float2half(pv[0].x * scaling), __float2half(pv[0].y * scaling)};
  pv_half[1] = {__float2half(pv[0].z * scaling), __float2half(pv[0].w * scaling)};
  pv_half[2] = {__float2half(pv[1].x * scaling), __float2half(pv[1].y * scaling)};
  pv_half[3] = {__float2half(pv[1].z * scaling), __float2half(pv[1].w * scaling)};

  // transpoisition allows efficient mem access of V matrix
  *(float4*)(&av_shmem[kv_row][(kv_col << 3)]) = *(float4*)(&pv_half[0]);
  __syncthreads();

  *(float4*)(&pv_half[0]) = *(float4*)(&av_shmem[o_row][(o_col << 3)]);

  pv_half[0] = warpReduceSum(pv_half[0], real_len_group);
  pv_half[1] = warpReduceSum(pv_half[1], real_len_group);
  pv_half[2] = warpReduceSum(pv_half[2], real_len_group);
  pv_half[3] = warpReduceSum(pv_half[3], real_len_group);

  __syncthreads();

  if (o_row % 32 == 0){
    *(float4*)(&av_shmem[(o_row >> 5)][(o_col << 3)]) = *(float4*)(&pv_half[0]);
  }

  __syncthreads();

  if (o_row == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      pv_half[0] = __hadd2(pv_half[0], *(half2*)(&av_shmem[r][(o_col << 3) + 0]));
      pv_half[1] = __hadd2(pv_half[1], *(half2*)(&av_shmem[r][(o_col << 3) + 2]));
      pv_half[2] = __hadd2(pv_half[2], *(half2*)(&av_shmem[r][(o_col << 3) + 4]));
      pv_half[3] = __hadd2(pv_half[3], *(half2*)(&av_shmem[r][(o_col << 3) + 6]));
    } 
    *(float4*)(&H[q_offset + (o_col << 3)]) = *(float4*)(&pv_half[0]);
  }
}


/*
    Decoding multi-query attention kernel without async softmax.
    FLOOP = DIV_UP(len_group, FLOAT_BANK_SIZE),
    Loop >= loop = DIV_UP(len, BLOCKSIZE).
    The kernel only supports KV cache len <= 6k due to the shared memory limitation.
    Layout:
        Q   [1, bs, hn, hs]
        K/V [max_len, bs, hn, hs]
*/
template <int BLOCK_SIZE, int FLOOP, int LOOP>
__global__ __forceinline__ void decode_mqa_fall_back_kernel(
                half* Q, half* K, half* V, const float p_scale, 
                int bs, int hn, int gn, int dim, int max_len, int len, int hs, int loop, half* H) {

  // [hongke @ 10.24: for GLM2-6B
  //    dim = 4096, hn = 32, gn = 2, hs = 128]

  unsigned int q_offset = blockIdx.x * dim + blockIdx.y * hs;
  unsigned int kv_offset = blockIdx.x * (gn * hs) + blockIdx.y / (hn / gn) * hs;
  
  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  
  int len_group = DIV_UP(BLOCK_SIZE, (hs >> 3)); // make sure len_group is a power-of-two number
  int pad_len = DIV_UP(len, len_group) * len_group;

  unsigned int k_row = tid / (hs >> 3);
  unsigned int k_col = tid % (hs >> 3);
  unsigned int j = k_col << 3;

  half2 q_val[4], k_val[4];
  float s_temp;
  half temp_sum;
  half PD[8] = {__float2half(0.0f)};

  // 4KB
  __shared__ float s_shmem[FLOOP * (FLOAT_BANK_SIZE + 8)];
  // 512B
  __shared__ float temp_max_val[MAX_LEN_GROUP];
  // 35KB
  __shared__ half av_shmem[MAX_LEN_GROUP][MAX_HEAD_SIZE + 8]; 
  // 1KB
  __shared__ float float_space[MAX_LOOP_SPACE];
  __shared__ float block_max, block_sum;

  // load Q
  *(float4*)(&q_val[0]) = *(float4*)(&Q[q_offset + j]);

  // to execute reduction in the loop
  float qxk_result = 0.0f;
  float sum_val = 0.0f, temp_sum_val = 0.0f;
  float max_val = -1e20f, temp_max_val_thread = -1e20f;

  // Q x K
#pragma unroll
  for (int row = k_row; row < pad_len; row += len_group){
    // if (row >= len) {s_shmem[row] = 0; break;}
    
    *(float4*)(&k_val[0]) = row < len ? 
      *(float4*)(&K[kv_offset + row * (bs * gn * hs) + j]) : 
      *(float4*)(&PD[0]);

    s_temp = __half2float(q_val[0].x) * __half2float(k_val[0].x);
    s_temp += __half2float(q_val[0].y) * __half2float(k_val[0].y);
    s_temp += __half2float(q_val[1].x) * __half2float(k_val[1].x);
    s_temp += __half2float(q_val[1].y) * __half2float(k_val[1].y);
    s_temp += __half2float(q_val[2].x) * __half2float(k_val[2].x);
    s_temp += __half2float(q_val[2].y) * __half2float(k_val[2].y);
    s_temp += __half2float(q_val[3].x) * __half2float(k_val[3].x);
    s_temp += __half2float(q_val[3].y) * __half2float(k_val[3].y);
    
    s_temp = warpReduceSum(s_temp, (hs >> 3));


    // now for all k_col == 0, a temp_s is stored in the register
    // push into share memory
    unsigned int s_row = row / FLOAT_BANK_SIZE;
    unsigned int s_col = row % FLOAT_BANK_SIZE;

    if (k_col == 0){
      qxk_result = s_temp * p_scale;
      temp_max_val_thread = max(temp_max_val_thread, qxk_result);
      s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col] = qxk_result;
    }
  }

  if (k_col == 0){
    temp_max_val[k_row] = temp_max_val_thread;
  }

  // make sure QxK and temporary max and sum are stored into shmem
  __syncthreads();
  
  max_val = tid < len_group ? temp_max_val[tid] : -1e20f;
  max_val = warpReduceMax(max_val, min(len_group, 32));
  
  if (tid % 32 == 0){
    float_space[(tid / 32)] = max_val;
  }
  __syncthreads();

  if (tid == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      float_space[0] = max(float_space[0], float_space[r]);
    } 
  }
  __syncthreads();

  float exp_val[LOOP] = {0.0f};
  __shared__ typename cub::BlockReduce<float, BLOCK_SIZE>::TempStorage temp_storage;
  
#pragma unroll
  for (int l = 0; l < loop; l += 1){
    unsigned int lid = l * BLOCK_SIZE + tid;
    unsigned int l_row = lid / FLOAT_BANK_SIZE;
    unsigned int l_col = lid % FLOAT_BANK_SIZE;
    s_shmem[l_row * (FLOAT_BANK_SIZE + 8) + l_col] = lid < len ? 
      __expf(s_shmem[l_row * (FLOAT_BANK_SIZE + 8) + l_col] - float_space[0]) : 0.0f;
    temp_sum_val += s_shmem[l_row * (FLOAT_BANK_SIZE + 8) + l_col];
  }
  temp_sum_val = cub::BlockReduce<float, BLOCK_SIZE>(temp_storage).Reduce(temp_sum_val, cub::Sum());
  if (tid == 0){block_sum = temp_sum_val;}
  __syncthreads();
  
  float a;
  half result;
  float2 pv[4];

#pragma unroll
  for (int i = 0; i < 4; i++){
    pv[i].x = 0.0f;
    pv[i].y = 0.0f;
  }

  // V partition
  unsigned int v_row = tid / (hs >> 3);
  unsigned int v_col = tid % (hs >> 3);

#pragma unroll
  for (int row = v_row; row < pad_len; row += len_group){

    *(float4*)(&k_val[0]) = row < len ? 
      *(float4*)(&V[kv_offset + row * (bs * gn * hs) + (v_col << 3)]) : 
      *(float4*)(&PD[0]);
    
    unsigned int a_row = row / FLOAT_BANK_SIZE;
    unsigned int a_col = row % FLOAT_BANK_SIZE;

    a = s_shmem[a_row * (FLOAT_BANK_SIZE + 8) + a_col] * 
      __fdividef(1.0f, block_sum + 1e-6f);

    pv[0].x = pv[0].x + a * __half2float(k_val[0].x);
    pv[0].y = pv[0].y + a * __half2float(k_val[0].y);
    pv[1].x = pv[1].x + a * __half2float(k_val[1].x);
    pv[1].y = pv[1].y + a * __half2float(k_val[1].y);
    pv[2].x = pv[2].x + a * __half2float(k_val[2].x);
    pv[2].y = pv[2].y + a * __half2float(k_val[2].y);
    pv[3].x = pv[3].x + a * __half2float(k_val[3].x);
    pv[3].y = pv[3].y + a * __half2float(k_val[3].y);
  }

  half2 pv_half[4];

  pv_half[0] = __float22half2_rn(pv[0]);
  pv_half[1] = __float22half2_rn(pv[1]);
  pv_half[2] = __float22half2_rn(pv[2]);
  pv_half[3] = __float22half2_rn(pv[3]);

  // transpoisition allows efficient mem access of V matrix
  *(float4*)(&av_shmem[v_row][(v_col << 3)]) = *(float4*)(&pv_half[0]);
  __syncthreads();

  unsigned int o_row = tid % len_group;
  unsigned int o_col = tid / len_group;

  *(float4*)(&pv_half[0]) = *(float4*)(&av_shmem[o_row][(o_col << 3)]);

  unsigned int real_len_group = min(len_group, 32);

  pv_half[0] = warpReduceSum(pv_half[0], real_len_group);
  pv_half[1] = warpReduceSum(pv_half[1], real_len_group);
  pv_half[2] = warpReduceSum(pv_half[2], real_len_group);
  pv_half[3] = warpReduceSum(pv_half[3], real_len_group);

  if (o_row % 32 == 0){
    *(float4*)(&av_shmem[(o_row / 32)][(o_col << 3)]) = *(float4*)(&pv_half[0]);
  }
  __syncthreads();

  if (o_row == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      pv_half[0] = __hadd2(pv_half[0], *(half2*)(&av_shmem[r][(o_col << 3) + 0]));
      pv_half[1] = __hadd2(pv_half[1], *(half2*)(&av_shmem[r][(o_col << 3) + 2]));
      pv_half[2] = __hadd2(pv_half[2], *(half2*)(&av_shmem[r][(o_col << 3) + 4]));
      pv_half[3] = __hadd2(pv_half[3], *(half2*)(&av_shmem[r][(o_col << 3) + 6]));
    } 
    *(float4*)(&H[q_offset + (o_col << 3)]) = *(float4*)(&pv_half[0]);
  }
}


/*
    Experimental version.
    Decoding multi-head attention kernel with async softmax and splitKV.
    FLOOP = DIV_UP(len_group, FLOAT_BANK_SIZE),
    Loop >= loop = DIV_UP(len, BLOCKSIZE).
    The kernel only supports KV cache len <= 6k due to the shared memory limitation.
    Layout:
        Q   [1, bs, hn, hs]
        K/V [max_len, bs, hn, hs]
*/
template <int BLOCK_SIZE, int FLOOP, int SPLIT>
__global__ void decode_mha_with_splitKV_kernel(
                half* Q, half* K, half* V, const float p_scale, const float max_val, 
                int bs, int hn, int dim, int max_len, int len, int hs, int loop, float* S, float* H) {

  unsigned int q_offset = blockIdx.x * dim + blockIdx.y * hs;
  unsigned int kv_offset = blockIdx.x * max_len * dim + blockIdx.y * hs;

  int cur_len = DIV_UP(len, SPLIT);
  unsigned int len_offset = blockIdx.z * cur_len;
  
  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  
  int len_group = DIV_UP(BLOCK_SIZE, (hs >> 3)); // make sure len_group is a power-of-two number
  //int len_span = DIV_UP(BLOCK_SIZE, hs); // must not exceed 4
  int pad_len = DIV_UP(cur_len, len_group) * len_group;

  unsigned int kv_row = tid / (hs >> 3);
  unsigned int kv_col = tid % (hs >> 3);
  unsigned int j = kv_col << 3;
  // int BK = blockDim.x * blockDim.y / (hs >> 3);

  half2 q_val[4], k_val[4], v_val[4], s_temp;
  half temp_sum = __float2half(0.0f);
  half PD[8] = {__float2half(0.0f)};

  // 4KB
  __shared__ float s_shmem[FLOOP * (FLOAT_BANK_SIZE + 8)];
  // 35KB
  __shared__ float av_shmem[MAX_LEN_GROUP][(MAX_HEAD_SIZE >> 1) + 8]; 
  __shared__ float re_shmem[(MAX_LEN_GROUP >> 5)][MAX_HEAD_SIZE + 8]; 
  // 1KB
  __shared__ float block_sum[4];

  // load Q
  *(float4*)(&q_val[0]) = *(float4*)(&Q[q_offset + j]);

  // to execute reduction in the loop
  // float qxk_result = 0.0f;
  float temp_sum_val = 0.0f;

  float a;
  float4 pv[2];

#pragma unroll
  for (int i = 0; i < 2; i++){
    pv[i].x = 0.0f;
    pv[i].y = 0.0f;
    pv[i].z = 0.0f;
    pv[i].w = 0.0f;
  }

  unsigned int s_row = kv_row / FLOAT_BANK_SIZE;
  unsigned int s_col = kv_row % FLOAT_BANK_SIZE;

  unsigned int t_row = tid / FLOAT_BANK_SIZE;
  unsigned int t_col = tid % FLOAT_BANK_SIZE;

#pragma unroll
  for (int row = kv_row; row < pad_len; row += len_group){

    // Q x K
    *(float4*)(&k_val[0]) = ((row < cur_len) && (len_offset + row < len)) ? 
      *(float4*)(&K[kv_offset + (len_offset + row) * dim + j]) : 
      *(float4*)(&PD[0]);
    *(float4*)(&v_val[0]) = ((row < cur_len) && (len_offset + row < len)) ? 
      *(float4*)(&V[kv_offset + (len_offset + row) * dim + j]) : 
      *(float4*)(&PD[0]);

    s_temp = __hmul2(q_val[0], k_val[0]);
    s_temp = __hadd2(s_temp, __hmul2(q_val[1], k_val[1]));
    s_temp = __hadd2(s_temp, __hmul2(q_val[2], k_val[2]));
    s_temp = __hadd2(s_temp, __hmul2(q_val[3], k_val[3]));

    temp_sum = __hadd(s_temp.x, s_temp.y);
    temp_sum = warpReduceSum(temp_sum, (hs >> 3));

    // now for all k_col == 0, a temp_s is stored in the register
    // push into share memory

    // softmax without scaling
    if (kv_col == 0){
      s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col] = ((row < cur_len) && (len_offset + row < len)) ? 
        __expf(__half2float(temp_sum) * p_scale - max_val) : 0.0f;
    }

    __syncthreads();
    // calculate the intermediate sum
    temp_sum_val += (tid < len_group) ? 
        s_shmem[t_row * (FLOAT_BANK_SIZE + 8) + t_col] : 0.0f;

    // AxV
    // a.x = __float2half(s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col]);
    // a.y = a.x;
    a = s_shmem[s_row * (FLOAT_BANK_SIZE + 8) + s_col];

    pv[0].x = pv[0].x + a * __half2float(v_val[0].x);
    pv[0].y = pv[0].y + a * __half2float(v_val[0].y);
    pv[0].z = pv[0].z + a * __half2float(v_val[1].x);
    pv[0].w = pv[0].w + a * __half2float(v_val[1].y);
    pv[1].x = pv[1].x + a * __half2float(v_val[2].x);
    pv[1].y = pv[1].y + a * __half2float(v_val[2].y);
    pv[1].z = pv[1].z + a * __half2float(v_val[3].x);
    pv[1].w = pv[1].w + a * __half2float(v_val[3].y);

  }

  unsigned int real_len_group = min(len_group, 32);

  unsigned int o_row = tid % len_group;
  unsigned int o_col = tid / len_group;

  // calculate the global sum
  temp_sum_val = warpReduceSum(temp_sum_val, real_len_group);

  if (tid % real_len_group == 0 && tid < len_group){
    block_sum[(tid >> 5)] = temp_sum_val; 
  }
  __syncthreads();

  if (tid == 0){
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      block_sum[0] += block_sum[r];
    }
    S[(blockIdx.x * hn + blockIdx.y) * SPLIT + blockIdx.z] = block_sum[0];
  }
  __syncthreads();

  float partial_scaling = __fdividef(1.0f, block_sum[0] + 1e-6f);

  pv[0].x = pv[0].x * partial_scaling;
  pv[0].y = pv[0].y * partial_scaling;
  pv[0].z = pv[0].z * partial_scaling;
  pv[0].w = pv[0].w * partial_scaling;
  pv[1].x = pv[1].x * partial_scaling;
  pv[1].y = pv[1].y * partial_scaling;
  pv[1].z = pv[1].z * partial_scaling;
  pv[1].w = pv[1].w * partial_scaling;

  // transpoisition allows efficient mem access of V matrix
  // first 64
  *(float4*)(&av_shmem[kv_row][(kv_col << 2)]) = *(float4*)(&pv[0]);
  __syncthreads();

  *(float4*)(&pv[0]) = *(float4*)(&av_shmem[o_row][(o_col << 2)]);

  pv[0].x = warpReduceSum(pv[0].x, real_len_group);
  pv[0].y = warpReduceSum(pv[0].y, real_len_group);
  pv[0].z = warpReduceSum(pv[0].z, real_len_group);
  pv[0].w = warpReduceSum(pv[0].w, real_len_group);

  if (o_row % 32 == 0){
    *(float4*)(&re_shmem[(o_row >> 5)][(o_col << 3)]) = *(float4*)(&pv[0]);
  }

  __syncthreads();

  // another 64
  *(float4*)(&av_shmem[kv_row][(kv_col << 2)]) = *(float4*)(&pv[1]);
  __syncthreads();

  *(float4*)(&pv[1]) = *(float4*)(&av_shmem[o_row][(o_col << 2)]);
  // __syncthreads();

  pv[1].x = warpReduceSum(pv[1].x, real_len_group);
  pv[1].y = warpReduceSum(pv[1].y, real_len_group);
  pv[1].z = warpReduceSum(pv[1].z, real_len_group);
  pv[1].w = warpReduceSum(pv[1].w, real_len_group);

  if (o_row % 32 == 0){
    *(float4*)(&re_shmem[(o_row >> 5)][(o_col << 3) + 4]) = *(float4*)(&pv[1]);
  }

  __syncthreads();

  if (kv_row == 0){
      pv[0].x = re_shmem[0][(kv_col << 3) + 0];
      pv[0].y = re_shmem[0][(kv_col << 3) + 1];
      pv[0].z = re_shmem[0][(kv_col << 3) + 2];
      pv[0].w = re_shmem[0][(kv_col << 3) + 3];
      pv[1].x = re_shmem[0][(kv_col << 3) + 4];
      pv[1].y = re_shmem[0][(kv_col << 3) + 5];
      pv[1].z = re_shmem[0][(kv_col << 3) + 6];
      pv[1].w = re_shmem[0][(kv_col << 3) + 7];
  #pragma unroll
    for (int r = 1; r < (len_group >> 5); r++){
      pv[0].x = pv[0].x + re_shmem[r][(kv_col << 3) + 0];
      pv[0].y = pv[0].y + re_shmem[r][(kv_col << 3) + 1];
      pv[0].z = pv[0].z + re_shmem[r][(kv_col << 3) + 2];
      pv[0].w = pv[0].w + re_shmem[r][(kv_col << 3) + 3];
      pv[1].x = pv[1].x + re_shmem[r][(kv_col << 3) + 4];
      pv[1].y = pv[1].y + re_shmem[r][(kv_col << 3) + 5];
      pv[1].z = pv[1].z + re_shmem[r][(kv_col << 3) + 6];
      pv[1].w = pv[1].w + re_shmem[r][(kv_col << 3) + 7];
    } 
    *(float4*)(&H[q_offset * SPLIT + blockIdx.z * hs + (kv_col << 3)]) = *(float4*)(&pv[0]);
    *(float4*)(&H[q_offset * SPLIT + blockIdx.z * hs + (kv_col << 3) + 4]) = *(float4*)(&pv[1]);
  }
}


template <int SPLIT>
__global__ void decode_splitKV_scaling_kernel(float* H_F, int dim, int hs, int hn, float* S, half* H){

  // HF: [bs, 1, hn, SPLIT, hs]
  // S: [bs, hn, SPLIT]

  unsigned int h_offset = blockIdx.x * dim + blockIdx.y * hs;
  unsigned int tid = threadIdx.x;

  float real_sum = 0.0f;
  float partial_sum[SPLIT] = {0.0f}; 
#pragma unroll
  for (int s = 0; s < SPLIT; s++){
    partial_sum[s] = S[(blockIdx.x * hn + blockIdx.y) * SPLIT + s];
    real_sum += partial_sum[s];
  }

  float real_value = 0.0f;
#pragma unroll
  for (int s = 0; s < SPLIT; s++){
    float res = H_F[h_offset * SPLIT + s * hs + tid];
    float scaling = __fdividef(partial_sum[s], real_sum);
    real_value += res * scaling;
  }

  H[h_offset + tid] = __float2half(real_value);
}
