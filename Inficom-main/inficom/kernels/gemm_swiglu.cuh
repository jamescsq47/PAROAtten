#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <mma.h>

#include "utils.h"

/*
    F.silu(linear1(x)) * linear2(x). GEMV version. With weights separated.
*/
__global__ __forceinline__ void dual_fast_gemv_acc_fp16_silu_dot_kernel(
                                half* x, half* w1, half* w2, 
                                int bs, int dim, int h_dim, 
                                half* res) {
  
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.x * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  half2 x_val[4];
  half2 w1_val[4];
  half2 w2_val[4];

  // half2 temp_sum = {__float2half(0.0f), __float2half(0.0f)};
  half2 temp_sum_1 = {__float2half(0.0f), __float2half(0.0f)};
  half2 temp_sum_2 = {__float2half(0.0f), __float2half(0.0f)};

#pragma unroll
  for (int iter = 0; iter < DIV_UP((dim >> 3), blockDim.x); iter++) {
    unsigned int j = (start_idx + iter * blockDim.x) << 3;
    if (j >= dim) {break;}
      
      *(float4*)(&x_val[0]) = *(float4*)(&x[j]);
      *(float4*)(&w1_val[0]) = *(float4*)(&w1[row * dim + j]);
      *(float4*)(&w2_val[0]) = *(float4*)(&w2[row * dim + j]);

      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[0],  w1_val[0]));  
      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[1],  w1_val[1]));
      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[2],  w1_val[2]));
      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[3],  w1_val[3])); 

      temp_sum_2 = __hadd2(temp_sum_2, __hmul2(x_val[0],  w2_val[0]));  
      temp_sum_2 = __hadd2(temp_sum_2, __hmul2(x_val[1],  w2_val[1]));
      temp_sum_2 = __hadd2(temp_sum_2, __hmul2(x_val[2],  w2_val[2]));
      temp_sum_2 = __hadd2(temp_sum_2, __hmul2(x_val[3],  w2_val[3])); 
  }

  float sum_1 = __half2float(__hadd(temp_sum_1.x, temp_sum_1.y));
  float sum_2 = __half2float(__hadd(temp_sum_2.x, temp_sum_2.y));

  static __shared__ float warpLevelSums[WARP_SIZE];

  sum_1 = blockReduceSum(sum_1, warpLevelSums);
  sum_2 = blockReduceSum(sum_2, warpLevelSums);

  if (tid == 0) {
    sum_1 = sum_1 * __fdividef(1.0f, __expf(-1 * sum_1) + 1.0f);
    res[row] = __float2half(sum_1 * sum_2);
  }
}


/*
    F.silu(linear1(x)) * linear2(x). GEMV version. With weights combined.
*/
__global__ __forceinline__ void fast_gemv_acc_fp16_swiglu_kernel(
                                half* x, half* w,
                                int bs, int dim, int h_dim, 
                                half* res) {
  
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.x * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;

  half2 x_val[4];
  half2 w1_val[4];
  half2 w2_val[4];

  // half2 temp_sum = {__float2half(0.0f), __float2half(0.0f)};
  half2 temp_sum_1 = {__float2half(0.0f), __float2half(0.0f)};
  half2 temp_sum_2 = {__float2half(0.0f), __float2half(0.0f)};

#pragma unroll
  for (int iter = 0; iter < DIV_UP((dim >> 3), blockDim.x); iter++) {
    unsigned int j = (start_idx + iter * blockDim.x) << 3;
    if (j >= dim) {break;}
      
      // float4 vec_val = vec4[j];
      // float4 mat_val = mat4[row * (n >> 3) + j];
      *(float4*)(&x_val[0]) = *(float4*)(&x[j]);
      *(float4*)(&w1_val[0]) = *(float4*)(&w[row * dim + j]);
      *(float4*)(&w2_val[0]) = *(float4*)(&w[(row + h_dim) * dim + j]);

      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[0],  w1_val[0]));  
      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[1],  w1_val[1]));
      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[2],  w1_val[2]));
      temp_sum_1 = __hadd2(temp_sum_1, __hmul2(x_val[3],  w1_val[3]));

      temp_sum_2 = __hadd2(temp_sum_2, __hmul2(x_val[0],  w2_val[0]));  
      temp_sum_2 = __hadd2(temp_sum_2, __hmul2(x_val[1],  w2_val[1]));
      temp_sum_2 = __hadd2(temp_sum_2, __hmul2(x_val[2],  w2_val[2]));
      temp_sum_2 = __hadd2(temp_sum_2, __hmul2(x_val[3],  w2_val[3])); 
  }

  float sum_1 = __half2float(__hadd(temp_sum_1.x, temp_sum_1.y));
  float sum_2 = __half2float(__hadd(temp_sum_2.x, temp_sum_2.y));

  static __shared__ float warpLevelSums[WARP_SIZE];

  sum_1 = blockReduceSum(sum_1, warpLevelSums);
  sum_2 = blockReduceSum(sum_2, warpLevelSums);

  if (tid == 0) {
    sum_1 = sum_1 * __fdividef(1.0f, __expf(-1 * sum_1) + 1.0f);
    res[row] = __float2half(sum_1 * sum_2);
  }
}


using namespace nvcuda;
/*
    F.silu(linear1(x)) * linear2(x). GEMM version. With weights combined.
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void dual_flat_gemm_m8n32k256x8_bz1_silu_dot_kernel(
    const half * __restrict__ a, const half * __restrict__ b0, 
    const half * __restrict__ b1, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half s_a[BM * (LDK)];
    __shared__ half s_b_0[BN * (LDK)];
    __shared__ half s_b_1[BN * (LDK)];
   
    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c[2];

    wmma::fill_fragment(frag_c[0], __float2half(0.0f));
    wmma::fill_fragment(frag_c[1], __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   
    int load_ab_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   

    int load_a_smem_addr_0 = __cvta_generic_to_shared(s_a) + OFFSET(load_a_smem_m, load_ab_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs_0[4];
    int load_b_smem_addrs_1[4];
    #pragma unroll
    for(int i=0; i<4; i++){
        load_b_smem_addrs_0[i] = __cvta_generic_to_shared(s_b_0) + (OFFSET(load_b_smem_n, load_ab_smem_k, LDK) + i * (LDK)) * sizeof(half);
        load_b_smem_addrs_1[i] = __cvta_generic_to_shared(s_b_1) + (OFFSET(load_b_smem_n, load_ab_smem_k, LDK) + i * (LDK)) * sizeof(half);
    }

    int load_a_gmem_addr = OFFSET(load_a_smem_m, (k_start + load_ab_smem_k), K);
    int load_b_gmem_addr = OFFSET((bx * BN + load_b_smem_n), (k_start + load_ab_smem_k), K);

    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++ ) {
        if (load_a_smem_m < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs_0[i]), "l"(&b0[load_b_gmem_addr + i * K]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs_1[i]), "l"(&b1[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b[0], &s_b_0[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c[0], frag_a, frag_b[0], frag_c[0]);
            wmma::load_matrix_sync(frag_b[1], &s_b_1[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c[1], frag_a, frag_b[1], frag_c[1]);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&s_b_0[wid * 8 * LDN], frag_c[0], LDN, wmma::mem_row_major);
    wmma::store_matrix_sync(&s_b_1[wid * 8 * LDN], frag_c[1], LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_n = tid & 31;   // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(load_a_smem_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(load_a_smem_m, bx * BN + shmem_c_n, N);

    float res[2] = {0.0f};
    
#pragma unroll
    for (int s = 0; s < 8; s++){
        res[0] += __half2float(s_b_0[shmem_c_addr + s * 8 * LDN]);
        res[1] += __half2float(s_b_1[shmem_c_addr + s * 8 * LDN]);
    }

    if (load_a_smem_m < M) {
        c[gmem_c_addr] = __float2half(res[0] * __fdividef(1.0f, __expf(-1 * res[0]) + 1.0f) * res[1]);
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void dual_flat_gemm_m16n32k256x8_bz1_silu_dot_kernel(
    const half * __restrict__ a, const half * __restrict__ b0, 
    const half * __restrict__ b1, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half s_a[BM * (LDK)];
    __shared__ half s_b_0[BN * (LDK)];
    __shared__ half s_b_1[BN * (LDK)];

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[2];

    wmma::fill_fragment(frag_c[0], __float2half(0.0f));
    wmma::fill_fragment(frag_c[1], __float2half(0.0f));

    int load_a_smem_m = (tid >> 5) << 1; // 0 ~ 14   | 0 2  4 ... 14  
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248 
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr_0 = __cvta_generic_to_shared(s_b_0);
    size_t s_b_base_addr_1 = __cvta_generic_to_shared(s_b_1);

    int load_a_smem_addrs[2];
    #pragma unroll
    for (int i=0; i<2; i++){
        load_a_smem_addrs[i] = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    }
    int load_b_smem_addrs_0[4];
    int load_b_smem_addrs_1[4];
    #pragma unroll
    for (int i=0; i<4; i++){
        load_b_smem_addrs_0[i] = s_b_base_addr_0 + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
        load_b_smem_addrs_1[i] = s_b_base_addr_1 + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    }

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++ ) {
        
        #pragma unroll
        for(int i=0; i<2; i++)
        {   
            if ((load_a_gmem_m + i) < M) {
                asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                    : "r"(load_a_smem_addrs[i]), "l"(&a[load_a_gmem_addr + i * K]));
            }
        }

        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs_0[i]), "l"(&b0[load_b_gmem_addr + i * K]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs_1[i]), "l"(&b1[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<4; i++){
            wmma::load_matrix_sync(frag_a, &s_a[(wid & 3) * 16 + 64 * i], LDK);
            wmma::load_matrix_sync(frag_b[0], &s_b_0[(wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
            wmma::mma_sync(frag_c[0], frag_a, frag_b[0], frag_c[0]);
            wmma::load_matrix_sync(frag_b[1], &s_b_1[(wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
            wmma::mma_sync(frag_c[1], frag_a, frag_b[1], frag_c[1]);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&s_b_0[(wid & 3) * 16 * LDN + (wid >> 2) * 16], frag_c[0], LDN, wmma::mem_row_major);
    wmma::store_matrix_sync(&s_b_1[(wid & 3) * 16 * LDN + (wid >> 2) * 16], frag_c[1], LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 4;           // 0, 1, 2, 3, 4, ..., 15
    int shmem_c_n = (tid & 15) << 1;    // 0, 2, 4, 6, 8, ..., 30
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

    if (shmem_c_m < M) {
        #pragma unroll
        for (int s = 1; s < 4; s++){
            *(half2*)(&s_b_0[shmem_c_addr]) = __hadd2(*(half2*)(&s_b_0[shmem_c_addr]), 
                *(half2*)(&s_b_0[shmem_c_addr + s * 16 * LDN]));
            *(half2*)(&s_b_1[shmem_c_addr]) = __hadd2(*(half2*)(&s_b_1[shmem_c_addr]), 
                *(half2*)(&s_b_1[shmem_c_addr + s * 16 * LDN]));
        }
        float2 to_store_1 = __half22float2(*(half2*)(&s_b_0[shmem_c_addr]));
        float2 to_store_2 = __half22float2(*(half2*)(&s_b_1[shmem_c_addr]));
        to_store_1.x = to_store_1.x * __fdividef(1.0f, __expf(-1 * to_store_1.x) + 1.0f) * to_store_2.x;
        to_store_1.y = to_store_1.y * __fdividef(1.0f, __expf(-1 * to_store_1.y) + 1.0f) * to_store_2.y;
        *(half2*)(&c[gmem_c_addr]) = __float22half2_rn(to_store_1);
    }
}


/*
    F.silu(linear1(x)) * linear2(x). GEMM version. With weights combined.
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m8n32k256x8_bz1_swiglu_kernel(
    const half * __restrict__ a, const half * __restrict__ b, 
    half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half s_a[BM * (LDK)];
    __shared__ half s_b_0[BN * (LDK)];
    __shared__ half s_b_1[BN * (LDK)];

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c[2];

    wmma::fill_fragment(frag_c[0], __float2half(0.0f));
    wmma::fill_fragment(frag_c[1], __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   
    int load_ab_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248  
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   

    int load_a_smem_addr_0 = __cvta_generic_to_shared(s_a) + OFFSET(load_a_smem_m, load_ab_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs_0[4];
    int load_b_smem_addrs_1[4];
    #pragma unroll
    for(int i=0; i<4; i++){
        load_b_smem_addrs_0[i] = __cvta_generic_to_shared(s_b_0) + (OFFSET(load_b_smem_n, load_ab_smem_k, LDK) + i * (LDK)) * sizeof(half);
        load_b_smem_addrs_1[i] = __cvta_generic_to_shared(s_b_1) + (OFFSET(load_b_smem_n, load_ab_smem_k, LDK) + i * (LDK)) * sizeof(half);
    }

    int load_a_gmem_addr = OFFSET(load_a_smem_m, (k_start + load_ab_smem_k), K);
    int load_b_gmem_addr_0 = OFFSET((bx * BN + load_b_smem_n), (k_start + load_ab_smem_k), K);
    int load_b_gmem_addr_1 = OFFSET(N + (bx * BN + load_b_smem_n), (k_start + load_ab_smem_k), K);

    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++ ) {
        if (load_a_smem_m < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs_0[i]), "l"(&b[load_b_gmem_addr_0 + i * K]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs_1[i]), "l"(&b[load_b_gmem_addr_1 + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b[0], &s_b_0[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c[0], frag_a, frag_b[0], frag_c[0]);
            wmma::load_matrix_sync(frag_b[1], &s_b_1[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c[1], frag_a, frag_b[1], frag_c[1]);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr_0 += BK;
        load_b_gmem_addr_1 += BK;
    }

    wmma::store_matrix_sync(&s_b_0[wid * 8 * LDN], frag_c[0], LDN, wmma::mem_row_major);
    wmma::store_matrix_sync(&s_b_1[wid * 8 * LDN], frag_c[1], LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_n = tid & 31;   // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(load_a_smem_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(load_a_smem_m, bx * BN + shmem_c_n, N);

    float res[2] = {0.0f};
    
#pragma unroll
    for (int s = 0; s < 8; s++){
        res[0] += __half2float(s_b_0[shmem_c_addr + s * 8 * LDN]);
        res[1] += __half2float(s_b_1[shmem_c_addr + s * 8 * LDN]);
    }

    if (load_a_smem_m < M) {
      c[gmem_c_addr] = __float2half(res[0] * __fdividef(1.0f, __expf(-1 * res[0]) + 1.0f) * res[1]);
    }
}
