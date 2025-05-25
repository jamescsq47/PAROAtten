#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <mma.h>

#include "utils.h"

using namespace nvcuda;

/*
    GEMM kernel with RoPE defined in Llama2, using FP16 to accumulate.
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m8n32k256x8_bz1_with_llama2_rope_kernel(
    half * __restrict__ a, half * __restrict__ bq, 
    half * __restrict__ bk, half * __restrict__ bv,  
    float* f, half* q, half* k, half* v, 
    int bs, int max_len, int len, int dim, int hn, int hs) {
#if __CUDA_ARCH__ < 800
    return;
#endif

    int M = bs;
    int K = dim;
    int N = 3 * dim;

    int bx = blockIdx.x;
    int bz = blockIdx.z;
    // int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248(32个数)  
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = load_a_smem_k;
    int load_b_gmem_k = load_b_smem_k;

    half* weight_ptr = (load_b_gmem_n < dim) ? &bq[0] :
                      (load_b_gmem_n < 2 * dim) ? &bk[0] : &bv[0];
    load_b_gmem_n = load_b_gmem_n % dim;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    // single buffer loop
    for (int bk = 0; bk < K / BK; bk++ ) {
        if (load_a_gmem_m < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(weight_ptr + load_b_gmem_addr + i * K));
            //     : "r"(load_b_smem_addrs[i]), "l"(ptr_b));
            // ptr_b += K;
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&smem[wid * 8 * LDN], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 5;   // 0, 1, 2, 3, 4, 5, 6, 7
    int shmem_c_n = tid & 31;   // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);

    int store_c_m = tid >> 4;   // 0, 1, 2, 3, 4, ..., 15
    int store_c_n = (tid & 15) << 1;  // 0, 2, 4, ..., 30
    int store_c_gmem_n = bx * BN + store_c_n;
    int store_c_addr = OFFSET(store_c_m, store_c_n, LDN);

    // if (wid == 0){
#pragma unroll
    for (int s = 1; s < 8; s++){
      smem[shmem_c_addr] = __hadd(smem[shmem_c_addr], smem[shmem_c_addr + s * 8 * LDN]);
    }
    // }
    __syncthreads();

    if (store_c_m >= M){return;}
    if (store_c_gmem_n < dim){
      int idx = store_c_gmem_n % hs;
      float2 to_rotate, res, gres;
      to_rotate = *(float2*)(&f[idx]);
      res = __half22float2(*(half2*)(&smem[store_c_addr]));
      gres.x = to_rotate.x * res.x - to_rotate.y * res.y;
      gres.y = to_rotate.x * res.y + to_rotate.y * res.x;
      *((half2 *)((&q[store_c_m * dim + 
        store_c_gmem_n]))) = __float22half2_rn(gres);
    }
    else if (store_c_gmem_n < 2 * dim){
      int idx = store_c_gmem_n % hs;
      float2 to_rotate, res, gres;
      to_rotate = *(float2*)(&f[idx]);
      res = __half22float2(*(half2*)(&smem[store_c_addr]));
      gres.x = to_rotate.x * res.x - to_rotate.y * res.y;
      gres.y = to_rotate.x * res.y + to_rotate.y * res.x;
      *((half2 *)(&k[store_c_m * (max_len * dim) + 
        len * dim + (store_c_gmem_n % dim)])) = __float22half2_rn(gres);
    }
    else if (store_c_gmem_n < 3 * dim){
      *((half2 *)(&v[store_c_m * (max_len * dim) + 
        len * dim + (store_c_gmem_n % dim)])) = 
          *(half2*)(&smem[store_c_addr]);
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n32k256x8_db_with_llama2_rope_kernel(
    half * __restrict__ a, half * __restrict__ bq, 
    half * __restrict__ bk, half * __restrict__ bv,
    float* f, half* q, half* k, half* v,
    int bs, int max_len, int len, int dim, int hn, int hs) {
#if __CUDA_ARCH__ < 800
    return;
#endif

    int M = bs;
    int K = dim;
    int N = 3 * dim;

    int bx = blockIdx.x;
    int bz = blockIdx.z;
    // int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * LDK;
    int s_a_offset = BM * LDK;
    int s_b_offset = BN * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5) << 1; // 0 ~ 14   | 0 2  4 ... 14  
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248 
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addrs[2];
    #pragma unroll
    for (int i=0; i<2; i++){
        load_a_smem_addrs[i] = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    }
    int load_b_smem_addrs[4];
    #pragma unroll
    for (int i=0; i<4; i++){
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    }

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = load_a_smem_k;
    int load_b_gmem_k = load_b_smem_k;

    half* weight_ptr = (load_b_gmem_n < dim) ? &bq[0] :
                      (load_b_gmem_n < 2 * dim) ? &bk[0] : &bv[0];
    load_b_gmem_n = load_b_gmem_n % dim;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    {
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
                : "r"(load_b_smem_addrs[i]), "l"(weight_ptr + load_b_gmem_addr + i * K));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    #pragma unroll
    for (int bk = 1; bk < (K / gridDim.z) / BK; bk++ ) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
        
        #pragma unroll
        for(int i=0; i<2; i++)
        {   
            if ((load_a_gmem_m + i) < M) {
                asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                    : "r"(load_a_smem_addrs[i] + smem_sel_next * s_a_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr + i * K]));
            }
        }

        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i] + smem_sel_next * s_b_offset * (int)sizeof(half)), "l"(weight_ptr + load_b_gmem_addr + i * K));
        }

        for(int i=0; i<4; i++){
            wmma::load_matrix_sync(frag_a, &s_a[smem_sel * s_a_offset + (wid & 3) * 16 + 64 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[smem_sel * s_b_offset + (wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    int smem_sel = (((K / gridDim.z) / BK) & 1) ^ 1;

    for(int i=0; i<4; i++){
        wmma::load_matrix_sync(frag_a, &s_a[smem_sel * s_a_offset + (wid & 3) * 16 + 64 * i], LDK);
        wmma::load_matrix_sync(frag_b, &s_b[smem_sel * s_b_offset + (wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    wmma::store_matrix_sync(&smem[(wid & 3) * 16 * LDN + (wid >> 2) * 16], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 4;           // 0, 1, 2, 3, 4, ..., 15
    int shmem_c_n = (tid & 15) << 1;    // 0, 2, 4, 6, 8, ..., 30
    int gmem_c_n = bx * BN + shmem_c_n;
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, gmem_c_n, N);

    if (shmem_c_m >= M){return;}
#pragma unroll
    for (int s = 1; s < 4; s++){
        *(half2*)(&smem[shmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), 
            *(half2*)(&smem[shmem_c_addr + s * 16 * LDN]));
    }

    if (gmem_c_n < dim){
        int idx = gmem_c_n % hs;
        float2 to_rotate, res, gres;
        to_rotate = *(float2*)(&f[idx]);
        res = __half22float2(*(half2*)(&smem[shmem_c_addr]));
        gres.x = to_rotate.x * res.x - to_rotate.y * res.y;
        gres.y = to_rotate.x * res.y + to_rotate.y * res.x;
        *((half2 *)((&q[shmem_c_m * dim + 
            gmem_c_n]))) = __float22half2_rn(gres);
    }
    else if (gmem_c_n < 2 * dim){
        int idx = gmem_c_n % hs;
        float2 to_rotate, res, gres;
        to_rotate = *(float2*)(&f[idx]);
        res = __half22float2(*(half2*)(&smem[shmem_c_addr]));
        gres.x = to_rotate.x * res.x - to_rotate.y * res.y;
        gres.y = to_rotate.x * res.y + to_rotate.y * res.x;
        *((half2 *)(&k[shmem_c_m * (max_len * dim) + 
            len * dim + (gmem_c_n % dim)])) = __float22half2_rn(gres);
    }
    else if (gmem_c_n < 3 * dim){
        *((half2 *)(&v[shmem_c_m * (max_len * dim) + 
            len * dim + (gmem_c_n % dim)])) = 
            *(half2*)(&smem[shmem_c_addr]);
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n64k128x8_db_with_llama2_rope_kernel(
    half * __restrict__ a, half * __restrict__ bq, 
    half * __restrict__ bk, half * __restrict__ bv,  
    float* f, half* Q, half* Kcache, half* Vcache, 
    int m, int k, int n, int kv_stride_bs, int kv_stride_seq,
    int len, int hn, int hs) {
#if __CUDA_ARCH__ < 800
    return;
#endif

    int M = m;
    int K = k;
    int N = 3 * n;

    int bx = blockIdx.x;
    int bz = blockIdx.z;
    // int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    // extern __shared__ half smem[];
    __shared__ half smem[2 * (BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * LDK;
    int s_a_offset = BM * LDK;
    int s_b_offset = BN * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 4);      // 0 ~ 15   | 0 1  2 ... 15  
    int load_a_smem_k = (tid & 15) << 3; // 0 ~ 120  | 0 8 16 ... 120
    int load_b_smem_n = (tid >> 4) << 2; // 0 ~ 60   | 0 4  8 ... 60   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    // #pragma unroll
    // for (int i=0; i<2; i++){
    //     load_a_smem_addrs[i] = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    // }
    int load_b_smem_addrs[4];
    #pragma unroll
    for (int i=0; i<4; i++){
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    }

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = load_a_smem_k;
    int load_b_gmem_k = load_b_smem_k;

    half* weight_ptr = (load_b_gmem_n < n) ? &bq[0] :
                      (load_b_gmem_n < 2 * n) ? &bk[0] : &bv[0];
    load_b_gmem_n = load_b_gmem_n % n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    {
        if ((load_a_gmem_m) < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                    : "r"(load_a_smem_addr), "l"(&a[load_a_gmem_addr]));
        }
        // }

        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(weight_ptr + load_b_gmem_addr + i * K));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    #pragma unroll
    for (int bk = 1; bk < (K / gridDim.z) / BK; bk++ ) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
        
        // #pragma unroll
        // for(int i=0; i<2; i++)
        // {   
        if ((load_a_gmem_m) < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                    : "r"(load_a_smem_addr + smem_sel_next * s_a_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr]));
        }
        // }

        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i] + + smem_sel_next * s_b_offset * (int)sizeof(half)), "l"(weight_ptr + load_b_gmem_addr + i * K));
        }

        for(int i=0; i<4; i++){
            wmma::load_matrix_sync(frag_a, &s_a[smem_sel * s_a_offset + (wid >> 2) * 16 + 32 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[smem_sel * s_b_offset + (wid & 3) * 16 * LDK + (wid >> 2) * 16 + 32 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    int smem_sel = (((K / gridDim.z) / BK) & 1) ^ 1;

    for(int i=0; i<4; i++){
        wmma::load_matrix_sync(frag_a, &s_a[smem_sel * s_a_offset + (wid >> 2) * 16 + 32 * i], LDK);
        wmma::load_matrix_sync(frag_b, &s_b[smem_sel * s_b_offset + (wid & 3) * 16 * LDK + (wid >> 2) * 16 + 32 * i], LDK);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    __syncthreads();

    wmma::store_matrix_sync(&smem[(wid >> 2) * 16 * LDN + (wid & 3) * 16], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 4;           // 0, 1, 2, 3, 4, ..., 15
    int shmem_c_n = (tid & 15) << 2;    // 0, 4, 8,12,16, ..., 60
    int gmem_c_n = bx * BN + shmem_c_n;
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, gmem_c_n, N);

    if (shmem_c_m >= M){return;}
// #pragma unroll
//     for (int s = 1; s < 4; s++){
    *(half2*)(&smem[shmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), 
            *(half2*)(&smem[shmem_c_addr + 16 * LDN]));
    *(half2*)(&smem[(shmem_c_addr + 2)]) = __hadd2(*(half2*)(&smem[(shmem_c_addr + 2)]), 
            *(half2*)(&smem[(shmem_c_addr + 2) + 16 * LDN]));
//     }

    if (gmem_c_n < n){
        int idx = gmem_c_n % hs;
        float2 to_rotate, res, gres;
        to_rotate = *(float2*)(&f[idx]);
        res = __half22float2(*(half2*)(&smem[shmem_c_addr]));
        gres.x = to_rotate.x * res.x - to_rotate.y * res.y;
        gres.y = to_rotate.x * res.y + to_rotate.y * res.x;
        *((half2 *)((&Q[shmem_c_m * n + 
            gmem_c_n]))) = __float22half2_rn(gres);
        to_rotate = *(float2*)(&f[idx + 2]);
        res = __half22float2(*(half2*)(&smem[shmem_c_addr + 2]));
        gres.x = to_rotate.x * res.x - to_rotate.y * res.y;
        gres.y = to_rotate.x * res.y + to_rotate.y * res.x;
        *((half2 *)((&Q[shmem_c_m * n + 
            gmem_c_n + 2]))) = __float22half2_rn(gres);
    }
    else if (gmem_c_n < 2 * n){
        int idx = gmem_c_n % hs;
        float2 to_rotate, res, gres;
        to_rotate = *(float2*)(&f[idx]);
        res = __half22float2(*(half2*)(&smem[shmem_c_addr]));
        gres.x = to_rotate.x * res.x - to_rotate.y * res.y;
        gres.y = to_rotate.x * res.y + to_rotate.y * res.x;
        *((half2 *)(&Kcache[shmem_c_m * kv_stride_bs + 
            len * kv_stride_seq + (gmem_c_n % n)])) = __float22half2_rn(gres);
        to_rotate = *(float2*)(&f[idx + 2]);
        res = __half22float2(*(half2*)(&smem[shmem_c_addr + 2]));
        gres.x = to_rotate.x * res.x - to_rotate.y * res.y;
        gres.y = to_rotate.x * res.y + to_rotate.y * res.x;
        *((half2 *)(&Kcache[shmem_c_m * kv_stride_bs + 
            len * kv_stride_seq + ((gmem_c_n + 2) % n)])) = __float22half2_rn(gres);
    }
    else if (gmem_c_n < 3 * n){
        *((float2 *)(&Vcache[shmem_c_m * kv_stride_bs + 
            len * kv_stride_seq + (gmem_c_n % n)])) = 
            *(float2*)(&smem[shmem_c_addr]);
    }
}


/*
    GEMM kernel with RoPE defined in ChatGLM2, using FP16 to accumulate.
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m8n32k256x8_bz1_with_chatglm2_rope_kernel(
    half * __restrict__ a, half * __restrict__ wqkv, half * __restrict__ bqkv, 
    half* f, half* q, half* k, half* v, 
    int bs, int max_len, int len, int dim, int hn, int gn, int hs) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    // [hongke @ 10.25: for GLM2-6B
    //   dim = 4096, hn = 32, gn = 2, hs = 128, gs = 256]

    int gs = gn * hs;

    int M = bs;
    int K = dim;
    int N = dim + 2 * gs;

    int bx = blockIdx.x;
    int bz = blockIdx.z;
    // int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248(32个数)  
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = load_a_smem_k;
    int load_b_gmem_k = load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    // single buffer loop
    for (int bk = 0; bk < K / BK; bk++ ) {
        if (load_a_gmem_m < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&wqkv[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&smem[wid * 8 * LDN], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 5;   // 0, 1, 2, 3, 4, 5, 6, 7
    int shmem_c_n = tid & 31;   // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);

    int store_c_m = tid >> 4;   // 0, 1, 2, 3, 4, ..., 15
    int store_c_n = (tid & 15) << 1;  // 0, 2, 4, ..., 30
    int store_c_gmem_n = bx * BN + store_c_n;
    int store_c_addr = OFFSET(store_c_m, store_c_n, LDN);

    // if (wid == 0){
#pragma unroll
    for (int s = 1; s < 8; s++){
      smem[shmem_c_addr] = __hadd(smem[shmem_c_addr], smem[shmem_c_addr + s * 8 * LDN]);
    }
    // }
    __syncthreads();

    if (store_c_m >= M){return;}
    // add bias
    float2 bias = __half22float2(*(half2*)(&bqkv[store_c_gmem_n]));
    float2 res = __half22float2(*(half2*)(&smem[store_c_addr]));
    res.x += bias.x;
    res.y += bias.y;
    // RoPE
    if (store_c_gmem_n < dim){
      int idx = store_c_gmem_n % hs;
      if (idx < hs / 2){
        float2 update_res = res;
        float2 to_rotate = __half22float2(*(half2*)(&f[idx]));
        update_res.x = to_rotate.x * res.x - to_rotate.y * res.y;
        update_res.y = to_rotate.x * res.y + to_rotate.y * res.x;
        *((half2 *)((&q[store_c_m * dim + 
          store_c_gmem_n]))) = __float22half2_rn(update_res);
      }
      else{
        *((half2 *)((&q[store_c_m * dim + 
          store_c_gmem_n]))) = __float22half2_rn(res);
      }
    }
    else if (store_c_gmem_n < (dim + gs)){
      int idx = store_c_gmem_n % hs;
      if (idx < hs / 2){
        float2 update_res = res;
        float2 to_rotate = __half22float2(*(half2*)(&f[idx]));
        update_res.x = to_rotate.x * res.x - to_rotate.y * res.y;
        update_res.y = to_rotate.x * res.y + to_rotate.y * res.x;
        *((half2 *)(&k[len * (bs * gs) + store_c_m * (gs) + 
          (store_c_gmem_n % dim)])) = __float22half2_rn(update_res);
      }
      else{
        *((half2 *)(&k[len * (bs * gs) + store_c_m * (gs) + 
          (store_c_gmem_n % dim)])) = __float22half2_rn(res);
      }
    }
    else if (store_c_gmem_n < (dim + 2 * gs)){
      *((half2 *)(&v[len * (bs * gs) + store_c_m * (gs) + 
        (store_c_gmem_n % dim % (gs))])) = __float22half2_rn(res);
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m8n32k256x8_bz1_with_bias_kernel(
    const half * __restrict__ a, const half * __restrict__ b, half * __restrict__ c, half * __restrict__ bias,
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

    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248(32个数)  
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++ ) {
        if (load_a_gmem_m < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&smem[wid * 8 * LDN], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 5;   // 0, 1, 2, 3, 4, 5, 6, 7
    int shmem_c_n = tid & 31;   // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

    
#pragma unroll
    for (int s = 1; s < 8; s++){
        smem[shmem_c_addr] = __hadd(smem[shmem_c_addr], smem[shmem_c_addr + s * 8 * LDN]);
    }

    __syncthreads();

    if (shmem_c_m < M) {
        c[gmem_c_addr] = __hadd(smem[shmem_c_addr], bias[gmem_c_addr % N]);
    }
}


/*
    GEMM kernel using FP16 to accumulate. For QKV Proj with separate weights and bias. 
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m8n32k256x8_bz1_for_qkv_proj_kernel(
    half * __restrict__ a, half * __restrict__ bq, half * __restrict__ bk, half * __restrict__ bv, 
    half* q, half* k, half* v, half* biasq, half* biask, half* biasv, int bs, int max_len, int len, int dim, int hn, int hs) {
#if __CUDA_ARCH__ < 800
    return;
#endif

    int M = bs;
    int K = dim;
    int N = 3 * dim;

    int bx = blockIdx.x;
    int bz = blockIdx.z;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN)
        return;

    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28  
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = load_a_smem_k;
    int load_b_gmem_k = load_b_smem_k;

    half* weight_ptr = (load_b_gmem_n < dim) ? &bq[0] :
                      (load_b_gmem_n < 2 * dim) ? &bk[0] : &bv[0];
    load_b_gmem_n = load_b_gmem_n % dim;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    // single buffer loop
    for (int bk = 0; bk < K / BK; bk++ ) {
        if (load_a_gmem_m < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(weight_ptr + load_b_gmem_addr + i * K));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

    
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;

    }

    wmma::store_matrix_sync(&smem[wid * 8 * LDN], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 5;   // 0, 1, 2, 3, 4, 5, 6, 7
    int shmem_c_n = tid & 31;   // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);

    int store_c_m = tid >> 4;   // 0, 1, 2, 3, 4, ..., 15
    int store_c_n = (tid & 15) << 1;  // 0, 2, 4, ..., 30
    int store_c_gmem_n = bx * BN + store_c_n;
    int store_c_addr = OFFSET(store_c_m, store_c_n, LDN);

#pragma unroll
    for (int s = 1; s < 8; s++){
      smem[shmem_c_addr] = __hadd(smem[shmem_c_addr], smem[shmem_c_addr + s * 8 * LDN]);
    }

    __syncthreads();

    if (store_c_m >= M){return;}
    
    if (store_c_gmem_n < dim){
      *((half2 *)((&q[store_c_m * dim + 
        store_c_gmem_n]))) = 
          __hadd2(*(half2*)(&smem[store_c_addr]), *(half2*)(&biasq[store_c_gmem_n % dim]));
    }
    else if (store_c_gmem_n < 2 * dim){
      *((half2 *)(&k[store_c_m * (max_len * dim) + 
        len * dim + (store_c_gmem_n % dim)])) = 
          __hadd2(*(half2*)(&smem[store_c_addr]), *(half2*)(&biask[store_c_gmem_n % dim]));
    }
    else if (store_c_gmem_n < 3 * dim){
      *((half2 *)(&v[store_c_m * (max_len * dim) + 
        len * dim + (store_c_gmem_n % dim)])) = 
          __hadd2(*(half2*)(&smem[store_c_addr]), *(half2*)(&biasv[store_c_gmem_n % dim]));
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m8n32k256x8_bz1_residual_kernel(
    const half * __restrict__ a, const half * __restrict__ b, 
    half * __restrict__ r, half * __restrict__ c,
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

    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   
    int load_ab_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   

    int load_a_smem_addr_0 = __cvta_generic_to_shared(s_a) + OFFSET(load_a_smem_m, load_ab_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = __cvta_generic_to_shared(s_b) + OFFSET(load_b_smem_n, load_ab_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

    int load_a_gmem_addr = OFFSET(load_a_smem_m, (k_start + load_ab_smem_k), K);
    int load_b_gmem_addr = OFFSET((bx * BN + load_b_smem_n), (k_start + load_ab_smem_k), K);

    for (int bk = 0; bk < K / BK; bk++ ) {
        if (load_a_smem_m < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    // residual handling (% 256 !=0 but % 128 == 0)
    for (int bk = K / BK; bk < DIV_UP(K, BK); bk++){
        if ((load_a_smem_m < M) && (load_ab_smem_k < 128)){
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
        {     
            if (load_ab_smem_k < 128){
                asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                    : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
            }
        }
        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        wmma::load_matrix_sync(frag_a, &s_a[wid * 16], LDK);
        wmma::load_matrix_sync(frag_b, &s_b[wid * 16], LDK);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        __syncthreads();
    }

    wmma::store_matrix_sync(&smem[wid * 8 * LDN], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_n = tid & 31;   // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(load_a_smem_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(load_a_smem_m, bx * BN + shmem_c_n, N);

    if (load_a_smem_m < M) {
        half to_store = r[gmem_c_addr];
        #pragma unroll
        for (int s = 0; s < 8; s++){
            to_store = __hadd(to_store, smem[shmem_c_addr + s * 8 * LDN]);
        }
        c[gmem_c_addr] = to_store;
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void flat_gemm_m8n32k128x8_bz1_residual_kernel(
    half * __restrict__ a, half * __restrict__ b, 
    half * __restrict__ r, half * __restrict__ c,
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

    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   
    int load_a_smem_k = (tid & 31) << 2; // 0 ~ 124  | 0 4  8 ... 124
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    #pragma unroll 32
    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++ ) {
        if (load_a_gmem_m < M) {
            asm("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<4; i++)
            asm("cp.async.ca.shared.global [%0], [%1], 8;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        wmma::load_matrix_sync(frag_a, &s_a[wid * 16], LDK);
        wmma::load_matrix_sync(frag_b, &s_b[wid * 16], LDK);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);


        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
        __syncthreads();
    }

    wmma::store_matrix_sync(&smem[wid * 8 * LDN], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_n = tid & 31;   // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(load_a_smem_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(load_a_smem_m, bx * BN + shmem_c_n, N);

    if (load_a_smem_m < M) {
        half to_store = r[gmem_c_addr];
        #pragma unroll
        for (int s = 0; s < 8; s++){
            to_store = __hadd(to_store, smem[shmem_c_addr + s * 8 * LDN]);
        }
        c[gmem_c_addr] = to_store;
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n32k256x8_bz1_residual_kernel(
    const half * __restrict__ a, const half * __restrict__ b, 
    half * __restrict__ r, half * __restrict__ c,
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

    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5) << 1; // 0 ~ 14   | 0 2  4 ... 14  
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248 
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addrs[2];
    #pragma unroll
    for (int i=0; i<2; i++){
        load_a_smem_addrs[i] = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    }
    int load_b_smem_addrs[4];
    #pragma unroll
    for (int i=0; i<4; i++){
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
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
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<4; i++){
            wmma::load_matrix_sync(frag_a, &s_a[(wid & 3) * 16 + 64 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[(wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&smem[(wid & 3) * 16 * LDN + (wid >> 2) * 16], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 4;           // 0, 1, 2, 3, 4, ..., 15
    int shmem_c_n = (tid & 15) << 1;    // 0, 2, 4, 6, 8, ..., 30
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

    if (shmem_c_m < M) {
        half2 to_store = *(half2*)&r[gmem_c_addr];
        #pragma unroll
        for (int s = 0; s < 4; s++){
            to_store = __hadd2(to_store, 
                *(half2*)(&smem[shmem_c_addr + s * 16 * LDN]));
        }
        *(half2*)(&c[gmem_c_addr]) = to_store;
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n32k256x8_db_residual_kernel(
    const half * __restrict__ a, const half * __restrict__ b, 
    half * __restrict__ r, half * __restrict__ c,
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

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * LDK;
    int s_a_offset = BM * LDK;
    int s_b_offset = BN * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5) << 1; // 0 ~ 14   | 0 2  4 ... 14  
    int load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248 
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addrs[2];
    #pragma unroll
    for (int i=0; i<2; i++){
        load_a_smem_addrs[i] = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    }
    int load_b_smem_addrs[4];
    #pragma unroll
    for (int i=0; i<4; i++){
        load_b_smem_addrs[i] = s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    }

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    {
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
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    #pragma unroll
    for (int bk = 1; bk < (K / gridDim.z) / BK; bk++ ) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
        
        #pragma unroll
        for(int i=0; i<2; i++)
        {   
            if ((load_a_gmem_m + i) < M) {
                asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                    : "r"(load_a_smem_addrs[i] + smem_sel_next * s_a_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr + i * K]));
            }
        }

        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i] + smem_sel_next * s_b_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + i * K]));
        }

        for(int i=0; i<4; i++){
            wmma::load_matrix_sync(frag_a, &s_a[smem_sel * s_a_offset + (wid & 3) * 16 + 64 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[smem_sel * s_b_offset + (wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    int smem_sel = (((K / gridDim.z) / BK) & 1) ^ 1;

    for(int i=0; i<4; i++){
        wmma::load_matrix_sync(frag_a, &s_a[smem_sel * s_a_offset + (wid & 3) * 16 + 64 * i], LDK);
        wmma::load_matrix_sync(frag_b, &s_b[smem_sel * s_b_offset + (wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    __syncthreads();

    wmma::store_matrix_sync(&smem[(wid & 3) * 16 * LDN + (wid >> 2) * 16], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 4;           // 0, 1, 2, 3, 4, ..., 15
    int shmem_c_n = (tid & 15) << 1;    // 0, 2, 4, 6, 8, ..., 30
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);
  
    if (shmem_c_m < M) {
        half2 to_store = *(half2*)&r[gmem_c_addr];
        #pragma unroll
        for (int s = 0; s < 4; s++){
            to_store = __hadd2(to_store, 
                *(half2*)(&smem[shmem_c_addr + s * 16 * LDN]));
        }
        *(half2*)(&c[gmem_c_addr]) = to_store;
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m8n32k256x8_bz_1_bias_relu_kernel(
    const half * __restrict__ a, const half * __restrict__ b0, 
    half * __restrict__ c, half * __restrict__ bias0, 
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

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b[1];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c[1];

    wmma::fill_fragment(frag_c[0], __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   
    int load_ab_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    // int load_b_smem_k = load_a_smem_k;

    int load_a_smem_addr_0 = __cvta_generic_to_shared(s_a) + OFFSET(load_a_smem_m, load_ab_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs_0[4];
    #pragma unroll
    for(int i=0; i<4; i++){
        load_b_smem_addrs_0[i] = __cvta_generic_to_shared(s_b_0) + (OFFSET(load_b_smem_n, load_ab_smem_k, LDK) + i * (LDK)) * sizeof(half);
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

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b[0], &s_b_0[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c[0], frag_a, frag_b[0], frag_c[0]);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&s_b_0[wid * 8 * LDN], frag_c[0], LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_n = tid & 31;   // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(load_a_smem_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(load_a_smem_m, bx * BN + shmem_c_n, N);

    float res[1] = {0.0f};

#pragma unroll
    for (int s = 0; s < 8; s++){
        res[0] += __half2float(s_b_0[shmem_c_addr + s * 8 * LDN]);
    }

    if (load_a_smem_m < M) {
        res[0] = res[0] + __half2float(bias0[gmem_c_addr % N]);
        // Relu
        if (res[0] < 0.0f) res[0] = 0.0f;
        c[gmem_c_addr] = __float2half(res[0]);
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m8k32n256x8_bz1_bias_residual_kernel(
    const half * __restrict__ a, const half * __restrict__ b, 
    half * __restrict__ r, half * __restrict__ c, half * __restrict__ bias,
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

    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   
    int load_ab_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248
    int load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   
    // int load_b_smem_k = load_a_smem_k;

    int load_a_smem_addr_0 = __cvta_generic_to_shared(s_a) + OFFSET(load_a_smem_m, load_ab_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
    #pragma unroll
    for(int i=0; i<4; i++)
        load_b_smem_addrs[i] = __cvta_generic_to_shared(s_b) + OFFSET(load_b_smem_n, load_ab_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

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
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<2; i++){
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&smem[wid * 8 * LDN], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_n = tid & 31;   // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(load_a_smem_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(load_a_smem_m, bx * BN + shmem_c_n, N);

    
#pragma unroll
    for (int s = 1; s < 8; s++){
        smem[shmem_c_addr] = __hadd(smem[shmem_c_addr], smem[shmem_c_addr + s * 8 * LDN]);
    }

    __syncthreads();

    if (load_a_smem_m < M) {
        c[gmem_c_addr] = __hadd(__hadd(smem[shmem_c_addr], r[gmem_c_addr]), bias[gmem_c_addr % N]);
    }
}