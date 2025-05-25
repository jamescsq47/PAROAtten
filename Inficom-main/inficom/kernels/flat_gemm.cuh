#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <mma.h>

#include "utils.h"

using namespace nvcuda;

template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m8n32k256x8_bz1_kernel(
    const half * __restrict__ a, const half * __restrict__ b, half * __restrict__ c,
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

    if (shmem_c_m < M) {
        c[gmem_c_addr] = smem[shmem_c_addr];
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n32k256x8_bz1_kernel(
    const half * __restrict__ a, const half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_start = (DIV_UP(K, 256) * 256) / gridDim.z * bz;

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

    // see if we need to handle the left K=128
    // if (load_a_gmem_k < 128){
    // #pragma unroll
    // for(int i=0; i<2; i++)
    // {   
    //     if ((load_a_gmem_m + i) < M) {
    //         asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
    //                 : "r"(load_a_smem_addrs[i]), "l"(&a[load_a_gmem_addr + i * K]));
    //     }
    // }

    // #pragma unroll
    // for(int i=0; i<4; i++)
    // {   
    //     asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
    //         : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
    // }
    // }
    // asm("cp.async.commit_group;\n" ::);
    // asm("cp.async.wait_group 0;\n" ::);
    // __syncthreads();

    // for(int i=0; i<2; i++){
    //     wmma::load_matrix_sync(frag_a, &s_a[(wid & 3) * 16 + 64 * i], LDK);
    //     wmma::load_matrix_sync(frag_b, &s_b[(wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
    //     wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    // }

    // __syncthreads();

    wmma::store_matrix_sync(&smem[(wid & 3) * 16 * LDN + (wid >> 2) * 16], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 4;           // 0, 1, 2, 3, 4, ..., 15
    int shmem_c_n = (tid & 15) << 1;    // 0, 2, 4, 6, 8, ..., 30
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

    
#pragma unroll
    for (int s = 1; s < 4; s++){
        *(half2*)(&smem[shmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), 
            *(half2*)(&smem[shmem_c_addr + s * 16 * LDN]));
    }

    if (shmem_c_m < M) {
        *(half2*)(&c[gmem_c_addr]) = *(half2*)(&smem[shmem_c_addr]);
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n64k128x8_bz1_kernel(
    const half * __restrict__ a, const half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
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

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    for (int bk = 0; bk < K / BK; bk++ ) {
        
        if ((load_a_gmem_m) < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
                    : "r"(load_a_smem_addr), "l"(&a[load_a_gmem_addr]));
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
            wmma::load_matrix_sync(frag_a, &s_a[(wid >> 2) * 16 + 32 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[(wid & 3) * 16 * LDK + (wid >> 2) * 16 + 32 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&smem[(wid >> 2) * 16 * LDN + (wid & 3) * 16], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 4;           // 0, 1, 2, 3, 4, ..., 15
    int shmem_c_n = (tid & 15) << 2;    // 0, 4, 8,12,16, ..., 60
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

    *(half2*)(&smem[shmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), 
            *(half2*)(&smem[shmem_c_addr + 16 * LDN]));
    *(half2*)(&smem[(shmem_c_addr + 2)]) = __hadd2(*(half2*)(&smem[(shmem_c_addr + 2)]), 
            *(half2*)(&smem[(shmem_c_addr + 2) + 16 * LDN]));

    if (shmem_c_m < M) {
        *(half2*)(&c[gmem_c_addr]) = *(half2*)(&smem[shmem_c_addr]);
        *(half2*)(&c[gmem_c_addr + 2]) = *(half2*)(&smem[shmem_c_addr + 2]);
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n32k256x8_db_kernel(
    const half * __restrict__ a, const half * __restrict__ b, half * __restrict__ c,
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

    // see if we need to handle the left K=128
    // load_a_gmem_addr += BK;
    // load_b_gmem_addr += BK;
    // if (load_a_gmem_k < 128){
    // #pragma unroll
    //     for(int i=0; i<2; i++)
    //     {   
    //         if ((load_a_gmem_m + i) < M) {
    //             asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
    //                 : "r"(load_a_smem_addrs[i] + s_a_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr + i * K]));
    //         }
    //     }

    //     #pragma unroll
    //     for(int i=0; i<4; i++)
    //     {   
    //         asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
    //             : "r"(load_b_smem_addrs[i] + s_b_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + i * K]));
    //     }
    // }
    // asm("cp.async.commit_group;\n" ::);
    // asm("cp.async.wait_group 0;\n" ::);
    // __syncthreads();
    // for(int i=0; i<2; i++){
    //     wmma::load_matrix_sync(frag_a, &s_a[s_a_offset + (wid & 3) * 16 + 64 * i], LDK);
    //     wmma::load_matrix_sync(frag_b, &s_b[s_b_offset + (wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
    //     wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    // }
    // __syncthreads();

    wmma::store_matrix_sync(&smem[(wid & 3) * 16 * LDN + (wid >> 2) * 16], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 4;           // 0, 1, 2, 3, 4, ..., 15
    int shmem_c_n = (tid & 15) << 1;    // 0, 2, 4, 6, 8, ..., 30
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

    if (shmem_c_m >= M) {return;}
    #pragma unroll
    for (int s = 1; s < 4; s++){
        *(half2*)(&smem[shmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), 
            *(half2*)(&smem[shmem_c_addr + s * 16 * LDN]));
    }
    *(half2*)(&c[gmem_c_addr]) = *(half2*)(&smem[shmem_c_addr]);
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n64k128x8_db_kernel(
    const half * __restrict__ a, const half * __restrict__ b, half * __restrict__ c,
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
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

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
                : "r"(load_b_smem_addrs[i] + + smem_sel_next * s_b_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + i * K]));
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
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

// #pragma unroll
//     for (int s = 1; s < 4; s++){
//     }

    if (shmem_c_m < M) {
        *(half2*)(&smem[shmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), 
            *(half2*)(&smem[shmem_c_addr + 16 * LDN]));
        *(half2*)(&smem[(shmem_c_addr + 2)]) = __hadd2(*(half2*)(&smem[(shmem_c_addr + 2)]), 
            *(half2*)(&smem[(shmem_c_addr + 2) + 16 * LDN]));
        // *(float2*)(&c[gmem_c_addr]) = *(float2*)(&smem[shmem_c_addr]);
        atomicAdd((half2*)(&c[gmem_c_addr]), *(half2*)(&smem[shmem_c_addr]));
        atomicAdd((half2*)(&c[gmem_c_addr + 2]), *(half2*)(&smem[shmem_c_addr + 2]));
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n64k128x8_db_bz_kernel(
    const half * __restrict__ a, const half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_slot = DIV_UP(DIV_UP(K, gridDim.z), BK) * BK;
    int k_start = k_slot * bz;
    int k_end = min(K, k_slot * (bz + 1));

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
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

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
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    #pragma unroll
    // for (int bk = 1; bk < (K / gridDim.z) / BK; bk++ ) {
    for (int bk = 1; bk < (k_end - k_start) / BK; bk++){

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
                : "r"(load_b_smem_addrs[i] + + smem_sel_next * s_b_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + i * K]));
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

    int smem_sel = (((k_end - k_start) / BK) & 1) ^ 1;

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
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

// #pragma unroll
//     for (int s = 1; s < 4; s++){
//     }

    if (shmem_c_m < M) {
        *(half2*)(&smem[shmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), 
            *(half2*)(&smem[shmem_c_addr + 16 * LDN]));
        *(half2*)(&smem[(shmem_c_addr + 2)]) = __hadd2(*(half2*)(&smem[(shmem_c_addr + 2)]), 
            *(half2*)(&smem[(shmem_c_addr + 2) + 16 * LDN]));
        // *(float2*)(&c[gmem_c_addr]) = *(float2*)(&smem[shmem_c_addr]);
        atomicAdd((half2*)(&c[gmem_c_addr]), *(half2*)(&smem[shmem_c_addr]));
        atomicAdd((half2*)(&c[gmem_c_addr + 2]), *(half2*)(&smem[shmem_c_addr + 2]));
    }
}


template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n32k256x8_db_bz_kernel(
    const half * __restrict__ a, const half * __restrict__ b, half * __restrict__ c,
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

    if (shmem_c_m >= M) {return;}
    #pragma unroll
    for (int s = 1; s < 4; s++){
        *(half2*)(&smem[shmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), 
            *(half2*)(&smem[shmem_c_addr + s * 16 * LDN]));
    }
    // *(half2*)(&c[gmem_c_addr]) = *(half2*)(&smem[shmem_c_addr]);
    atomicAdd((half2*)(&c[gmem_c_addr]), *(half2*)(&smem[shmem_c_addr]));
}


/***
 * B转置 A:M*K B:N*K
 * bs1-8的最快版本，A100上略超过cublas平均9%，RTX3090上表现更好
 * BM=8 BN=32 BK=256
 * LDK = BK + PAD = 264
 * LDN = BN + PAD = 40
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void flat_gemm_m8n32k256x8_bt_reduce(
    const half * __restrict__ a, const half * __restrict__ b, const half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;         // WARP id
    int laneid = tid & 31;

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

    /**
     * 通过位运算获取每个thread对应的索引位置
     * load_a 每个warp访问1*64个元素，通过cp.async指定访问8B即4个half完成
     * load_b 每个warp访问4*256个元素，通过cp.async指定访问16B即8个half完成
    */
    short load_a_smem_m = (tid >> 5);      // 0 ~ 7    | 0 1  2 ...  7   每个索引32个一组 共8组
    short load_a_smem_k = (tid & 31) << 3; // 0 ~ 248  | 0 8 16 ... 248(32个数)  循环8组  间隔是8个half 16B
    short load_b_smem_n = (tid >> 5) << 2; // 0 ~ 28   | 0 4  8 ... 28   每个索引32个一组 共8组
    short load_b_smem_k = load_a_smem_k;

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
    int load_fragment_offet;

    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++ ) {
        if (load_a_gmem_m < M) {
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
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

        #pragma unroll
        for(int i=0; i<2; i++){
            load_fragment_offet = wid * 16 + 128 * i;
            // if((load_fragment_offet + bk * BK) >= (K / gridDim.z)){
            //     break;
            // }
            wmma::load_matrix_sync(frag_a, &s_a[load_fragment_offet], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[load_fragment_offet], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }
        __syncthreads();

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    int st_ld = 8 * 32 + 8;
    wmma::store_matrix_sync(&s_a[wid * 32], frag_c, st_ld, wmma::mem_row_major);
    __syncthreads();
    // 写回SMEM后进行block内归约，block间使用原子加
    short reduce_c_m = laneid >> 2, reduce_c_n = (tid & 3) << 3;
    if (reduce_c_m < M) {
        int offset = OFFSET(reduce_c_m, reduce_c_n, st_ld) + wid * 32;
        #pragma unroll
        for(int s = 8 / 2; s > 0; s >>= 1){         // 8个warp进行归约
            if(wid < s){
                #pragma unroll
                for(int i=0; i<4; i++){
                    *(half2 *)(&s_a[offset + 2 * i]) = __hadd2(*(half2 *)(&s_a[offset + 2 * i]), *(half2 *)(&s_a[offset + 2 * i + s * 32]));
                    __syncthreads();
                }
            }
        }
        if(wid==0)
        {
            int store_c_smem_addr = OFFSET(reduce_c_m, reduce_c_n, st_ld);
            int store_c_gmem_addr = OFFSET(reduce_c_m, reduce_c_n + bx * BN, N);
            if(gridDim.z==1){
                *(float4*)(&c[store_c_gmem_addr]) = *(float4*)(&s_a[store_c_smem_addr]);
            }else{
                #pragma unroll
                for(int i=0; i<4; i++){
                    atomicAdd(((half2 *)(&c[store_c_gmem_addr + 2 * i])),
                            *((half2 *)(&s_a[store_c_smem_addr + 2 * i])));
                }
            }
        }
    }
}

/***
 * B转置 A:M*K B:N*K
 * 适用于bs在9~16的情况
 * 在Llama2-7B和OPT-7B的大部分caseA100和RTX3090表现均超过cublas，在GLM-6B上表现较差
 * BM=16 BN=32 BK=128
 * LDK = BK + PAD = 136
 * LDN = BN + PAD = 40
*/
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void flat_gemm_m16n32k128x8_bt_reduce(
    const half * __restrict__ a, const half * __restrict__ b, const half * __restrict__ c,
    const int M, const int N, const int K) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    short bz = blockIdx.z;
    int k_start = K / gridDim.z * bz;

    short tid = threadIdx.x;
    short wid = tid >> 5;         // WARP id
    short laneid = tid & 31;

    // bx for N, if bx is out of range, return
    // if (bx >= N / BN)
    //     return;

    __shared__ half s_a[BM * (LDK)];
    __shared__ half s_b[BN * (LDK)];

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[2];

    wmma::fill_fragment(frag_c[0], __float2half(0.0f));
    wmma::fill_fragment(frag_c[1], __float2half(0.0f));

    int load_a_smem_m = (tid >> 4);     
    int load_a_smem_k = (tid & 15) << 3; 
    int load_b_smem_n = load_a_smem_m << 1;
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[2];
    #pragma unroll
    for(int i=0; i<2; i++)
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
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_a_smem_addr_0),
                    "l"(&a[load_a_gmem_addr]));
        }
        #pragma unroll
        for(int i=0; i<2; i++)
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        wmma::load_matrix_sync(frag_a, &s_a[wid * 16], LDK);
        wmma::load_matrix_sync(frag_b[0], &s_b[wid * 16], LDK);
        wmma::load_matrix_sync(frag_b[1], &s_b[wid * 16 + 16 * LDK], LDK);
        wmma::mma_sync(frag_c[0], frag_a, frag_b[0], frag_c[0]);
        wmma::mma_sync(frag_c[1], frag_a, frag_b[1], frag_c[1]);
        __syncthreads();

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }
    int st_ld = 8 * 16 + 8;
    wmma::store_matrix_sync(&s_a[wid * 16             ], frag_c[0], st_ld, wmma::mem_row_major);
    wmma::store_matrix_sync(&s_a[wid * 16 + 16 * st_ld], frag_c[1], st_ld, wmma::mem_row_major);

    __syncthreads();
    // 写回SMEM后进行block内归约，block间使用原子加
    short reduce_c_m = laneid >> 1, reduce_c_n = (tid & 1) << 3;
    if (reduce_c_m < M) {
        int offset[2] = {OFFSET(reduce_c_m, reduce_c_n, st_ld) + wid * 16, OFFSET(reduce_c_m, reduce_c_n, st_ld) + wid * 16 + 16 * st_ld};
        #pragma unroll
        for(int s = 8 / 2; s > 0; s >>= 1){         // 8个warp进行归约
            if(wid < s){
                #pragma unroll
                for(int i=0; i<4; i++){
                    *(half2 *)(&s_a[offset[0] + 2 * i]) = __hadd2(*(half2 *)(&s_a[offset[0] + 2 * i]), *(half2 *)(&s_a[offset[0] + 2 * i + s * 16]));
                    *(half2 *)(&s_a[offset[1] + 2 * i]) = __hadd2(*(half2 *)(&s_a[offset[1] + 2 * i]), *(half2 *)(&s_a[offset[1] + 2 * i + s * 16]));
                    __syncthreads();
                }
            }
        }
        if(wid==0)
        {
            int store_c_smem_addr[2] = {OFFSET(reduce_c_m, reduce_c_n, st_ld), OFFSET(reduce_c_m + 16, reduce_c_n, st_ld)};
            int store_c_gmem_addr[2] = {OFFSET(reduce_c_m, reduce_c_n + bx * BN, N), OFFSET(reduce_c_m, reduce_c_n + 16 + bx * BN, N)};
            if(gridDim.z==1){
                *(float4*)(&c[store_c_gmem_addr[0]]) = *(float4*)(&s_a[store_c_smem_addr[0]]);
                *(float4*)(&c[store_c_gmem_addr[1]]) = *(float4*)(&s_a[store_c_smem_addr[1]]);
            }else{
                #pragma unroll
                for(int i=0; i<4; i++){
                    atomicAdd(((half2 *)(&c[store_c_gmem_addr[0] + 2 * i])),
                            *((half2 *)(&s_a[store_c_smem_addr[0] + 2 * i])));
                    atomicAdd(((half2 *)(&c[store_c_gmem_addr[1] + 2 * i])),
                            *((half2 *)(&s_a[store_c_smem_addr[1] + 2 * i])));
                }
            }
        }
    }

}

// A special GEMM kernel with quantized W/A for FPGA'25 Rebuttal
template <int BM, int BN, int BK, int LDK, int LDN>
__global__ __forceinline__ void flat_gemm_m16n64k128x8_bz1_w4a12_kernel(
    const uint8_t * __restrict__ activation, 
    const half * act_scales, 
    const uint8_t * __restrict__ weight, 
    const uint8_t * __restrict__ wet_zeros,
    const uint8_t * __restrict__ wet_scales1, 
    const half * wet_scales2, 
    half * output,
    const int M, const int N, const int K, const int GS) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int bx = blockIdx.x;
    // int bz = blockIdx.z;
    int by = blockIdx.y;
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
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 4);      // 0 ~ 15   | 0 1  2 ... 15  
    int load_a_smem_k = (tid & 15) << 3; // 0 ~ 120  | 0 8 16 ... 120
    int load_b_smem_n = (tid >> 4) << 2; // 0 ~ 60   | 0 4  8 ... 60   
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    // size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    // size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    half *load_a_smem_addr = &s_a[0] + OFFSET(load_a_smem_m, load_a_smem_k, LDK);
    // #pragma unroll
    // for (int i=0; i<2; i++){
    //     load_a_smem_addrs[i] = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
    // }
    half *load_b_smem_addrs[4];
    #pragma unroll
    for (int i=0; i<4; i++){
        load_b_smem_addrs[i] = &s_b[0] + OFFSET(load_b_smem_n, load_b_smem_k, LDK) + i * (LDK);
    }

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = load_a_smem_k;
    int load_b_gmem_k = load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    uint8_t qa[12];
    uint8_t qw[4];

    for (int bk = 0; bk < K / BK; bk++ ) {
        
        // INT12 Activation x 16 -> FP16 Activation x 16
        if ((load_a_gmem_m) < M) {
            // asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :
            //         : "r"(load_a_smem_addr), "l"(&a[load_a_gmem_addr]));
            *(float3*)(&qa[0]) = *(float3*)(&activation[load_a_gmem_addr / 8 * 12]);
        }
        // Activation: INT12 -> FP16 (Scales: FP16)
        *(float4*)(load_a_smem_addr) = _12bits_dequant(&qa[0]);

        // INT4 Weight x 16 -> FP16 Weight x 16
        #pragma unroll
        for(int i=0; i<4; i++)
        {   
            // asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            //     : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
            *(float*)(&qw[0]) = *(float*)(&weight[(load_b_gmem_addr + i * K)/2]);

            // Weight: INT4 -> FP16 (Zeros: INT4, Scales: INT8)
            *(float4*)(load_b_smem_addrs[i]) = _4bits_dequant(&qw[0]);
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int i=0; i<4; i++){
            wmma::load_matrix_sync(frag_a, &s_a[(wid >> 2) * 16 + 32 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[(wid & 3) * 16 * LDK + (wid >> 2) * 16 + 32 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&smem[(wid >> 2) * 16 * LDN + (wid & 3) * 16], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 4;           // 0, 1, 2, 3, 4, ..., 15
    int shmem_c_n = (tid & 15) << 2;    // 0, 4, 8,12,16, ..., 60
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(by * BM + shmem_c_m, bx * BN + shmem_c_n, N);

    
// #pragma unroll
//     for (int s = 1; s < 4; s++){
    *(half2*)(&smem[shmem_c_addr]) = __hadd2(*(half2*)(&smem[shmem_c_addr]), 
            *(half2*)(&smem[shmem_c_addr + 16 * LDN]));
    *(half2*)(&smem[(shmem_c_addr + 2)]) = __hadd2(*(half2*)(&smem[(shmem_c_addr + 2)]), 
            *(half2*)(&smem[(shmem_c_addr + 2) + 16 * LDN]));
//     }

    if (shmem_c_m < M) {
        *(half2*)(&output[gmem_c_addr]) = *(half2*)(&smem[shmem_c_addr]);
        *(half2*)(&output[gmem_c_addr + 2]) = *(half2*)(&smem[shmem_c_addr + 2]);
    }
}
