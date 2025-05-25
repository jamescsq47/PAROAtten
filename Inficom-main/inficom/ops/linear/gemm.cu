#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <driver_functions.h>
#include <torch/extension.h>

#include "../../kernels/flat_gemm.cuh"

void flat_gemm_m8n32k256x8_bz1(at::Tensor A, at::Tensor B, at::Tensor O) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);  // weight shape: N * K
    
    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(transposed output row) must be multiple of 128!");
    }

    if (M > 8) {
        throw std::invalid_argument("M(input row) must not exceed multiple of 128!");
    }

    // at::Tensor O = torch::empty({M, N}, 
    //     at::device(A.device()).dtype(at::ScalarType::Half));

    flat_gemm_m8n32k256x8_bz1_kernel<8, 32, 256, 264, 40><<<dim3(N / 32, 1, 1), dim3(256)>>>(
            reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(B.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
            M, N, K
    );

    // return O;
}

void flat_gemm_m16n32k256x8_bz1(at::Tensor A, at::Tensor B, at::Tensor O) {
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);  // weight shape: N * K
    
    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(transposed output row) must be multiple of 128!");
    }

    if (M > 16) {
        throw std::invalid_argument("M(input row) must not exceed 16!");
    }

    // at::Tensor O = torch::empty({M, N}, 
    //     at::device(A.device()).dtype(at::ScalarType::Half));
    
    cudaFuncSetAttribute(flat_gemm_m16n32k256x8_db_kernel<16, 32, 256, 264, 40>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 50688);
    
    unsigned int dsmem = 2 * (16 + 32) * (256 + 8) * sizeof(half);

    flat_gemm_m16n32k256x8_db_kernel<16, 32, 256, 264, 40><<<dim3(N / 32, 1, 1), dim3(256), dsmem>>>(
            reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(B.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
            M, N, K
    );

    // cudaFuncSetAttribute(flat_gemm_m16n32k256x8_db_kernel<16, 32, 256, 264, 40>,
    //             cudaFuncAttributeMaxDynamicSharedMemorySize, 50688);
    
    // unsigned int dsmem = 2 * (16 + 32) * (256 + 8) * sizeof(half);

    // flat_gemm_m16n32k256x8_db_kernel<16, 32, 256, 264, 40>
    //     <<<dim3(N / 32, 1, 1), dim3(256), dsmem>>>(
    //         reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
    //         reinterpret_cast<half *>(B.data_ptr<at::Half>()),  
    //         reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
    //         M, N, K
    // );

    // flat_gemm_m16n32k256x8_bz1_kernel<16, 32, 256, 264, 40><<<dim3(N / 32, 1, 1), dim3(256)>>>(
    //         reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
    //         reinterpret_cast<half *>(B.data_ptr<at::Half>()),  
    //         reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
    //         M, N, K
    // );

    // return O;
}


void flat_gemm_m16n64k128x8_bz1(at::Tensor A, at::Tensor B, at::Tensor O) {
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);  // weight shape: N * K
    
    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(transposed output row) must be multiple of 128!");
    }

    if (M > 16) {
        throw std::invalid_argument("M(input row) must not exceed 16!");
    }

    // at::Tensor O = torch::empty({M, N}, 
    //     at::device(A.device()).dtype(at::ScalarType::Half));
    
    // cudaFuncSetAttribute(flat_gemm_m16n32k256x8_db_kernel<16, 32, 256, 264, 40>,
    //             cudaFuncAttributeMaxDynamicSharedMemorySize, 50688);
    
    // unsigned int dsmem = 2 * (16 + 32) * (256 + 8) * sizeof(half);

    // flat_gemm_m16n32k256x8_db_kernel<16, 32, 256, 264, 40><<<dim3(N / 32, 1, 1), dim3(256), dsmem>>>(
    //         reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
    //         reinterpret_cast<half *>(B.data_ptr<at::Half>()),  
    //         reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
    //         M, N, K
    // );

    // cudaFuncSetAttribute(flat_gemm_m16n32k256x8_db_bz_kernel<16, 32, 256, 264, 40>,
    //             cudaFuncAttributeMaxDynamicSharedMemorySize, 50688);
    
    // unsigned int dsmem = 2 * (16 + 32) * (256 + 8) * sizeof(half);

    flat_gemm_m16n64k128x8_db_bz_kernel<16, 64, 128, 136, 72>
        <<<dim3(N / 64, 1, 4), dim3(256)>>>(
            reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(B.data_ptr<at::Half>()),  
            reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
            M, N, K
    );

    // flat_gemm_m16n64k128x8_bz1_kernel<16, 64, 128, 136, 72>
    //        <<<dim3(N / 64, 1, 1), dim3(256)>>>(
    //         reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
    //         reinterpret_cast<half *>(B.data_ptr<at::Half>()),  
    //         reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
    //         M, N, K
    // );

    // return O;
}


void flat_gemm_mix_for_decode(at::Tensor A, at::Tensor B, at::Tensor O){
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);  // weight shape: N * K
    
    if (N % 128 || K % 128) {
        throw std::invalid_argument("K(input column) & N(transposed output row) must be multiple of 128!");
    }

    if (M > 16) {
        throw std::invalid_argument("M(input row) must not exceed 16!");
    }

    // at::Tensor O = torch::empty({M, N}, 
    //     at::device(A.device()).dtype(at::ScalarType::Half));


    if(M <= 8){
        const int BM = 8, BN = 32, BK = 256;
        dim3 blockDim(256);
        dim3 gridDim;
        if ((K / 256) % 4 == 0)
            gridDim = dim3(N / BN, 1, 4);
        else if ((K / 256) % 2 == 0)
            gridDim = dim3(N / BN, 1, 2);
        else
            gridDim = dim3(N / BN, 1, 1);
        flat_gemm_m8n32k256x8_bt_reduce<BM, BN, BK, BK+8, BN+8><<<gridDim, blockDim>>>(
                reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
                reinterpret_cast<half *>(B.data_ptr<at::Half>()),  
                reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
                M, N, K
        );
    }else if(M <= 16){
        const int BM = 16, BN = 32, BK = 128;
        dim3 blockDim(256);
        dim3 gridDim;
        if ((K / 256) % 8 == 0)
            gridDim = dim3(N / BN, 1, 8);
        else if ((K / 256) % 4 == 0)
            gridDim = dim3(N / BN, 1, 4);
        else if ((K / 256) % 2 == 0)
            gridDim = dim3(N / BN, 1, 2);
        else
            gridDim = dim3(N / BN, 1, 1);
        flat_gemm_m16n32k128x8_bt_reduce<BM, BN, BK, BK+8, BN+8><<<gridDim, blockDim>>>(
                reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
                reinterpret_cast<half *>(B.data_ptr<at::Half>()),  
                reinterpret_cast<half *>(O.data_ptr<at::Half>()), 
                M, N, K
        );
    }
    // return O;
}

void flat_gemm_m16n64k128x8_bz1_for_fpga(
    at::Tensor activation, // INT12, [24, 8192]; INT8, [24, 12288]
    at::Tensor act_scales, // FP16, [1]
    at::Tensor weight, // Transposed. INT4, [8192, 8192]; INT8, [8192, 4096]
    at::Tensor wet_zeros, // INT4, [631]; INT8, [316]
    at::Tensor wet_scales1, // INT8, [631];
    at::Tensor wet_scales2, // FP16, [8192] per output channel
    at::Tensor output // FP16, [24, 8192]
){
    int M = activation.size(0); // 24
    int N = weight.size(0); // 8192
    int Quant_K = weight.size(1); // 4096
    int K = Quant_K * 2; // 8192

    int GS = 13; // group size

    const int BM = 16, BN = 64, BK = 128;
    dim3 blockDim(256);
    dim3 gridDim = dim3(N / BN, 2, 1); // 8192 / 64 = 128 for N-dim, 2 for M-dim, 1

    flat_gemm_m16n64k128x8_bz1_w4a12_kernel<BM, BN, BK, BK+8, BN+8><<<gridDim, blockDim>>>(
        activation.data_ptr<uint8_t>(), 
        reinterpret_cast<half *>(act_scales.data_ptr<at::Half>()), 
        weight.data_ptr<uint8_t>(), 
        wet_zeros.data_ptr<uint8_t>(), 
        wet_scales1.data_ptr<uint8_t>(),
        reinterpret_cast<half *>(wet_scales2.data_ptr<at::Half>()), 
        reinterpret_cast<half *>(output.data_ptr<at::Half>()),
        M, N, K, GS
        );
}

