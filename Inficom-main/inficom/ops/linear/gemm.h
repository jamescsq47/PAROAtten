#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// at::Tensor flat_gemm_m8n32k256x8_bz1(at::Tensor A, at::Tensor B);
// at::Tensor flat_gemm_m16n32k256x8_bz1(at::Tensor A, at::Tensor B);
// at::Tensor flat_gemm_m16n64k128x8_bz1(at::Tensor A, at::Tensor B);
// at::Tensor flat_gemm_mix_for_decode(at::Tensor A, at::Tensor B);

void flat_gemm_m8n32k256x8_bz1(at::Tensor A, at::Tensor B, at::Tensor O);
void flat_gemm_m16n32k256x8_bz1(at::Tensor A, at::Tensor B, at::Tensor O);
void flat_gemm_m16n64k128x8_bz1(at::Tensor A, at::Tensor B, at::Tensor O);
void flat_gemm_mix_for_decode(at::Tensor A, at::Tensor B, at::Tensor O);

void flat_gemm_m16n64k128x8_bz1_for_fpga(
    at::Tensor activation, // INT12, [24, 8192]; INT8, [24, 12288]
    at::Tensor act_scales, // FP16, [1]
    at::Tensor weight, // Transposed. INT4, [8192, 8192]; INT8, [8192, 4096]
    at::Tensor wet_zeros, // INT4, [631]; INT8, [316]
    at::Tensor wet_scales1, // INT8, [631];
    at::Tensor wet_scales2, // FP16, [8192] per output channel
    at::Tensor output // FP16, [8192, 8192]
);