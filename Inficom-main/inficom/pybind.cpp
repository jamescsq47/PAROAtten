#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "ops/attn/decode_attn.h"
#include "ops/norm/norm.h"
#include "ops/element/residual.h"
#include "ops/element/rope.h"
#include "ops/linear/gemv.h"
#include "ops/linear/gemm.h"
#include "ops/linear/gemm_rope.h"
#include "ops/linear/gemm_swiglu.h"
#include "layers/layer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  // ops: decode attention
  m.def("decode_mha_with_async_softmax", &decode_mha_with_async_softmax);
  m.def("decode_mha_fall_back", &decode_mha_fall_back);
  m.def("decode_mqa_with_async_softmax", &decode_mha_with_async_softmax);
  m.def("decode_mqa_fall_back", &decode_mha_fall_back);
  // ops: norm
  m.def("rmsnorm", &rmsnorm);
  m.def("layernorm", &layernorm);
  m.def("reorder_layernorm", &reorder_layernorm);
  m.def("inv_reorder_layernorm", &inv_reorder_layernorm);
  // ops: add residual
  m.def("add_residual", &add_residual);
  m.def("rope_permutation", &rope_permutation);
  // ops: fused add residual and norm
  m.def("residual_rmsnorm", &residual_rmsnorm);
  m.def("residual_layernorm", &residual_layernorm);
  m.def("residual_rmsnorm_test", &residual_rmsnorm_test);
  // ops: linear - gemv
  m.def("gemv_acc_fp16", &gemv_acc_fp16);
  m.def("gemv_acc_fp32", &gemv_acc_fp32);
  // ops: linear - flat gemm
  m.def("flat_gemm_m8n32k256x8_bz1", &flat_gemm_m8n32k256x8_bz1);
  m.def("flat_gemm_m16n32k256x8_bz1", &flat_gemm_m16n32k256x8_bz1);
  m.def("flat_gemm_m16n64k128x8_bz1", &flat_gemm_m16n64k128x8_bz1);
  m.def("flat_gemm_mix_for_decode", &flat_gemm_mix_for_decode);
  // ops: fused
  m.def("attn_proj_rope_kv_cat_fwd", &attn_proj_rope_kv_cat_fwd);
  m.def("dual_linear_silu_dot_fwd", &dual_linear_silu_dot_fwd);
  // layers: attn
  m.def("llama2_attn_layer_fwd", &llama2_attn_layer_fwd);
  m.def("chatglm2_attn_layer_fwd", &chatglm2_attn_layer_fwd);
  m.def("opt_attn_layer_fwd", &opt_attn_layer_fwd);
  // layers: ffn
  m.def("llama2_ffn_layer_fwd", &llama2_ffn_layer_fwd);
  m.def("chatglm2_ffn_layer_fwd", &chatglm2_ffn_layer_fwd);
  m.def("opt_ffn_layer_fwd", &opt_ffn_layer_fwd);
  // ops: quantized GEMM for FPGA'25 Rebuttal
  m.def("flat_gemm_m16n64k128x8_bz1_for_fpga", &flat_gemm_m16n64k128x8_bz1_for_fpga);
}