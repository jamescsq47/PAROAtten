#include <torch/extension.h>
#include <cuda_fp16.h>
#include "rope.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("rope_permutation", &rope_permutation, "RoPE op with permutation.");
}
