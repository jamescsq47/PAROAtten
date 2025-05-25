#include <torch/extension.h>
#include <vector>
std::vector<at::Tensor> fused_apply_rotary_cuda(
    const at::Tensor& x,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t F, int64_t H, int64_t W,
    const std::string& pattern,
    bool inv
);

at::Tensor fused_apply_rotary(
    const at::Tensor& x,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t F, int64_t H, int64_t W,
    const std::string& pattern,
    bool inv_reorder
) {
  auto outs = fused_apply_rotary_cuda(
    x, cos, sin, F,H,W, pattern, inv_reorder
  );
  return outs[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_apply_rotary,
        "Fused rotary + (inv_)reorder (CUDA)");
}
