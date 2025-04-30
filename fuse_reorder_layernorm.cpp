#include <torch/extension.h>
#include <vector>

// pattern_id: 0=FHW,1=FWH,2=WFH,3=WHF,4=HFW,5=HWF
// mode: 0=reorder, 1=inv_reorder
std::vector<at::Tensor> fused_reorder_layernorm_cuda_forward(
    const at::Tensor& x,       // [B, T, C]
    const at::Tensor& shift,   // [B, C]
    const at::Tensor& scale,   // [B, C]
    int64_t F, int64_t H, int64_t W,
    double eps,
    int pattern_id,
    int mode
);

std::vector<at::Tensor> fused_reorder_layernorm_forward(
    const at::Tensor& x,
    const at::Tensor& shift,
    const at::Tensor& scale,
    int64_t F, int64_t H, int64_t W,
    double eps,
    const std::string& pattern,
    bool inv_reorder
) {
    static const std::vector<std::string> patterns = {
        "FHW","FWH","WFH","WHF","HFW","HWF"
    };
    int pid = -1;
    for (int i = 0; i < 6; ++i) {
        if (patterns[i] == pattern) { pid = i; break; }
    }
    TORCH_CHECK(pid >= 0, "Unknown pattern ", pattern);
    int m = inv_reorder ? 1 : 0;
    return fused_reorder_layernorm_cuda_forward(
        x, shift, scale, F, H, W, eps, pid, m
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_reorder_layernorm_forward,
          "Fused reorder+inv_reorder + conditioned LayerNorm (CUDA)");
}
