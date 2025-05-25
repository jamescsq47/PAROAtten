# test_reorder_layernorm.py
import torch
from inficom import reorder_layernorm, inv_reorder_layernorm
import time
import numpy as np

# Supported spatial patterns
PATTERNS = ["FHW", "FWH", "WFH", "WHF", "HFW", "HWF"]

def per_head_reorder(x, patterns_tensor, F, H, W, inv_reorder=False):
    B, T, C = x.shape
    head_count = patterns_tensor.numel()
    head_dim = C // head_count
    assert C % head_count == 0, "C must be divisible by head_count"

    out = torch.empty_like(x)
    for h in range(head_count):
        pid = patterns_tensor[h].item()
        pat = PATTERNS[pid]
        slice_ = x[..., h*head_dim:(h+1)*head_dim]
        if not inv_reorder:
            view = slice_.reshape(B, F, H, W, head_dim)
            if   pat == "FHW": perm = view
            elif pat == "FWH": perm = view.permute(0,1,3,2,4)
            elif pat == "WFH": perm = view.permute(0,3,1,2,4)
            elif pat == "WHF": perm = view.permute(0,3,2,1,4)
            elif pat == "HFW": perm = view.permute(0,2,1,3,4)
            elif pat == "HWF": perm = view.permute(0,2,3,1,4)
            out_slice = perm.reshape(B, T, head_dim)
        else:
            if   pat == "FHW": view = slice_.reshape(B, F, H, W, head_dim); inv = view
            elif pat == "FWH": view = slice_.reshape(B, F, W, H, head_dim); inv = view.permute(0,1,3,2,4)
            elif pat == "WFH": view = slice_.reshape(B, W, F, H, head_dim); inv = view.permute(0,2,3,1,4)
            elif pat == "WHF": view = slice_.reshape(B, W, H, F, head_dim); inv = view.permute(0,3,2,1,4)
            elif pat == "HFW": view = slice_.reshape(B, H, F, W, head_dim); inv = view.permute(0,2,1,3,4)
            elif pat == "HWF": view = slice_.reshape(B, H, W, F, head_dim); inv = view.permute(0,3,1,2,4)
            out_slice = inv.reshape(B, T, head_dim)
        out[..., h*head_dim:(h+1)*head_dim] = out_slice
    return out

def naive_layernorm(x, shift, scale, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var  = x.var (dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def run_test(B, F, H, W, C, inv_reorder=False, eps=1e-5, dtype=torch.bfloat16):
    print(f"Testing: B={B}, F={F}, H={H}, W={W}, C={C}, inv_reorder={inv_reorder}")
    T = F * H * W
    x     = torch.randn(B, T, C, device="cuda", dtype=dtype)
    shift = torch.randn(B, C, device="cuda", dtype=dtype)
    scale = torch.randn(B, C, device="cuda", dtype=dtype)

    head_dim = 64
    if C % head_dim != 0:
        head_dim = C // 4
    head_count = C // head_dim
    patterns_tensor = torch.zeros(head_count, dtype=torch.int32, device="cuda")

    # 执行 forward
    if inv_reorder:
        x_ln       = naive_layernorm(x, shift, scale, eps)
        x_reord    = per_head_reorder(x_ln, patterns_tensor, F, H, W, inv_reorder=False)
        cuda_out   = inv_reorder_layernorm(x_reord, scale, shift, patterns_tensor, F, H, W, head_dim)
        x_back     = per_head_reorder(x_reord, patterns_tensor, F, H, W, inv_reorder=True)
        naive_out  = naive_layernorm(x_back, shift, scale, eps)
    else:
        x_ln       = naive_layernorm(x, shift, scale, eps)
        naive_out  = per_head_reorder(x_ln, patterns_tensor, F, H, W, inv_reorder)
        cuda_out   = reorder_layernorm(x, scale, shift, patterns_tensor, F, H, W, head_dim)

    # 性能测量（略）

    # 误差计算
    diff     = torch.abs(cuda_out - naive_out)
    max_diff = diff.max().item()
    mean_diff= diff.mean().item()
    max_idx  = torch.where(diff == max_diff)
    print(f"Max abs error {max_diff:.3e} at position {max_idx}, mean abs error {mean_diff:.3e}")

    # 定位出错 head
    b = int(max_idx[0][0]); t = int(max_idx[1][0]); c = int(max_idx[2][0])
    head_id         = c // head_dim
    channel_in_head = c %  head_dim
    print(f"→ Error located at batch={b}, token={t}, channel={c}")
    print(f"  head_id = {head_id}, channel_in_head = {channel_in_head}")
    print(f"  pattern for this head = {PATTERNS[patterns_tensor[head_id].item()]}")

    # 打印这整个 head 的所有 channel 值对比
    start = head_id * head_dim
    end   = start + head_dim
    print("CUDA out for this head slice: ")
    print(cuda_out[b, t, start:end].cpu().numpy())
    print("Naive out for this head slice:")
    print(naive_out[b, t, start:end].cpu().numpy())

    # 最后断言
    tol = 1e-2 if dtype == torch.float32 else 1
    assert max_diff < tol, f"Error {max_diff} exceeds tol {tol}"
    print("Passed ✓\n")
    return True

if __name__ == "__main__":
    all_ok = True
    for B, F, H, W, C in [(2,11,48,85,3072)]:
        for inv in (False, True):
            ok = run_test(B, F, H, W, C, inv_reorder=inv)
            all_ok &= ok
    print(f"\nAll tests: {'Passed ✓' if all_ok else 'Failed ✗'}")
