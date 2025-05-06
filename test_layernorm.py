# test_reorder_layernorm.py
import torch
import fused_reorder_ln
import numpy as np
import time

def naive_reorder(x, pattern, F, H, W, inv_reorder=False):
    """朴素的reorder实现"""
    B, T, C = x.shape
    assert T == F*H*W, f"期望 T={F*H*W}，但得到 T={T}"
    x_reshaped = x.reshape(B, F, H, W, C)
    if pattern == "FHW":
        reordered = x_reshaped
    elif pattern == "FWH":
        reordered = x_reshaped.permute(0, 1, 3, 2, 4)
    elif pattern == "WFH":
        reordered = x_reshaped.permute(0, 3, 1, 2, 4)
    elif pattern == "WHF":
        reordered = x_reshaped.permute(0, 3, 2, 1, 4)
    elif pattern == "HFW":
        reordered = x_reshaped.permute(0, 2, 1, 3, 4)
    elif pattern == "HWF":
        reordered = x_reshaped.permute(0, 2, 3, 1, 4)
    else:
        raise ValueError(f"不支持的模式: {pattern}")

    if inv_reorder:
        # 逆重排序：直接把 permuted 结果当作“已重排”，再展平
        return reordered.reshape(B, T, C)
    else:
        return reordered.reshape(B, T, C)

def naive_inv_reorder(x, pattern, F, H, W):
    """朴素的逆重排到 FHW 展开顺序"""
    B, T, C = x.shape
    assert T == F*H*W
    if pattern == "FHW":
        x_view = x.reshape(B, F, H, W, C)
        x_perm = x_view
    elif pattern == "FWH":
        x_view = x.reshape(B, F, W, H, C)
        x_perm = x_view.permute(0,1,3,2,4)
    elif pattern == "WFH":
        x_view = x.reshape(B, W, F, H, C)
        x_perm = x_view.permute(0,2,3,1,4)
    elif pattern == "WHF":
        x_view = x.reshape(B, W, H, F, C)
        x_perm = x_view.permute(0,3,2,1,4)
    elif pattern == "HFW":
        x_view = x.reshape(B, H, F, W, C)
        x_perm = x_view.permute(0,2,1,3,4)
    elif pattern == "HWF":
        x_view = x.reshape(B, H, W, F, C)
        x_perm = x_view.permute(0,3,1,2,4)
    else:
        raise ValueError(f"Unsupported pattern: {pattern}")
    return x_perm.reshape(B, T, C)

def naive_layernorm(x, shift, scale, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def run_test(B=2, F=4, H=8, W=8, C=32, pattern="FWH", inv_reorder=False, eps=1e-5, dtype=torch.float32):
    print(f"\n测试配置: B={B},F={F},H={H},W={W},C={C},pattern={pattern},inv_reorder={inv_reorder},dtype={dtype}")
    T = F*H*W
    x = torch.randn(B, T, C, device="cuda", dtype=dtype)
    # 保持 shift/scale 同 x dtype
    shift = torch.randn(B, C, device="cuda", dtype=dtype)
    scale = torch.randn(B, C, device="cuda", dtype=dtype)

    x_cuda = x.clone()
    x_naive = x.clone()
    if not inv_reorder:

    # 调用 CUDA kernel
        cuda_out = fused_reorder_ln.forward(x_cuda, shift, scale, F, H, W, eps, pattern, inv_reorder)[0]

        # 朴素流程
        # 1. reorder 或 inv_reorder
        naive_re = naive_reorder(x_naive, pattern, F, H, W, inv_reorder)
        # 2. layernorm
        naive_out = naive_layernorm(naive_re, shift, scale, eps)
    if inv_reorder:
        # first reorder x_cuda into the pattern layout
        x_cuda = naive_reorder(x_cuda, pattern, F, H, W, inv_reorder=False).contiguous()
        x_naive = naive_reorder(x_naive, pattern, F, H, W, inv_reorder=False).contiguous()

        cuda_out = fused_reorder_ln.forward(x_cuda, shift, scale, F, H, W, eps, pattern, inv_reorder)[0]
        naive_re = naive_inv_reorder(x_naive, pattern, F, H, W)
        naive_out = naive_layernorm(naive_re, shift, scale, eps)
    # 对比
    abs_diff = torch.abs(cuda_out - naive_out)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    rel_diff = abs_diff / (torch.abs(naive_out) + 1e-8)
    max_rel, mean_rel = rel_diff.max().item(), rel_diff.mean().item()
    print(f"最大绝对误差 {max_diff:.3e}, 平均绝对误差 {mean_diff:.3e}")
    print(f"最大相对误差 {max_rel:.3e}, 平均相对误差 {mean_rel:.3e}")
    passed = max_diff < (1e-3 if dtype==torch.float32 else 1e-2)
    print("通过 ✓" if passed else "失败 ✗")
    return passed

if __name__ == "__main__":
    all_passed = True
    patterns = ["FHW","FWH","WFH","WHF","HFW","HWF"]
    for pattern in patterns:
        # test both modes
        for inv in (False, True):
            passed = run_test(pattern=pattern, inv_reorder=inv)
            all_passed &= passed

    sizes = [(2,4,8,8,256),(2,11,48,85,3072)]
    for B,F,H,W,C in sizes:
        for inv in (False, True):
            passed = run_test(B, F, H, W, C, inv_reorder=inv)
            all_passed &= passed

    print(f"\n总体结果: {'全部通过 ✓' if all_passed else '部分失败 ✗'}")
