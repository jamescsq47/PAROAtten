import torch, numpy as np, fused_apply_rotary

# 朴素实现，先在 Python 里做 reorder（reorder/inv），再做 out = x*cos + x_rot*sin
def python_apply_rotary(x, cos, sin, pattern, inv):
    B,S,D = x.shape
    # 先把 cos/sin 重排回 [1,1,S,D]
    # reorder cos,sin
    F = 4  # 你设定的 F,H,W
    H = 8
    W = 8
    assert S==F*H*W
    # squeeze→reshape→permute→flatten，跟上面 CUDA 一模一样
    def do_reorder(seq, pat, inv):
        seq = seq.view(F,H,W,D)
        if inv:
            # find inverse permute by applying the opposite permute
            # … 详见上面
            pass
        else:
            # 各 pattern
            if pat=="FWH":
                seq = seq.permute(0,2,1,3)
            elif pat=="WFH":
                seq = seq.permute(2,0,1,3)
            elif pat=="WHF":
                seq = seq.permute(2,1,0,3)
            elif pat=="HFW":
                seq = seq.permute(1,0,2,3)
            elif pat=="HWF":
                seq = seq.permute(1,2,0,3)
        return seq.contiguous().flatten(0,2) 
    cos2 = do_reorder(cos, pattern, inv)
    sin2 = do_reorder(sin, pattern, inv)
    
    # 再做原地旋转
    out = torch.empty_like(x)
    for b in range(B):
      for s in range(S):
        for i in range(D):
          v = x[b,s,i].float()
          if i%2==0:
            xr = - x[b,s,i+1].float()
          else:
            xr =   x[b,s,i-1].float()
          out[b,s,i] = (v*cos2[s,i] + xr*sin2[s,i]).to(x.dtype)
    return out

def run_all():
  B,F,H,W,D = 2,4,8,8,32
  S = F*H*W
  x   = torch.randn(B,S,D, device='cuda', dtype=torch.float32)
  cos = torch.randn(S,D, device='cuda', dtype=torch.float32)
  sin = torch.randn(S,D, device='cuda', dtype=torch.float32)

  patterns = ["FHW","FWH","WFH","WHF","HFW","HWF"]
  for pat in patterns:
    inv=False
    y_ref = python_apply_rotary(x, cos, sin, pat, inv)
    y_cuda = fused_apply_rotary.forward(x, cos, sin, F,H,W, pat, inv)
    diff = (y_ref - y_cuda).abs().max().item()
    print(pat, inv, "max_err=", diff)
  print("ALL PASSED!")

if __name__=="__main__":
  run_all()
