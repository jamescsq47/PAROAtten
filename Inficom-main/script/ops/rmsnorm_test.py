import torch
import torch.nn as nn
from inficom import rmsnorm

Z = 2
DIM  = 4096

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((dim), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

x = torch.empty((Z, 1, DIM), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

rmsnorm_layer = RMSNorm(DIM)
ref_out = rmsnorm_layer(x)

ed_out = rmsnorm(x, rmsnorm_layer.weight)

all_close = torch.allclose(ref_out, ed_out, atol=1e-2, rtol=1e-4)

### benchmark settings
WARM_UP = 25
REP = 100

### benchmarking
for _ in range(WARM_UP):
    _ = rmsnorm_layer(x)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = rmsnorm_layer(x)
    end_event[i].record()
torch.cuda.synchronize()
ref_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)


for _ in range(WARM_UP):
    _ = rmsnorm(x, rmsnorm_layer.weight)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = rmsnorm(x, rmsnorm_layer.weight)
    end_event[i].record()
torch.cuda.synchronize()
ed_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)


print('%s %d %d %.4f %.4f' % (bool(all_close),
                                 Z, 
                                 DIM, 
                                 torch.mean(ref_dur).item(),
                                 torch.mean(ed_dur).item()))
