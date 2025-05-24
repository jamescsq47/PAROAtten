import torch
import torch.nn.functional as F
from inficom import dual_linear_silu_dot_fwd

### test setting ###
Z = 8
DIM1 = 5120
DIM2 = 13824
# DIM1 = 4096
# DIM2 = 11008

x = torch.empty((Z, 1, DIM1), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
r = torch.empty((Z, 1, DIM1), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

linear1 = torch.nn.Linear(DIM1, DIM2, bias=False, device="cuda", dtype=torch.float16)
linear2 = torch.nn.Linear(DIM1, DIM2, bias=False, device="cuda", dtype=torch.float16)
w1 = linear1.weight
w2 = linear2.weight


ref_out = F.silu(linear1(x)) * linear2(x)
print('ref value:', ref_out)


ed_out = dual_linear_silu_dot_fwd(x, w1, w2)
print('ed value:', ed_out)

all_close = torch.allclose(ref_out, ed_out, atol=1e-2, rtol=1e-4)

if all_close:
    print('pass!')

print(abs(ed_out - ref_out).max())

WARM_UP = 25
REP = 100

### benchmarking
for _ in range(WARM_UP):
    ref_out = F.silu(linear1(x)) * linear2(x)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    ref_out = F.silu(linear1(x)) * linear2(x)
    end_event[i].record()
torch.cuda.synchronize()
ref_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)


for _ in range(WARM_UP):
    ed_out = dual_linear_silu_dot_fwd(x, w1, w2)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    ed_out = dual_linear_silu_dot_fwd(x, w1, w2)
    end_event[i].record()
torch.cuda.synchronize()
ed_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

print('%s %d %d %d %.4f %.4f' % (
                                 bool(all_close),
                                 Z,
                                 DIM1,
                                 DIM2, 
                                 torch.mean(ref_dur).item(),
                                 torch.mean(ed_dur).item()))


