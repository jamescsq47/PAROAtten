import torch
import torch.nn as nn
from inficom import add_residual

Z = 2
DIM  = 4096

x = torch.empty((Z, 1, DIM), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
r = torch.empty((Z, 1, DIM), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

ref_out = x + r

ed_out = add_residual(x, r)

all_close = torch.allclose(ref_out, ed_out, atol=1e-2, rtol=1e-4)

### benchmark settings
WARM_UP = 25
REP = 100

### benchmarking
for _ in range(WARM_UP):
    _ = x + r

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = x + r
    end_event[i].record()
torch.cuda.synchronize()
ref_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)


for _ in range(WARM_UP):
    _ = add_residual(x, r)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = add_residual(x, r)
    end_event[i].record()
torch.cuda.synchronize()
ed_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)


print('%s %d %d %.4f %.4f' % (bool(all_close),
                                 Z, 
                                 DIM, 
                                 torch.mean(ref_dur).item(),
                                 torch.mean(ed_dur).item()))
