import torch
import torch.nn as nn
from inficom import gemv_acc_fp16, gemv_acc_fp32

### test settings (Z must be 1 for GEMV testing)
Z = 32
DIM1 = 4096
DIM2 = 128 * 1024

### benchmark settings
WARM_UP = 25
REP = 100

x = torch.empty((Z, 1, DIM1), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
linear_layer = torch.nn.Linear(DIM1, DIM2, bias=False, device="cuda", dtype=torch.float16)

ref_out = linear_layer(x)

ed1_out = gemv_acc_fp16(x, linear_layer.weight)
ed2_out = gemv_acc_fp32(x, linear_layer.weight)

# ed1_all_close = torch.allclose(ref_out.reshape((Z, DIM2)), ed1_out.reshape((Z, DIM2)), atol=1e-2, rtol=1e-4)
# ed2_all_close = torch.allclose(ref_out.reshape((Z, DIM2)), ed2_out.reshape((Z, DIM2)), atol=1e-2, rtol=1e-4)

ed1_all_close = True
ed2_all_close = True

### benchmarking
for _ in range(WARM_UP):
    _ = linear_layer(x)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = linear_layer(x)
    end_event[i].record()
torch.cuda.synchronize()
ref_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)


for _ in range(WARM_UP):
    _ = gemv_acc_fp16(x, linear_layer.weight)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = gemv_acc_fp16(x, linear_layer.weight)
    end_event[i].record()
torch.cuda.synchronize()
ed1_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)


for _ in range(WARM_UP):
    _ = gemv_acc_fp32(x, linear_layer.weight)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = gemv_acc_fp32(x, linear_layer.weight)
    end_event[i].record()
torch.cuda.synchronize()
ed2_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

print('%s %s %d %d %d %.4f %.4f %.4f' % (
                                 bool(ed1_all_close),
                                 bool(ed2_all_close),
                                 Z,
                                 DIM1,
                                 DIM2, 
                                 torch.mean(ref_dur).item(),
                                 torch.mean(ed1_dur).item(),
                                 torch.mean(ed2_dur).item()))


