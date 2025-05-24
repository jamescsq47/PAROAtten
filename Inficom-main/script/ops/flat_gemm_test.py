import torch
import torch.nn as nn
import argparse
from inficom import flat_gemm_m8n32k256x8_bz1, flat_gemm_mix_for_decode, flat_gemm_m16n32k256x8_bz1, flat_gemm_m16n64k128x8_bz1

use_op = flat_gemm_m16n64k128x8_bz1
### test settings (Z must not exceed 8 for flat GEMM testing)

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--input-dim', type=int, default=4096)
parser.add_argument('--output-dim', type=int, default=4096)
args = parser.parse_args()

Z = args.batch_size
DIM1 = args.input_dim
DIM2 = args.output_dim

### benchmark settings
WARM_UP = 100
REP = 1000

x = torch.empty((Z, 1, DIM1), dtype=torch.float16, device="cuda").normal_(mean=0., std=1.0)
linear_layer = torch.nn.Linear(DIM1, DIM2, bias=False, device="cuda", dtype=torch.float16)
x_reshape = x.reshape((Z, DIM1))
weight = linear_layer.weight

ref_out = linear_layer(x)

ed_out = torch.empty((Z, DIM2), dtype=torch.float16, device="cuda")
use_op(x_reshape, linear_layer.weight, ed_out)

print(ref_out)
print(ed_out)

ed_all_close = torch.allclose(ref_out, ed_out.reshape((Z, 1, DIM2)), atol=1e-2, rtol=1e-2)

err_matrix = torch.abs(ref_out - ed_out.reshape((Z, 1, DIM2)))

print(err_matrix.max())

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
    use_op(x_reshape, linear_layer.weight, ed_out)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    use_op(x_reshape, linear_layer.weight, ed_out)
    end_event[i].record()
torch.cuda.synchronize()
ed_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

print('%s %d %d %d %.4f %.4f' % (
                                 bool(ed_all_close),
                                 Z,
                                 DIM1,
                                 DIM2, 
                                 torch.mean(ref_dur).item(),
                                 torch.mean(ed_dur).item()))


