import torch
import torch.nn as nn
import argparse
from inficom import flat_gemm_m16n64k128x8_bz1_for_fpga

use_op = flat_gemm_m16n64k128x8_bz1_for_fpga
### test settings (Z must not exceed 8 for flat GEMM testing)

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=24)
parser.add_argument('--input-dim', type=int, default=8192)
parser.add_argument('--output-dim', type=int, default=8192)
args = parser.parse_args()

M = args.batch_size
K = args.input_dim
N = args.output_dim

### benchmark settings
WARM_UP = 100
REP = 1000

activation = torch.randint(0, 127, (M, K // 8 * 12), dtype=torch.uint8, device="cuda")
linear_layer = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
# x_reshape = x.reshape((Z, DIM1))
weight = torch.randint(0, 127, (N, K // 2), dtype=torch.uint8, device="cuda")
wet_zeros = torch.randint(0, 127, ((((K + 12)// 13) + 1)//2, 1), dtype=torch.uint8, device="cuda")
wet_scales1 = torch.randint(0, 127, ((K + 12)// 13, 1), dtype=torch.uint8, device="cuda")

act_scales = torch.empty((1), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
wet_scales2 = torch.empty((N), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
output = torch.empty((M, N), dtype=torch.float16, device="cuda")
x = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

ref_out = linear_layer(x)

# torch.cuda.cudart().cudaProfilerStart()
# use_op(activation, act_scales, weight, wet_zeros, wet_scales1, wet_scales2, output)
# torch.cuda.cudart().cudaProfilerStop()

# ed_out = torch.empty((Z, DIM2), dtype=torch.float16, device="cuda")
# use_op(x_reshape, linear_layer.weight, ed_out)

# print(ref_out)
# print(ed_out)

# ed_all_close = torch.allclose(ref_out, ed_out.reshape((Z, 1, DIM2)), atol=1e-2, rtol=1e-2)

# err_matrix = torch.abs(ref_out - ed_out.reshape((Z, 1, DIM2)))

# print(err_matrix.max())

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
    use_op(activation, act_scales, weight, wet_zeros, wet_scales1, wet_scales2, output)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    use_op(activation, act_scales, weight, wet_zeros, wet_scales1, wet_scales2, output)
    end_event[i].record()
torch.cuda.synchronize()
ed_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

print('%d %d %d %.4f %.4f' % (
                                 M,
                                 N,
                                 K, 
                                 torch.mean(ref_dur).item(),
                                 torch.mean(ed_dur).item()))


