import torch
import torch.nn.functional as F
from inficom import chatglm2_ffn_layer_fwd

Z = 2
D = 4096
HD = 13696

x = torch.empty((Z, 1, D), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
r = torch.empty((Z, 1, D), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
rw = torch.empty((D), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
# W1 = torch.empty((D, HD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
# W2 = torch.empty((D, HD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

linear1 = torch.nn.Linear(D, 2 * HD, bias=False, device="cuda", dtype=torch.float16)
linear2 = torch.nn.Linear(HD, D, bias=False, device="cuda", dtype=torch.float16)

w1 = linear1.weight
w2 = linear2.weight

ref_res = (x + r).float()
x_i = ref_res * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + 1e-5)
x_i = x_i.type_as(rw) * rw

# print(x_i)

ref_out = linear1(x_i)
ref_out = torch.chunk(ref_out, 2, dim=-1)
ref_out = F.silu(ref_out[0]) * ref_out[1]
ref_out = linear2(ref_out)
ref_out = ref_out + ref_res.half()

print(ref_out)

# ed_out, ed_res = res_rms_dual_gemm_silu_dot(r, x, rw, w1, w2)
# ed_out = ed_res + linear3(ed_out)
x_t = x.transpose(0, 1).contiguous()
r_t = r.transpose(0, 1).contiguous()
ed_out = chatglm2_ffn_layer_fwd(r_t, x_t, rw, w1, w2)

print(ed_out)

all_close = torch.allclose(ref_out, ed_out.transpose(0, 1), atol=1e-2, rtol=1e-4)

if all_close:
    print('pass!')

print(abs(ed_out.transpose(0, 1) - ref_out).max())

WARM_UP = 25
REP = 100

### benchmarking
for _ in range(WARM_UP):
    ref_res = (x + r).float()
    x_i = ref_res * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + 1e-5)
    x_i = x_i.type_as(rw) * rw
    ref_out = linear1(x_i)
    ref_out = torch.chunk(ref_out, 2, dim=-1)
    ref_out = F.silu(ref_out[0]) * ref_out[1]
    ref_out = linear2(ref_out)
    ref_out = ref_out + ref_res.half()

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    ref_res = (x + r).float()
    x_i = ref_res * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + 1e-5)
    x_i = x_i.type_as(rw) * rw
    ref_out = linear1(x_i)
    ref_out = torch.chunk(ref_out, 2, dim=-1)
    ref_out = F.silu(ref_out[0]) * ref_out[1]
    ref_out = linear2(ref_out)
    ref_out = ref_out + ref_res.half()
    end_event[i].record()
torch.cuda.synchronize()
ref_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)


for _ in range(WARM_UP):
    ed_out = chatglm2_ffn_layer_fwd(r_t, x_t, rw, w1, w2)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    ed_out = chatglm2_ffn_layer_fwd(r_t, x_t, rw, w1, w2)
    end_event[i].record()
torch.cuda.synchronize()
ed_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

print('%.4f %.4f' % (
                    torch.mean(ref_dur).item(),
                    torch.mean(ed_dur).item()))