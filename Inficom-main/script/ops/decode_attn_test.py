import torch
import argparse
import math
from inficom import decode_mha_with_async_softmax, decode_mha_fall_back
from xformers.ops.fmha import memory_efficient_attention_forward

try:
    from flash_attn.flash_attn_interface import \
            flash_attn_kvpacked_func as flash_attn_func
    FLASH_VER = 2
except BaseException:
    try: 
        from flash_attn.flash_attn_interface import \
            flash_attn_unpadded_kvpacked_func as flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
# print('FLASH VERSION: ', FLASH_VER)

### PyTorch implemetation as baseline
def pytorch_attn(q, k, v, scale):
    p = torch.matmul(q, k.transpose(2, 3)) * scale
    p = torch.softmax(p.float(), dim=-1).half()
    out = torch.matmul(p, v)

    return out

### benchmark settings
WARM_UP = 25
REP = 1

### benchmark on various bs, hn, hs and kv cache length
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--head-num', type=int, default=32)
parser.add_argument('--head-size', type=int, default=128)
parser.add_argument('--cache-len', type=int, default=128)
parser.add_argument('--xformers-decoder', default=False, action='store_true')
args = parser.parse_args()
Z = args.batch_size
H = args.head_num
P_SEQ = args.cache_len
D_HEAD = args.head_size

# sm_scale = 1 / math.sqrt(D_HEAD)
sm_scale = 1 / 2

q = torch.empty((Z, H, 1, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
k = torch.empty((Z, H, P_SEQ, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
v = torch.empty((Z, H, P_SEQ, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

q_f = q.transpose(1, 2).contiguous()
k_f = k.transpose(1, 2).contiguous()
v_f = v.transpose(1, 2).contiguous()
kv = torch.zeros((Z, P_SEQ, 2, H, D_HEAD), dtype=torch.float16, device="cuda")
kv[:, :, 0, :, :] = k_f
kv[:, :, 1, :, :] = v_f

seq_list = [P_SEQ] * Z
seq_pos = torch.tensor(seq_list, dtype=torch.int32, device="cuda")

### correctness check
ref_out = pytorch_attn(q, k, v, sm_scale)
ed1_out = decode_mha_with_async_softmax(q_f, k_f, v_f, P_SEQ, P_SEQ, sm_scale, 8)
ed2_out = decode_mha_fall_back(q_f, k_f, v_f, P_SEQ, P_SEQ, sm_scale)
xf_out = memory_efficient_attention_forward(q_f, k_f, v_f, p=0.0, attn_bias=None, scale=sm_scale)
xf_out = xf_out.transpose(1, 2).contiguous()
if args.xformers_decoder:
    xd_out = xformers_mqa_attn(q_f, k_f, v_f, seq_pos, sm_scale)
    xd_out = xd_out.transpose(1, 2).contiguous()
if FLASH_VER == 1:
    q_fl = q_f.reshape((-1, H, D_HEAD)).contiguous()
    kv_fl = kv.reshape((-1, 2, H, D_HEAD)).contiguous()
    q_lengths = torch.full((Z,), fill_value=1, device="cuda")
    cu_seqlens_q = torch.zeros((Z + 1,), device="cuda", dtype=torch.int32)
    cu_seqlens_q[1:] = q_lengths.cumsum(0)
    k_lengths = torch.full((Z,), fill_value=P_SEQ, device="cuda")
    cu_seqlens_k = torch.zeros((Z + 1,), device="cuda", dtype=torch.int32)
    cu_seqlens_k[1:] = k_lengths.cumsum(0)
    flash_out = flash_attn_func(q_fl, kv_fl, cu_seqlens_q, cu_seqlens_k, 1, P_SEQ, 
            0.0, causal=False, softmax_scale=sm_scale).half()
    flash_out = flash_out.reshape((Z, 1, H, D_HEAD)).transpose(1, 2).contiguous()
elif FLASH_VER == 2:
    flash_out = flash_attn_func(q_f, kv, 0.0, causal=False, softmax_scale=sm_scale).half()
    flash_out = flash_out.transpose(1, 2).contiguous() 
elif FLASH_VER == 3:
    flash_out, _ = flash_attn_func(q_f, k_f, v_f, dropout_p=0.0, softmax_scale=sm_scale, causal=False, return_attn_probs=False)

ed1_all_close = torch.allclose(ref_out, ed1_out.reshape((Z, H, 1, D_HEAD)), atol=1e-2, rtol=1e-4)
ed2_all_close = torch.allclose(ref_out, ed2_out.reshape((Z, H, 1, D_HEAD)), atol=1e-2, rtol=1e-4)
# wf_all_close = torch.allclose(ref_out, wf_out, atol=1e-2, rtol=1e-4)
# if ed_all_close:
#     print('edattn passed.')
fs_all_close = torch.allclose(ref_out, flash_out, atol=1e-2, rtol=1e-4)
# if fs_all_close:
#     print('flash passed.')
xf_all_close = torch.allclose(ref_out, xf_out, atol=1e-2, rtol=1e-4)
if args.xformers_decoder:
    xd_all_close = torch.allclose(ref_out, xd_out, atol=1e-2, rtol=1e-4)

torch.cuda.cudart().cudaProfilerStart()

### benchmarking
for _ in range(WARM_UP):
    _ = pytorch_attn(q, k, v, sm_scale)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = pytorch_attn(q, k, v, sm_scale)
    end_event[i].record()
torch.cuda.synchronize()
ref_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

# print('pytorch dur: %.4f ms' % (torch.mean(ref_dur).item()))


for _ in range(WARM_UP):
    _ = decode_mha_with_async_softmax(q_f, k_f, v_f, P_SEQ, P_SEQ, sm_scale, 8)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = decode_mha_with_async_softmax(q_f, k_f, v_f, P_SEQ, P_SEQ, sm_scale, 8)
    end_event[i].record()
torch.cuda.synchronize()
ed1_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

# print('edattn dur: %.4f ms' % (torch.mean(ed_dur).item()))
for _ in range(WARM_UP):
    _ = decode_mha_fall_back(q_f, k_f, v_f, P_SEQ, P_SEQ, sm_scale)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = decode_mha_fall_back(q_f, k_f, v_f, P_SEQ, P_SEQ, sm_scale)
    end_event[i].record()
torch.cuda.synchronize()
ed2_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

if FLASH_VER == 1:
    for _ in range(WARM_UP):
        _ = flash_attn_func(q_fl, kv_fl, cu_seqlens_q, cu_seqlens_k, 1, P_SEQ, 
            0.0, causal=False, softmax_scale=sm_scale)

    start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]  
    for i in range(REP):
        start_event[i].record()
        _ = flash_attn_func(q_fl, kv_fl, cu_seqlens_q, cu_seqlens_k, 1, P_SEQ, 
            0.0, causal=False, softmax_scale=sm_scale)
        end_event[i].record()
    torch.cuda.synchronize()
    fs_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    # print('flash attn %s dur: %.4f ms' % (str(FLASH_VER), torch.mean(fs_dur).item()))

elif FLASH_VER == 2:
    for _ in range(WARM_UP):
        _ = flash_attn_func(q_f, kv, 0.0, causal=False, softmax_scale=sm_scale)

    start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]  
    for i in range(REP):
        start_event[i].record()
        _ = flash_attn_func(q_f, kv, 0.0, causal=False, softmax_scale=sm_scale)
        end_event[i].record()
    torch.cuda.synchronize()
    fs_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

elif FLASH_VER == 3:
    for _ in range(WARM_UP):
        _ = flash_attn_func(q_f, k_f, v_f, dropout_p=0.0, softmax_scale=sm_scale, causal=False, return_attn_probs=False)

    start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]  
    for i in range(REP):
        start_event[i].record()
        _ = flash_attn_func(q_f, k_f, v_f, dropout_p=0.0, softmax_scale=sm_scale, causal=False, return_attn_probs=False)
        end_event[i].record()
    torch.cuda.synchronize()
    fs_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

for _ in range(WARM_UP):
    _ = memory_efficient_attention_forward(q_f, k_f, v_f, p=0.0, attn_bias=None, scale=sm_scale)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = memory_efficient_attention_forward(q_f, k_f, v_f, p=0.0, attn_bias=None, scale=sm_scale)
    end_event[i].record()
torch.cuda.synchronize()
xf_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

if args.xformers_decoder:
    for _ in range(WARM_UP):
        _ = xformers_mqa_attn(q_f, k_f, v_f, seq_pos, sm_scale)
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
    for i in range(REP):
        start_event[i].record()
        _ = xformers_mqa_attn(q_f, k_f, v_f, seq_pos, sm_scale)
        end_event[i].record()
    torch.cuda.synchronize()
    xd_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

torch.cuda.cudart().cudaProfilerStop()

# print('xattn dur: %.4f ms' % (torch.mean(xf_dur).item()))
if args.xformers_decoder:
    print('%s %s %s %s %s %d %.4f %.4f %.4f %.4f %.4f %.4f' % (
                                 bool(fs_all_close),
                                 bool(xf_all_close),
                                 bool(xd_all_close),
                                 bool(ed1_all_close),
                                 bool(ed2_all_close),
                                #  bool(wf_all_close),
                                 P_SEQ, 
                                 torch.mean(ref_dur).item(),
                                 torch.mean(fs_dur).item(),
                                 torch.mean(xf_dur).item(),
                                 torch.mean(xd_dur).item(),
                                 torch.mean(ed1_dur).item(),
                                 torch.mean(ed2_dur).item()))
else:
    print('%s %s %s %s %d %.4f %.4f %.4f %.4f %.4f' % (
                                 bool(fs_all_close),
                                 bool(xf_all_close),
                                 bool(ed1_all_close),
                                 bool(ed2_all_close),
                                #  bool(wf_all_close),
                                 P_SEQ, 
                                 torch.mean(ref_dur).item(),
                                 torch.mean(fs_dur).item(),
                                 torch.mean(xf_dur).item(),
                                 torch.mean(ed1_dur).item(),
                                 torch.mean(ed2_dur).item()))

