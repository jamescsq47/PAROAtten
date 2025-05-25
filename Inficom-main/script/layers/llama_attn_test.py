import torch
import math
from inficom import llama2_attn_layer_fwd


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


Z = 1
H = 32
P_SEQ = 13
D_HEAD = 128
M_LEN = 1024

sm_scale = 1 / math.sqrt(D_HEAD)
attn_max = 8

data = torch.load('/share/hongke/data/attn-data/1696869148.7623668.pth')
# data = torch.load('/share/hongke/data/attn-data/1697616045.820528.pth')
# data = torch.load('/share/hongke/data/attn-data/1697633571.0250895.pth')
x = data['x']
# x = torch.empty((Z, 1, H * D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=5.0)
print(x)

wq = torch.nn.Linear(H * D_HEAD, H * D_HEAD, bias=False, device="cuda", dtype=torch.float16)
wk = torch.nn.Linear(H * D_HEAD, H * D_HEAD, bias=False, device="cuda", dtype=torch.float16)
wv = torch.nn.Linear(H * D_HEAD, H * D_HEAD, bias=False, device="cuda", dtype=torch.float16)
wo = torch.nn.Linear(H * D_HEAD, H * D_HEAD, bias=False, device="cuda", dtype=torch.float16)

# RW = torch.empty((H * D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
RW = data['RW']

wq.weight = data['WQ']
wk.weight = data['WK']
wv.weight = data['WV']
wo.weight = data['WO']

WQ = wq.weight
WK = wk.weight
WV = wv.weight
WO = wo.weight

# k = torch.empty((Z, P_SEQ - 1, H, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
# v = torch.empty((Z, P_SEQ - 1, H, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
k = data['cache_k'][:, :P_SEQ - 1, :, :].expand(Z, P_SEQ - 1, H, D_HEAD).reshape((Z, P_SEQ - 1, H, D_HEAD))
v = data['cache_v'][:, :P_SEQ - 1, :, :].expand(Z, P_SEQ - 1, H, D_HEAD).reshape((Z, P_SEQ - 1, H, D_HEAD))

k_ref = torch.zeros((Z, P_SEQ, H, D_HEAD), dtype=torch.float16, device="cuda")
k_ref[:, :P_SEQ - 1, :, :] = k
v_ref = torch.zeros((Z, P_SEQ, H, D_HEAD), dtype=torch.float16, device="cuda")
v_ref[:, :P_SEQ - 1, :, :] = v

# k_cache = torch.empty((Z, M_LEN, H, D_HEAD), dtype=torch.float16, device="cuda")
# v_cache = torch.empty((Z, M_LEN, H, D_HEAD), dtype=torch.float16, device="cuda")
# k_cache[:, :P_SEQ - 1, :, :] = k
# v_cache[:, :P_SEQ - 1, :, :] = v
k_cache = data['cache_k'].expand(Z, M_LEN, H, D_HEAD).contiguous()
v_cache = data['cache_v'].expand(Z, M_LEN, H, D_HEAD).contiguous()

freqs_cis_ref = torch.tensor([[-0.5328-0.8462j,  0.4822+0.8761j, -0.0312-0.9995j, -0.7164+0.6977j,
          0.9332+0.3593j,  0.2033-0.9791j, -0.9626-0.2708j, -0.5184+0.8551j,
          0.5486+0.8361j,  0.9999+0.0152j,  0.6756-0.7373j,  0.0107-0.9999j,
         -0.5829-0.8125j, -0.9210-0.3896j, -0.9972+0.0744j, -0.8844+0.4667j,
         -0.6663+0.7457j, -0.4086+0.9127j, -0.1534+0.9882j,  0.0771+0.9970j,
          0.2739+0.9618j,  0.4357+0.9001j,  0.5654+0.8248j,  0.6675+0.7446j,
          0.7470+0.6649j,  0.8081+0.5890j,  0.8549+0.5188j,  0.8905+0.4549j,
          0.9175+0.3977j,  0.9379+0.3468j,  0.9533+0.3019j,  0.9649+0.2625j,
          0.9737+0.2280j,  0.9802+0.1979j,  0.9852+0.1716j,  0.9889+0.1488j,
          0.9916+0.1290j,  0.9937+0.1118j,  0.9953+0.0968j,  0.9965+0.0839j,
          0.9974+0.0727j,  0.9980+0.0629j,  0.9985+0.0545j,  0.9989+0.0472j,
          0.9992+0.0409j,  0.9994+0.0354j,  0.9995+0.0307j,  0.9996+0.0266j,
          0.9997+0.0230j,  0.9998+0.0199j,  0.9999+0.0172j,  0.9999+0.0149j,
          0.9999+0.0129j,  0.9999+0.0112j,  1.0000+0.0097j,  1.0000+0.0084j,
          1.0000+0.0073j,  1.0000+0.0063j,  1.0000+0.0055j,  1.0000+0.0047j,
          1.0000+0.0041j,  1.0000+0.0035j,  1.0000+0.0031j,  1.0000+0.0027j]], device="cuda")

freqs_cis = torch.view_as_real(freqs_cis_ref).reshape((1, 1, 1, D_HEAD))

# RMSNorm
x_i = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
x_i = x_i.type_as(RW) * RW

# torch baseline
xq = wq(x_i).view(Z, 1, H, D_HEAD)
xk = wk(x_i).view(Z, 1, H, D_HEAD)
xv = wv(x_i).view(Z, 1, H, D_HEAD)

xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
freqs_cis_ref = reshape_for_broadcast(freqs_cis_ref, xq_)
xq_out = torch.view_as_real(xq_ * freqs_cis_ref).flatten(3).type_as(xq)
xk_out = torch.view_as_real(xk_ * freqs_cis_ref).flatten(3).type_as(xk)

# print(xq)
# print(xk_out.view(Z, 1, H * D_HEAD))
# print(xv.view(Z, 1, H * D_HEAD))
# print(freqs_cis)
# print(xq_out.view(Z, 1, H * D_HEAD))

k_ref[:, P_SEQ - 1, :, :] = xk_out.reshape((Z, H, D_HEAD))
v_ref[:, P_SEQ - 1, :, :] = xv.reshape((Z, H, D_HEAD))

k_ref = k_ref.transpose(1, 2)
v_ref = v_ref.transpose(1, 2)
q_ref = xq_out.transpose(1, 2)

p = torch.matmul(q_ref, k_ref.transpose(2, 3)) * sm_scale

# print(p.float())
# print(torch.exp(p.float() - attn_max).sum())

p = torch.softmax(p.float(), dim=-1).half()

# p = p.float() - attn_max

# p = torch.exp(p)

# print(p.shape)

# p_sum = p.sum(dim=-1, keepdim=True)

# p = (p / p_sum).half()

ref_out = torch.matmul(p, v_ref)
ref_s = ref_out.transpose(1, 2).contiguous().reshape((Z, 1, -1))

# print(ref_s)
ref_out = wo(ref_s)

print(ref_out)
# print(k_ref.transpose(1, 2)[:, P_SEQ - 1, :, :])

# print(x.shape)
# print(RW.shape)
# print(WQ.shape)
# print(WO.shape)
# print(k_cache.shape)
# print(v_cache.shape)
# print(freqs_cis.shape)

torch.cuda.synchronize()

# rx_out, q_out, s_out, ed_out = attn_for_llama(x, RW, WQ, WK, WV, WO, k_cache, v_cache, freqs_cis, M_LEN, P_SEQ - 1, sm_scale, attn_max)
ed_out = llama2_attn_layer_fwd(x, RW, WQ, WK, WV, WO, k_cache, v_cache, freqs_cis, M_LEN, P_SEQ - 1, sm_scale, attn_max)

torch.cuda.synchronize()

# print(q_out)
# print(s_out)
print(ed_out)
# print(ed_out)

all_close = torch.allclose(ref_out, ed_out, atol=1e-4, rtol=1e-2)

if all_close:
    print('out proj pass!')


# all_close = torch.allclose(ref_s, s_out, atol=1e-2, rtol=1e-4)

# # Q after RoPE checked
# # all_close = torch.allclose(xq_out.view(Z, 1, H * D_HEAD), ed_out, atol=1e-2, rtol=1e-4)

# if all_close:
#     print('attention pass!')

# all_close = torch.allclose(xk_out.view(Z, 1, H * D_HEAD), k_cache[:, P_SEQ - 1, :, :].view(Z, 1, H * D_HEAD), atol=1e-2, rtol=1e-4)


# if all_close:
#     print('GEMM + RoPE pass!')

WARM_UP = 25
REP = 100

for _ in range(WARM_UP):
    ed_out = llama2_attn_layer_fwd(x, RW, WQ, WK, WV, WO, k_cache, v_cache, freqs_cis, M_LEN, P_SEQ - 1, sm_scale, attn_max)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    ed_out = llama2_attn_layer_fwd(x, RW, WQ, WK, WV, WO, k_cache, v_cache, freqs_cis, M_LEN, P_SEQ - 1, sm_scale, attn_max)
    end_event[i].record()
torch.cuda.synchronize()
ed_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

print('%.4f' % (torch.mean(ed_dur).item()))