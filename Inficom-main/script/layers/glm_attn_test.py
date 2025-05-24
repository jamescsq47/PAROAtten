import torch
import math
from inficom import chatglm2_attn_layer_fwd


def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


data = torch.load('/share/hongke/data/attn-data/glm21698230938.0715861.pth')

Z = 1
H = 32
P_SEQ = data['pos_start'] + 1
D_HEAD = 128
M_LEN = data['max_seq_len']
sm_scale = data['scale']
attn_max = 8
x = data['x']
# x = torch.empty((Z, 1, H * D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=5.0)
print(x)

wqkv = torch.nn.Linear(H * D_HEAD, (H * D_HEAD + 256 * 2), bias=False, device="cuda", dtype=torch.float16)
wo = torch.nn.Linear(H * D_HEAD, H * D_HEAD, bias=False, device="cuda", dtype=torch.float16)

# RW = torch.empty((H * D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
RW = data['RW']

wqkv.weight = data['WQKV']
wqkv.bias = data['BQKV']
wo.weight = data['OW']

WQKV = wqkv.weight
BQKV = wqkv.bias
WO = wo.weight

k = data['cache_k'][:P_SEQ - 1, :, :, :].expand(P_SEQ - 1, Z, 2, D_HEAD).reshape((P_SEQ - 1, Z, 2, D_HEAD))
v = data['cache_v'][:P_SEQ - 1, :, :, :].expand(P_SEQ - 1, Z, 2, D_HEAD).reshape((P_SEQ - 1, Z, 2, D_HEAD))

k_ref = torch.zeros((P_SEQ, Z, 2, D_HEAD), dtype=torch.float16, device="cuda")
k_ref[:P_SEQ - 1, :, :, :] = k
v_ref = torch.zeros((P_SEQ, Z, 2, D_HEAD), dtype=torch.float16, device="cuda")
v_ref[:P_SEQ - 1, :, :, :] = v

# k_cache = torch.empty((Z, M_LEN, H, D_HEAD), dtype=torch.float16, device="cuda")
# v_cache = torch.empty((Z, M_LEN, H, D_HEAD), dtype=torch.float16, device="cuda")
# k_cache[:, :P_SEQ - 1, :, :] = k
# v_cache[:, :P_SEQ - 1, :, :] = v
k_cache = data['cache_k'].expand(M_LEN, Z, 2, D_HEAD).contiguous()
v_cache = data['cache_v'].expand(M_LEN, Z, 2, D_HEAD).contiguous()

rotary_pos_emb = data['freq']

# RMSNorm
x_i = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
x_i = x_i.type_as(RW) * RW

# torch baseline
xqkv = wqkv(x_i).view(1, Z, (H * D_HEAD + 512))

(query_layer, key_layer, value_layer) = xqkv.split([4096, 256, 256,], dim=-1,)
query_layer = query_layer.view(query_layer.size()[:-1] + (32, 128))
key_layer = key_layer.view(key_layer.size()[:-1] + (2, 128))
value_layer = value_layer.view(value_layer.size()[:-1] + (2, 128))
query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

# print(xq)
# print(xk_out.view(Z, 1, H * D_HEAD))
# print(xv.view(Z, 1, H * D_HEAD))
# print(freqs_cis)
# print(xq_out.view(Z, 1, H * D_HEAD))

k_ref[P_SEQ - 1, :, :, :] = key_layer.reshape((1, Z, 2, D_HEAD))
v_ref[P_SEQ - 1, :, :, :] = value_layer.reshape((1, Z, 2, D_HEAD))

key_layer = key_layer.unsqueeze(-2)
key_layer = key_layer.expand(-1, -1, -1, 16, -1)
key_layer = key_layer.contiguous().view(key_layer.size()[:2] + (32, 128))
value_layer = value_layer.unsqueeze(-2)
value_layer = value_layer.expand(-1, -1, -1, 16, -1)
value_layer = value_layer.contiguous().view(value_layer.size()[:2] + (32, 128))

query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, None)
context_layer = context_layer.permute(2, 0, 1, 3)
new_context_layer_shape = context_layer.size()[:-2] + (128,)
context_layer = context_layer.reshape(*new_context_layer_shape)

# print(ref_s)
ref_out = wo(context_layer)

print(ref_out)
# print(k_ref.transpose(1, 2)[:, P_SEQ - 1, :, :])

# print(x.shape)
# print(RW.shape)
# print(WQ.shape)
# print(WO.shape)
# print(k_cache.shape)
# print(v_cache.shape)
# print(freqs_cis.shape)

# rx_out, q_out, s_out, ed_out = attn_for_llama(x, RW, WQ, WK, WV, WO, k_cache, v_cache, freqs_cis, M_LEN, P_SEQ - 1, sm_scale, attn_max)
ed_out = chatglm2_attn_layer_fwd(x, RW, WQKV, BQKV, WO, k_cache, v_cache, rotary_pos_emb, M_LEN, P_SEQ - 1, sm_scale, attn_max)


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
    ed_out = chatglm2_attn_layer_fwd(x, RW, WQKV, BQKV, WO, k_cache, v_cache, rotary_pos_emb, M_LEN, P_SEQ - 1, sm_scale, attn_max)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    ed_out = chatglm2_attn_layer_fwd(x, RW, WQKV, BQKV, WO, k_cache, v_cache, rotary_pos_emb, M_LEN, P_SEQ - 1, sm_scale, attn_max)
    end_event[i].record()
torch.cuda.synchronize()
ed_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

print('%.4f' % (torch.mean(ed_dur).item()))