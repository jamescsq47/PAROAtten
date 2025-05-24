import torch
from inficom import attn_proj_rope_kv_cat_fwd

### test setting ###
Z = 1
DIM = 4096         # hidden_size(Llama2-7B)
LEN = 1025           # seq_len, use LEN-1 context to infer the LEN-th token
MAX_LEN = 4096       # max length of KV cache
HN = 32              # num_heads in attn
HS = 128             # head_dim in attn (hidden_size / head_num)

num_heads = HN
head_dim = HS

### test tensors init ##
x = torch.empty((Z, 1, DIM), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)  
K = torch.empty((Z, MAX_LEN, HN, HS), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
V = torch.empty((Z, MAX_LEN, HN, HS), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
k_ref = torch.zeros((Z, LEN, HN, HS), dtype=torch.float16, device="cuda")
k_ref[:, :LEN - 1, :, :] = K[:, :LEN - 1, :, :]
v_ref = torch.zeros((Z, LEN, HN, HS), dtype=torch.float16, device="cuda")
v_ref[:, :LEN - 1, :, :] = V[:, :LEN - 1, :, :]

linearQ = torch.nn.Linear(DIM, DIM, bias=False, device="cuda", dtype=torch.float16)
linearK = torch.nn.Linear(DIM, DIM, bias=False, device="cuda", dtype=torch.float16)
linearV = torch.nn.Linear(DIM, DIM, bias=False, device="cuda", dtype=torch.float16)
wq = linearQ.weight
wk = linearK.weight
wv = linearV.weight

# freqs_cis is used in rope layer, simplify for test
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

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

freqs_cis_ref = precompute_freqs_cis(head_dim, MAX_LEN)
freqs_cis_ref = freqs_cis_ref.to("cuda")
freqs_cis_ref = freqs_cis_ref[LEN : LEN + 1]
freqs_cis = torch.view_as_real(freqs_cis_ref).reshape((1, 1, 1, head_dim))

# torch baseline
def llama2_hf_impl(X, LQ, LK, LV, freqs_cis, k_ref, v_ref):
    xq = LQ(X).view(Z, 1, num_heads, head_dim)
    xk = LK(X).view(Z, 1, num_heads, head_dim)
    xv = LV(X).view(Z, 1, num_heads, head_dim)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_ref = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis_ref).flatten(3).type_as(xq)
    xk_out = torch.view_as_real(xk_ * freqs_cis_ref).flatten(3).type_as(xk)

    k_ref[:, LEN - 1, :, :] = xk_out.reshape((Z, num_heads, head_dim))
    v_ref[:, LEN - 1, :, :] = xv.reshape((Z, num_heads, head_dim))

    k_ref = k_ref.transpose(1, 2)
    v_ref = v_ref.transpose(1, 2)
    q_ref = xq_out.transpose(1, 2)
    q_ref = q_ref.reshape(Z, 1, DIM)
    return q_ref

# infi impl
def infi_fuse_impl():
    query_states = attn_proj_rope_kv_cat_fwd(x, wq, wk, wv, K, V, freqs_cis, MAX_LEN, LEN - 1)
    return query_states

ref_out = llama2_hf_impl(x, linearQ, linearK, linearV, freqs_cis_ref, k_ref, v_ref)
print('ref value:', ref_out)


ed_out = infi_fuse_impl()
print('ed value:', ed_out)

all_close = torch.allclose(ref_out, ed_out, atol=1e-2, rtol=1e-4)

if all_close:
    print('pass!')

print(abs(ed_out - ref_out).max())

WARM_UP = 25
REP = 100

### benchmarking
for _ in range(WARM_UP):
    ref_out = llama2_hf_impl(x, linearQ, linearK, linearV, freqs_cis_ref, k_ref, v_ref)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    ref_out = llama2_hf_impl(x, linearQ, linearK, linearV, freqs_cis_ref, k_ref, v_ref)
    end_event[i].record()
torch.cuda.synchronize()
ref_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)


for _ in range(WARM_UP):
    ed_out = infi_fuse_impl()

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    ed_out = infi_fuse_impl()
    end_event[i].record()
torch.cuda.synchronize()
ed_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

print('%s %d %d %d %.4f %.4f' % (
                                 bool(all_close),
                                 Z,
                                 DIM,
                                 LEN, 
                                 torch.mean(ref_dur).item(),
                                 torch.mean(ed_dur).item()))