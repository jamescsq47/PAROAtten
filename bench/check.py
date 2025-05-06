import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from flash_attn.utils.benchmark import benchmark_forward
import paroattention._qattn_sm80 as qattn
import sageattention._qattn_sm80 as qattn_sage
import argparse
from quant import per_block_int8 as per_block_int8_cuda
from quant import per_warp_int8 as per_warp_int8_cuda
import matplotlib.pyplot as plt

def compact(x,y):
    return ((x <<4) | (y&0x0F))
def process_tensor(tensor):
    tensor_flat = tensor.view(-1)
    result = torch.empty(tensor_flat.size(0) // 2, dtype=torch.int8, device=tensor.device)    
    result[:] = compact(tensor_flat[0::2], tensor_flat[1::2])
    result = result.view(tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)//2)
    zero = torch.zeros(tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)//2, dtype=torch.int8, device=tensor.device)
    result=torch.cat((result,result),dim=3)
    return result


parser = argparse.ArgumentParser(description='Benchmark QK INT8 PV FP16')
parser.add_argument('--quant_gran', type=str, default='per_warp', choices=['per_block', 'per_warp', 'per_thread'], help='Quantization granularity')
parser.add_argument('--pv_accum_dtype', type=str, default='fp16+fp32', choices=['fp16', 'fp16+fp32', 'fp32'])
args = parser.parse_args()


print(f"PAROAttention QK Int8 PV fp16 Benchmark with Given Data")
print(f"pv_accum_dtype: {args.pv_accum_dtype}")

CTA_Q = 64
CTA_K = 64
WARP_Q = 32 
WARP_K = 64

kernel_int8 = qattn.qk_int8_sv_f16_accum_f16_attn
kernel_sage = qattn_sage.qk_int8_sv_f16_accum_f16_attn

_qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2 # 'per_warp'

sparse = torch.load("/home/xieruiqi/diffuser-dev/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/kernel_sparse_plan.pth")
sparse = sparse[0,1,:,:,:]
sparse = sparse.to(torch.bool).cuda() 
print(sparse.shape) #torch.Size([48, 278, 278]) one: calculate; zero: skip
all_one_sparse = torch.ones((48, 278, 278), dtype=torch.bool).cuda()  

q = torch.load("/home/xieruiqi/diffuser-dev/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/query_permute.pth").cuda()
print(q.dtype)# torch.Size([2, 48, 17776, 64]) torch.bfloat16
q = q[:1,:,:,:]
k = torch.load("/home/xieruiqi/diffuser-dev-225exp/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/key_permute.pth").cuda()
print(k.dtype)# torch.Size([2, 48, 17776, 64]) torch.bfloat16
k = k[:1,:,:,:]
v = torch.load("/home/xieruiqi/diffuser-dev-225exp/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/value_permute.pth").cuda()
print(v.dtype)# torch.Size([2, 48, 17776, 64]) torch.bfloat16
v = v.to(torch.float16)
v = v[:1,:,:,:]
out = torch.load("/home/xieruiqi/diffuser-dev-225exp/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/attn_out_test.pth").cuda()
print(out.dtype) # torch.Size([2, 48, 17776, 64]) torch.bfloat16
out = out[:1,:,:,:]


batch = q.shape[0]
head = q.shape[1]
seq_len = q.shape[2]
headdim = q.shape[3]
sm_scale = 1 / (headdim ** 0.5)
tensor_layout = "HND"
is_causal = False
_is_causal = 1 if is_causal else 0
flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)
sparse_ratio = torch.sum(sparse).item() / (batch*head*seq_len*seq_len/CTA_Q/CTA_Q)

q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, BLKQ=64, WARPQ=32, BLKK=64, tensor_layout=tensor_layout)
# q_int8, q_scale, k_int8, k_scale = per_block_int8_cuda(q, k, BLKQ=64, BLKK=64, tensor_layout=tensor_layout)
o_int8 = torch.empty(batch, head, seq_len, headdim, dtype=torch.float16).cuda()
dense_o_int8 = torch.empty(batch, head, seq_len, headdim, dtype=torch.float16).cuda()

for i in range(5): sdpa(q.to(torch.float16), k.to(torch.float16), v, is_causal=is_causal)
torch.cuda.synchronize()
_, time_fa = benchmark_forward(sdpa, q.to(torch.float16), k.to(torch.float16), v, is_causal=is_causal, repeats=100, verbose=False, desc='Triton')
print(f'FA2: latency:{time_fa.mean*1e3}, flops: {flops/time_fa.mean*1e-12}')

for i in range(5): kernel_sage(q_int8, k_int8, v, o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0)
torch.cuda.synchronize()
_, time_sage = benchmark_forward(kernel_sage, q_int8, k_int8, v, o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
print(f'Sage: latency:{time_sage.mean*1e3}, flops: {flops/time_sage.mean*1e-12}')

for i in range(5): kernel_int8(q_int8, k_int8, v, dense_o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, all_one_sparse)
torch.cuda.synchronize()
_, time_all_dense_int8 = benchmark_forward(kernel_int8, q_int8, k_int8, v, dense_o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, all_one_sparse, repeats=100, verbose=False, desc='Triton')
print(f'PARO: sparse ratio:1, latency:{time_all_dense_int8.mean*1e3}, flops: {flops/time_all_dense_int8.mean*1e-12}')      

for i in range(5): kernel_int8(q_int8, k_int8, v, o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, sparse)
torch.cuda.synchronize()
_, time_int8 = benchmark_forward(kernel_int8, q_int8, k_int8, v, o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, sparse, repeats=100, verbose=False, desc='Triton')
print(f'PARO: sparse ratio:{sparse_ratio}, latency:{time_int8.mean*1e3}, flops: {flops/time_int8.mean*1e-12}')      

diff = torch.abs(o_int8 - out)
mean_diff = torch.mean(diff)
print(f"Mean difference: {mean_diff.item()}")
cos_sim = torch.cosine_similarity(o_int8[:, :, :, :], out[:, :, :, :], dim=3)  #[1, 48, 17762]
print(torch.mean(cos_sim))