import torch
from flash_attn.utils.benchmark import benchmark_forward

import paroattention._qattn_sm80 as qattn
import argparse

# ---- parameters for attention computing ------
head = 48
batch = 2
headdim = 64
pv_accum_dtype='fp16'  # currently on
_qk_quant_gran = 2  # '2' per_

print(f"PAROAttention QK Int8 with Varying Sparsity.")
print(f"batch: {batch}, head: {head}, headdim: {headdim}, pv_accum_dtype: {pv_accum_dtype}")

# ---- parameters for kernel ------
WARP_Q = 16 if (headdim == 128 and pv_accum_dtype == "fp16+fp32") else 32
WARP_K = 64
seq_len = 17792
_is_causal = 0 
num_iterations = 50
num_profile_iterations = 10

# use the kernel from paroattention package.
paro_attn_sparse_kernel = qattn.qk_int8_sv_f16_accum_f16_attn



def run_attention_loop(q, k, v, o, q_scale, k_scale, tensor_layout, _is_causal, _qk_quant_gran, sm_scale, return_lse, sparse, num_iterations=5):
    
    for _ in range(num_iterations):
        paro_attn_sparse_kernel(
            q, k, v.to(torch.float16), 
            o, q_scale, k_scale, 0,  # tensor_layout
            _is_causal, _qk_quant_gran, sm_scale, 0,  # return_lse
            sparse)
    torch.cuda.synchronize()
    
    
def run_attention_with_prefetch(q, k, v, o, q_scale, k_scale, tensor_layout, _is_causal, _qk_quant_gran, sm_scale, return_lse, sparse, num_iterations=5):
    
    # sparse is on CPU and LARGE. of shape [num_iterations, mask_size]

    double_buffer = [
        torch.empty((batch*head*seq_len*seq_len//4096), dtype=bool, device='cuda'),
        torch.empty((batch*head*seq_len*seq_len//4096), dtype=bool, device='cuda')
    ]
    prefetch_stream = torch.cuda.Stream()
    
    for i_iter in range(num_iterations):
        buf_id = i_iter % 2
        
        # print('sparse_mask device:',sparse.device,'buffer device:',double_buffer[0].device)  # INFO: check that the sparse is on CPU.
        
        with torch.cuda.stream(prefetch_stream):
            double_buffer[buf_id].copy_(sparse[i_iter], non_blocking=True)
        prefetch_stream.synchronize()

        paro_attn_sparse_kernel(
            q, k, v.to(torch.float16), 
            o, q_scale, k_scale, 0,  # tensor_layout
            _is_causal, _qk_quant_gran, sm_scale, 0,  # return_lse
            double_buffer[buf_id])
        
    torch.cuda.synchronize()

time_baseline = None
flops = 4 * head * batch * headdim * seq_len * seq_len * num_iterations
q = torch.randint(-7, 8,(batch, seq_len, head, headdim), dtype=torch.int8).cuda()
k = torch.randint(-7, 8,(batch, seq_len, head, headdim), dtype=torch.int8).cuda()

q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float).cuda()
k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float).cuda()

v = torch.randint(-7, 8,(batch, seq_len, head, headdim), dtype=torch.int8).cuda()
o = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16).cuda()
sm_scale = 1 / (headdim ** 0.5)

for sparse_ratio in [0.75, 0.5, 0.3, 0.25, 0.2]:    
    if time_baseline is None:
        full_dense_sparse_mask = torch.ones((batch*head*seq_len*seq_len//4096), dtype=bool).cuda()
        _, time_baseline = benchmark_forward(
            run_attention_loop,
            q, k, v.to(torch.float16), o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, full_dense_sparse_mask, num_iterations=num_iterations,
            repeats=10, verbose=False)
    
    sparse = torch.zeros((num_iterations,batch*head*seq_len*seq_len//4096), dtype=bool)
    random_tensor = torch.rand(sparse.shape).cuda()
    sparse[random_tensor < sparse_ratio] = True
            
    # for i in range(5): 
    #     paro_attn_sparse_kernel(
    #         q, k, v.to(torch.float16), 
    #         o, q_scale, k_scale, 0,  # tensor_layout
    #         _is_causal, _qk_quant_gran, sm_scale, 0 ,  # return_lse
    #         sparse)
    # torch.cuda.synchronize()
    
    print('[Run with prefetch:]')
    _, time_paro = benchmark_forward(
        run_attention_with_prefetch,
        q, k, v.to(torch.float16), o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, sparse, num_iterations=num_iterations,
        repeats=num_profile_iterations, verbose=False)

    speedup = time_baseline.mean / time_paro.mean
    print(f'seq len: {seq_len}, sparse ratio: {sparse_ratio}, latency:{time_paro.mean*1e3:.4f}, flops: {flops/time_paro.mean*1e-12:.4f}, speedup: {speedup:.4f}')
    
    print('[Run with sparse mask on GPU:]')
    sparse_gpu = sparse.cuda()
    _, time_paro = benchmark_forward(
        run_attention_loop,
        q, k, v.to(torch.float16), o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, sparse_gpu[0], num_iterations=num_iterations,
        repeats=num_profile_iterations, verbose=False)
    
    speedup = time_baseline.mean / time_paro.mean
    print(f'seq len: {seq_len}, sparse ratio: {sparse_ratio}, latency:{time_paro.mean*1e3:.4f}, flops: {flops/time_paro.mean*1e-12:.4f}, speedup: {speedup:.4f}')
    

    
