import torch
from flash_attn.utils.benchmark import benchmark_forward

import paroattention._qattn_sm80 as qattn
from paroattention.quant import per_warp_int8, per_block_int8
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
WARP_Q = 32
WARP_K = 64
seq_len = 17792
_is_causal = 0 
num_iterations = 40

num_profile_iterations = 3

# use the kernel from paroattention package.
paro_attn_sparse_kernel = qattn.qk_int8_sv_f16_accum_f16_attn

def run_attention_loop(q, k, v, o, q_scale, k_scale, tensor_layout, _is_causal, _qk_quant_gran, sm_scale, return_lse, sparse, num_iterations=5):
    
    for i_iter in range(num_iterations):
        paro_attn_sparse_kernel(
            q, k, v.to(torch.float16), 
            o, q_scale, k_scale, tensor_layout,  # tensor_layout
            _is_causal, _qk_quant_gran, sm_scale, 0,  # return_lse
            sparse[i_iter])
    torch.cuda.synchronize()
    
    
def run_attention_with_prefetch(q, k, v, o, q_scale, k_scale, tensor_layout, _is_causal, _qk_quant_gran, sm_scale, return_lse, sparse, num_iterations=5):
    
    # sparse is on CPU and LARGE. of shape [num_iterations, mask_size]
    # the mask granularity is 64.
    double_buffer = [
        torch.empty((batch*head*seq_len*seq_len//4096), dtype=bool, device='cuda'),
        torch.empty((batch*head*seq_len*seq_len//4096), dtype=bool, device='cuda')
    ]
    prefetch_stream = torch.cuda.Stream()
    
    # get the initial mask
    double_buffer[0].copy_(sparse[0], non_blocking=False)
    for i_iter in range(num_iterations):
        load_buf_id = (i_iter+1) % 2
        use_buf_id = i_iter % 2
        
        with torch.cuda.stream(prefetch_stream):
            if i_iter < num_iterations - 1:
                double_buffer[load_buf_id].copy_(sparse[i_iter+1], non_blocking=True)
        prefetch_stream.synchronize()

        paro_attn_sparse_kernel(
            q, k, v.to(torch.float16), 
            o, q_scale, k_scale, tensor_layout,  # tensor_layout
            _is_causal, _qk_quant_gran, sm_scale, 0,  # return_lse
            double_buffer[use_buf_id])
        
    torch.cuda.synchronize()

time_baseline = None
flops = 4 * head * batch * headdim * seq_len * seq_len * num_iterations
q = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16).cuda()
k = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16).cuda()
tensor_layout = "HND"
q_int8, q_scale, k_int8, k_scale = per_warp_int8(q, k, BLKQ=64, WARPQ=32, BLKK=64, tensor_layout=tensor_layout)
v = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16).cuda()
o = torch.empty(batch, head, seq_len, headdim, dtype=torch.float16).cuda()
sm_scale = 1 / (headdim ** 0.5)
tensor_layout_ = 1

for sparse_ratio in [0.75, 0.5, 0.3, 0.2]:  
    if time_baseline is None:
        full_dense_sparse_mask = torch.ones((batch*head*seq_len*seq_len//4096), dtype=bool).cuda()
        _, time_baseline = benchmark_forward(
            run_attention_loop,
            q_int8, k_int8, v, o, q_scale, k_scale, tensor_layout_, _is_causal, _qk_quant_gran, sm_scale, 0, full_dense_sparse_mask, num_iterations=num_iterations,
            repeats=10, verbose=False)
    
    
    USE_ACTUAL_SPARSE_MASK = False
    if USE_ACTUAL_SPARSE_MASK:
        sparse = torch.load("/home/xieruiqi/diffuser-dev/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/kernel_sparse_plan.pth")
        sparse = sparse[:,0:2,:,:,:].bool().reshape([10,-1])  # [0:2] to simluate batch size.
    else:
        mask_granularity = 64
        sparse = torch.zeros(
            (num_iterations,
            batch,head,
            seq_len//mask_granularity,
            seq_len//mask_granularity), dtype=bool)
        random_tensor = torch.rand(sparse.shape).cuda()
        sparse[random_tensor < sparse_ratio] = True
        # sparse[:,:,:,277,:] = True
        sparse[:,:,:,:,277] = True      # make the last column all True to avoid nan, when whole row is empty. I donot know why.
        sparse = sparse.reshape([num_iterations, batch*head*(seq_len//mask_granularity)*(seq_len//mask_granularity)])
                    
    # for i in range(5): 
    #     paro_attn_sparse_kernel(
    #         q, k, v.to(torch.float16), 
    #         o, q_scale, k_scale, 0,  # tensor_layout
    #         _is_causal, _qk_quant_gran, sm_scale, 0 ,  # return_lse
    #         sparse)
    # torch.cuda.synchronize()
    
    # -----------------------------------------------------------
    print('[Run with sparse mask on GPU:]')
    torch.cuda.synchronize()
    sparse_gpu = sparse.cuda()
    output, time_paro = benchmark_forward(
        run_attention_loop,
        q_int8, k_int8, v, o, q_scale, k_scale, tensor_layout_, _is_causal, _qk_quant_gran, sm_scale, 0, sparse_gpu, num_iterations=num_iterations,
        repeats=num_profile_iterations, verbose=False)
    
    o_wo_prefetch = o.clone()
    
    if o_wo_prefetch.isnan().any():
        print('o_wo_prefetch is nan')
        print(o_wo_prefetch.isnan().sum())
        import ipdb; ipdb.set_trace()
    
    speedup = time_baseline.mean / time_paro.mean
    print(f'seq len: {seq_len}, sparse ratio: {sparse_ratio}, latency:{time_paro.mean*1e3:.4f}, flops: {flops/time_paro.mean*1e-12:.4f}, speedup: {speedup:.4f}')
    
    # -----------------------------------------------------------
    print('[Run with prefetch:]')
    torch.cuda.synchronize()
    output, time_paro = benchmark_forward(
        run_attention_with_prefetch,
        q_int8, k_int8, v, o, q_scale, k_scale, tensor_layout_, _is_causal, _qk_quant_gran, sm_scale, 0, sparse, num_iterations=num_iterations,
        repeats=num_profile_iterations, verbose=False)
    
    o_with_prefetch = o.clone()
    
    speedup = time_baseline.mean / time_paro.mean
    print(f'seq len: {seq_len}, sparse ratio: {sparse_ratio}, latency:{time_paro.mean*1e3:.4f}, flops: {flops/time_paro.mean*1e-12:.4f}, speedup: {speedup:.4f}')
    
    
    # check for correctness 
    is_close = torch.allclose(o_with_prefetch, o_wo_prefetch, rtol=1e-3, atol=1e-3)
    print(f"Results are {'close' if is_close else 'different'}")
    if not is_close:
        # 如果结果不接近，打印更详细的差异信息
        print(o_with_prefetch - o_wo_prefetch)
        max_diff = (o_with_prefetch - o_wo_prefetch).abs().max().item()
        mean_diff = (o_with_prefetch - o_wo_prefetch).abs().mean().item()
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Relative difference: {max_diff / (o_with_prefetch.abs().max().item() + 1e-6):.6f}")
    

    
