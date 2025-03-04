import torch
from flash_attn.utils.benchmark import benchmark_forward

import sageattention._qattn_sm80 as qattn

import argparse

def compact(x,y):
    return ((x <<4) | (y&0x0F))
def process_tensor(tensor):
    tensor_flat = tensor.view(-1)
    result = torch.empty(tensor_flat.size(0) // 2, dtype=torch.int8, device=tensor.device)    
    result[:] = compact(tensor_flat[0::2], tensor_flat[1::2])
    result = result.view(tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)//2)
    zero = torch.zeros(tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)//2, dtype=torch.int8, device=tensor.device)
    # result=torch.cat((result,zero),dim=3)
    result=torch.cat((result,result),dim=3)
    return result


parser = argparse.ArgumentParser(description='Benchmark QK Int8 PV FP16')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--quant_gran', type=str, default='per_warp', choices=['per_warp', 'per_thread'], help='Quantization granularity')
parser.add_argument('--pv_accum_dtype', type=str, default='fp16', choices=['fp16', 'fp16+fp32', 'fp32'])
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim

print(f"CUDA QK Int4 PV FP16 Benchmark")
print(f"batch: {batch}, head: {head}, headdim: {headdim}, pv_accum_dtype: {args.pv_accum_dtype}")

WARP_Q = 16 if (headdim == 128 and args.pv_accum_dtype == "fp16+fp32") else 32
WARP_K = 64

if args.pv_accum_dtype == 'fp32':
    kernel_int4 = qattn.qk_int4_sv_f16_accum_f32_attn # the kernel with fully fp32 accumulator
elif args.pv_accum_dtype == 'fp16+fp32':
    kernel_int4 = qattn.qk_int4_sv_f16_accum_f16_attn_inst_buf if headdim == 64 else qattn.qk_int4_sv_f16_accum_f16_attn_buf # the kernel with fp32 longterm buffer and fp16 shortterm accumulator
elif args.pv_accum_dtype == 'fp16':
    kernel_int4 = qattn.qk_int4_sv_f16_accum_f16_attn # the kernel with fully fp16 accumulator

if args.pv_accum_dtype == 'fp32':
    kernel_int8 = qattn.qk_int8_sv_f16_accum_f32_attn # the kernel with fully fp32 accumulator
elif args.pv_accum_dtype == 'fp16+fp32':
    kernel_int8 = qattn.qk_int8_sv_f16_accum_f16_attn_inst_buf if headdim == 64 else qattn.qk_int8_sv_f16_accum_f16_attn_buf # the kernel with fp32 longterm buffer and fp16 shortterm accumulator
elif args.pv_accum_dtype == 'fp16':
    kernel_int8 = qattn.qk_int8_sv_int8_accum_f16_attn # the kernel with fully fp16 accumulator

_qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2

for sparse_ratio in {0.3, 0.5}:
    # print(f"sparse_ratio: {sparse_ratio}")
    is_causal = False
    _is_causal = 1 if is_causal else 0
    # print(f"is_causal: {is_causal}")
    for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}: #1024, 2048, 4096, 8192, 16384, 32768
        sparse = torch.zeros((batch*head*seq_len*seq_len//4096), dtype=bool).cuda()
        random_tensor = torch.rand(sparse.shape).cuda()
        sparse[random_tensor < sparse_ratio] = True
        flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)
        q = torch.randint(-7, 8,(batch, seq_len, head, headdim), dtype=torch.int8).cuda()
        k = torch.randint(-7, 8,(batch, seq_len, head, headdim), dtype=torch.int8).cuda()
        q_compact = process_tensor(q).cuda()
        k_compact = process_tensor(k).cuda()

        vm = torch.randn(batch, head, headdim, dtype=torch.float16).cuda()

        if args.quant_gran == 'per_warp':
            q_scale = torch.randn(batch, head, seq_len // WARP_Q, dtype=torch.float).cuda()
            k_scale = torch.randn(batch, head, seq_len // WARP_K, dtype=torch.float).cuda()
        elif args.quant_gran == 'per_thread':
            q_scale = torch.randn(batch, head, seq_len // WARP_Q * 8, dtype=torch.float).cuda()
            k_scale = torch.randn(batch, head, seq_len // WARP_K * 4, dtype=torch.float).cuda()
        
        # v = torch.randn(batch, seq_len, head, headdim, dtype=torch.float16).cuda()
        v = torch.randint(-7, 8,(batch, seq_len, head, headdim), dtype=torch.int8).cuda()
        o_int4 = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16).cuda()
        o_int8 = torch.empty(batch, seq_len, head, headdim, dtype=torch.float16).cuda()
        sm_scale = 1 / (headdim ** 0.5)
        # for i in range(5): kernel_int8(q, k, v, o_int8, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0,1000)
        # torch.cuda.synchronize()
        # _, time_int8 = benchmark_forward(kernel_int8, q, k, v, o_int8, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0,1000, repeats=100, verbose=False, desc='Triton')
        for i in range(5): kernel_int8(q_compact, k_compact, v, o_int4, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0,sparse)
        torch.cuda.synchronize()
        _, time_int8 = benchmark_forward(kernel_int8, q_compact, k_compact, v, o_int4, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0, sparse,repeats=100, verbose=False, desc='Triton')
        # print(o_int4)
        # print(o_int8)
        print(f'sparse ratio: {sparse_ratio}, latency:{time_int8.mean*1e3}')
        # print(f'latency:{time_int4.mean}')
        # print(f'int4 {seq_len} flops:{flops/time_int8.mean*1e-12}')