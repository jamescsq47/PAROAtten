import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from flash_attn.utils.benchmark import benchmark_forward
from spas_sage_attn.utils import hyperparameter_check, get_block_map_meansim, get_block_map_meansim_fuse_quant
import spas_sage_attn._qattn as qattn_sparge
import paroattention._qattn_sm80 as qattn
import sageattention._qattn_sm80 as qattn_sage
import argparse
from quant import per_block_int8 as per_block_int8_cuda
from quant import per_warp_int8 as per_warp_int8_cuda
kernel_sage = qattn_sage.qk_int8_sv_f16_accum_f16_attn
kernel_sparge = qattn_sparge.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold

parser = argparse.ArgumentParser(description='Benchmark QK INT8 PV FP16')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--num_heads', type=int, default=48, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=64, help='Head dimension')
parser.add_argument('--quant_gran', type=str, default='per_warp', choices=['per_warp', 'per_thread'], help='Quantization granularity')
parser.add_argument('--pv_accum_dtype', type=str, default='fp16', choices=['fp16', 'int8'])
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim

print(f"PAROAttention QK Int8 PV FP16 Benchmark with Varying Sparsity.")
print(f"batch: {batch}, head: {head}, headdim: {headdim}, pv_accum_dtype: {args.pv_accum_dtype}")

CTA_Q = 64
CTA_K = 64
WARP_Q = 32
WARP_K = 64

if args.pv_accum_dtype == 'fp16':
    kernel_int4 = qattn.qk_int4_sv_f16_accum_f16_attn
elif args.pv_accum_dtype == 'int8':
    kernel_int4 = qattn.qk_int4_sv_int8_accum_f16_attn

if args.pv_accum_dtype == 'fp16':
    kernel_int8 = qattn.qk_int8_sv_f16_accum_f16_attn
elif args.pv_accum_dtype == 'int8':
    kernel_int8 = qattn.qk_int8_sv_int8_accum_f16_attn

_qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2


for sparse_ratio in {0.0,0.5,1}:
    is_causal = False
    _is_causal = 1 if is_causal else 0
    for seq_len in {17776}: #1024, 2048, 4096, 8192, 16384, 32768
        sparse = torch.zeros((batch,head,((CTA_K-1+seq_len)//CTA_K),((CTA_Q-1+seq_len)//CTA_Q)), dtype=bool).cuda()
        sparse[:,:,:,-1] = True  
        random_tensor = torch.rand(sparse.shape).cuda()
        sparse[random_tensor < sparse_ratio] = True
        print(f"sparse ratio: {sparse.sum()/sparse.numel()}")
        flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)
        q = torch.randint(-10, 10,(batch, head, seq_len, headdim), dtype=torch.int8).cuda()
        k = torch.randint(-10, 10,(batch, head, seq_len, headdim), dtype=torch.int8).cuda()

        vm = torch.randn(batch, head, headdim, dtype=torch.float16).cuda()

        if args.quant_gran == 'per_warp':
            q_scale = torch.randn(batch, head, (seq_len+CTA_Q-1) // CTA_Q * (CTA_Q // WARP_Q), dtype=torch.float).cuda()
            k_scale = torch.randn(batch, head, (seq_len+CTA_K-1) // CTA_K * (CTA_K // WARP_K), dtype=torch.float).cuda()
        elif args.quant_gran == 'per_thread':
            q_scale = torch.randn(batch, head, (seq_len+WARP_Q-1) // WARP_Q * 8, dtype=torch.float).cuda()
            k_scale = torch.randn(batch, head, (seq_len+WARP_K-1) // WARP_K * 4, dtype=torch.float).cuda()
        
        v = torch.randint(-10, 10,(batch, head, seq_len, headdim), dtype=torch.int8).cuda()
        o_int8 = torch.empty(batch, head, seq_len, headdim, dtype=torch.bfloat16).cuda()
        sm_scale = 1 / (headdim ** 0.5)
        if(args.pv_accum_dtype=='fp16'):
            for i in range(5): kernel_int8(q, k, v.to(torch.float16), o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, sparse)
            torch.cuda.synchronize()
            _, time_int8 = benchmark_forward(kernel_int8, q, k, v.to(torch.float16), o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, sparse,repeats=100, verbose=False, desc='Triton')
        elif(args.pv_accum_dtype=='int8'):
            for i in range(5): kernel_int8(q, k, v, o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0,sparse)
            torch.cuda.synchronize()
            _, time_int8 = benchmark_forward(kernel_int8, q, k, v, o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, sparse,repeats=100, verbose=False, desc='Triton')
        for i in range(5): sdpa(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), is_causal=is_causal)
        torch.cuda.synchronize()
        _, time_fa = benchmark_forward(sdpa, q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), is_causal=is_causal, repeats=100, verbose=False, desc='Triton')
        print(f'FA2: latency:{time_fa.mean*1e3}ms, flops: {flops/time_fa.mean*1e-12}TFLOPS/s')
        print(f'PARO: shape of input data: {q.shape}, sparse ratio: {sparse_ratio}, latency:{time_int8.mean*1e3}ms, flops: {flops/time_int8.mean*1e-12}TFLOPS/s, speed-up ratio(compared to FA2): {time_fa.mean/time_int8.mean}')
        print(o_int8.isnan().sum())
        
