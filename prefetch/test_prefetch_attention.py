import torch
from flash_attn.utils.benchmark import benchmark_forward
import paroattention._qattn_sm80 as qattn
import argparse

class PipelinedQKAttentionRunner:
    def __init__(self, batch_size, num_heads, head_dim, quant_gran='per_warp', pv_accum_dtype='fp16'):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.quant_gran = quant_gran
        self.pv_accum_dtype = pv_accum_dtype
        
        # 设置CUDA流
        self.stream = torch.cuda.Stream()
        
        # 初始化kernel
        if pv_accum_dtype == 'fp16':
            self.kernel_int8 = qattn.qk_int8_sv_f16_accum_f16_attn
        elif pv_accum_dtype == 'int8':
            self.kernel_int8 = qattn.qk_int8_sv_int8_accum_f16_attn
            
        self._qk_quant_gran = 3 if quant_gran == 'per_thread' else 2
        
        # 设置warp大小
        self.WARP_Q = 16 if (head_dim == 128 and pv_accum_dtype == "fp16+fp32") else 32
        self.WARP_K = 64
        
        # 创建双缓冲
        self.buffers = [
            {
                'q': None,
                'k': None,
                'v': None,
                'q_scale': None,
                'k_scale': None,
                'o': None
            },
            {
                'q': None,
                'k': None,
                'v': None,
                'q_scale': None,
                'k_scale': None,
                'o': None
            }
        ]

    def prepare_buffers(self, seq_len):
        for buf in self.buffers:
            buf['q'] = torch.empty((self.batch_size, seq_len, self.num_heads, self.head_dim), 
                                 dtype=torch.int8, device='cuda')
            buf['k'] = torch.empty((self.batch_size, seq_len, self.num_heads, self.head_dim), 
                                 dtype=torch.int8, device='cuda')
            buf['v'] = torch.empty((self.batch_size, seq_len, self.num_heads, self.head_dim), 
                                 dtype=torch.int8, device='cuda')
            buf['o'] = torch.empty((self.batch_size, seq_len, self.num_heads, self.head_dim), 
                                 dtype=torch.float16, device='cuda')
            
            if self.quant_gran == 'per_warp':
                buf['q_scale'] = torch.empty((self.batch_size, self.num_heads, seq_len // self.WARP_Q), 
                                           dtype=torch.float, device='cuda')
                buf['k_scale'] = torch.empty((self.batch_size, self.num_heads, seq_len // self.WARP_K), 
                                           dtype=torch.float, device='cuda')
            else:  # per_thread
                buf['q_scale'] = torch.empty((self.batch_size, self.num_heads, seq_len // self.WARP_Q * 8), 
                                           dtype=torch.float, device='cuda')
                buf['k_scale'] = torch.empty((self.batch_size, self.num_heads, seq_len // self.WARP_K * 4), 
                                           dtype=torch.float, device='cuda')

    def run(self, seq_len, sparse_ratio, is_causal=False):
        # 准备稀疏掩码
        sparse = torch.zeros((self.batch_size * self.num_heads * seq_len * seq_len // 4096), 
                           dtype=bool, device='cuda')
        random_tensor = torch.rand(sparse.shape, device='cuda')
        sparse[random_tensor < sparse_ratio] = True
        
        # 准备输入数据
        q = torch.randint(-7, 8, (self.batch_size, seq_len, self.num_heads, self.head_dim), 
                         dtype=torch.int8, device='cuda')
        k = torch.randint(-7, 8, (self.batch_size, seq_len, self.num_heads, self.head_dim), 
                         dtype=torch.int8, device='cuda')
        v = torch.randint(-7, 8, (self.batch_size, seq_len, self.num_heads, self.head_dim), 
                         dtype=torch.int8, device='cuda')
        
        # 准备scale
        if self.quant_gran == 'per_warp':
            q_scale = torch.randn(self.batch_size, self.num_heads, seq_len // self.WARP_Q, 
                                dtype=torch.float, device='cuda')
            k_scale = torch.randn(self.batch_size, self.num_heads, seq_len // self.WARP_K, 
                                dtype=torch.float, device='cuda')
        else:  # per_thread
            q_scale = torch.randn(self.batch_size, self.num_heads, seq_len // self.WARP_Q * 8, 
                                dtype=torch.float, device='cuda')
            k_scale = torch.randn(self.batch_size, self.num_heads, seq_len // self.WARP_K * 4, 
                                dtype=torch.float, device='cuda')
        
        # 准备输出
        o = torch.empty(self.batch_size, seq_len, self.num_heads, self.head_dim, 
                       dtype=torch.float16, device='cuda')
        
        # 设置其他参数
        sm_scale = 1 / (self.head_dim ** 0.5)
        _is_causal = 1 if is_causal else 0
        
        # 使用pipeline prefetch进行计算
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        # 预热
        for _ in range(5):
            self.kernel_int8(q, k, v.to(torch.float16), o, q_scale, k_scale, 0, 
                           _is_causal, self._qk_quant_gran, sm_scale, 0, sparse)
        
        # 实际计时
        for _ in range(100):
            with torch.cuda.stream(self.stream):
                self.kernel_int8(q, k, v.to(torch.float16), o, q_scale, k_scale, 0, 
                               _is_causal, self._qk_quant_gran, sm_scale, 0, sparse)
            self.stream.synchronize()
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
        flops = 4 * self.num_heads * self.batch_size * self.head_dim * seq_len * seq_len / (2 if is_causal else 1)
        
        return elapsed_time, flops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark QK INT8 PV FP16 with Pipeline Prefetch')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--quant_gran', type=str, default='per_warp', 
                       choices=['per_warp', 'per_thread'], help='Quantization granularity')
    parser.add_argument('--pv_accum_dtype', type=str, default='fp16', 
                       choices=['fp16', 'int8'])
    args = parser.parse_args()
    
    runner = PipelinedQKAttentionRunner(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        quant_gran=args.quant_gran,
        pv_accum_dtype=args.pv_accum_dtype
    )
    
    print(f"PAROAttention QK Int8 PV INT8 Benchmark with Pipeline Prefetch")
    print(f"batch: {args.batch_size}, head: {args.num_heads}, headdim: {args.head_dim}, "
          f"pv_accum_dtype: {args.pv_accum_dtype}")
    
    for sparse_ratio in [0.2, 0.3, 0.5, 0.75, 1]:
        for seq_len in [17792]:
            elapsed_time, flops = runner.run(seq_len, sparse_ratio)
            print(f'seq len: {seq_len}, sparse ratio: {sparse_ratio}, '
                  f'latency: {elapsed_time*1e3:.2f}ms, '
                  f'flops: {flops/elapsed_time*1e-12:.2f} TFLOPS') 