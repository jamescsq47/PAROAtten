import torch
import math
from torch.nn.attention.flex_attention import flex_attention
from .attention import prepare_flexattention


flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3

def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len ** 2
    
    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements
      
    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size
    
    return width_frame

def benchmark_flex_attention(
    B=2, H=48, N=16726, D=64,
    dtype=torch.float16, device='cuda',
    num_warmup=10, num_runs=20,
    sparsity=0.3
):
    # 随机生成 q, k, v
    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    diag_width = multiplier = sparsity_to_width(sparsity, 226, 11, 1500)
    block_mask = prepare_flexattention(2, 24, 128, torch.bfloat16, "cuda", 226, 11, 1500, diag_width, multiplier)
    print(f"block_mask 形状: {block_mask.shape}")
    # 1) 预热，排除首次编译/微调开销
    for _ in range(num_warmup):
        _ = flex_attention(q, k, v, block_mask=block_mask)

    # 2) 正式跑测
    torch.cuda.synchronize()
    times = []
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        start.record()
        out = flex_attention(q, k, v, block_mask=block_mask)
        end.record()

        # 等待结束
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_time = sum(times) / len(times)
    # 计算总计算量 (FLOPs)
    # 对于每个注意力头:
    # 1. QK矩阵乘法: B*H*N*N*D
    # 2. softmax: B*H*N*N
    # 3. 与V的矩阵乘法: B*H*N*N*D
    total_flops = B * H * N * N * D * 4
    # 转换为TFLOPs
    total_tflops = total_flops / 1e12
    # 计算吞吐量 (TFLOPs/s)
    throughput = total_tflops / (avg_time / 1000)  # 将ms转换为s
    print(f'FlexAttention full_attention 平均耗时：{avg_time:.2f} ms over {num_runs} runs')
    print(f'计算吞吐量：{throughput:.2f} TFLOPs/s')

if __name__ == '__main__':
    benchmark_flex_attention()
