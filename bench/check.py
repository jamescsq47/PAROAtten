import torch
from flash_attn.utils.benchmark import benchmark_forward
import paroattention._qattn_sm80 as qattn
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
    # result=torch.cat((result,zero),dim=3)
    result=torch.cat((result,result),dim=3)
    return result


parser = argparse.ArgumentParser(description='Benchmark QK INT8 PV INT8')
parser.add_argument('--quant_gran', type=str, default='per_warp', choices=['per_block', 'per_warp', 'per_thread'], help='Quantization granularity')
parser.add_argument('--pv_accum_dtype', type=str, default='fp16+fp32', choices=['fp16', 'fp16+fp32', 'fp32'])
args = parser.parse_args()


print(f"PAROAttention QK Int8 PV fp16 Benchmark with Given Data")
print(f"pv_accum_dtype: {args.pv_accum_dtype}")

WARP_Q = 32 
WARP_K = 64

kernel_int8 = qattn.qk_int8_sv_f16_accum_f16_attn

_qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2 # 'per_warp'
# _qk_quant_gran = 1 if args.quant_gran == 'per_block' else _qk_quant_gran

sparse = torch.load("/home/xieruiqi/diffuser-dev/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/kernel_sparse_plan.pth")
# sparse = sparse_plan['sparse']
sparse = sparse[0,1,:,:,:]
sparse = sparse.to(torch.bool).cuda() 
print(sparse.shape) #torch.Size([48, 278, 278]) one: calculate; zero: skip
all_one_sparse = torch.ones((48, 278, 278), dtype=torch.bool).cuda()  # 全True初始化
# new_sparse[:, 4:278, 4:278] = sparse
# new_sparse = new_sparse.view(-1)
# print(new_sparse.shape) #torch.Size([48, 279, 279]) 

# # 创建子图，8 行 6 列，共 48 个子图
# fig, axes = plt.subplots(8, 6, figsize=(24, 32))  # 每个子图大小为 4x4，总大小为 24x32

# # 遍历每个 head（第 0 维的索引）
# for head_index in range(48):
#     ax = axes[head_index // 6, head_index % 6]  # 确定子图位置
#     slice_data = new_sparse[head_index].cpu().numpy()  # 提取切片并转为 NumPy 数组
#     ax.imshow(slice_data, cmap="gray", interpolation="nearest")  # 绘制切片
#     ax.set_title(f"Head {head_index}")  # 设置标题
#     ax.axis("off")  # 隐藏坐标轴

# # 调整子图布局
# plt.tight_layout()

# # 保存合并后的图片
# output_path = "new_sparse_all_heads.png"
# plt.savefig(output_path)
# print(f"Visualization saved to {output_path}")



# torch.Size([30, 42, 48, 274, 274]) 17536=64*274
q = torch.load("/home/xieruiqi/diffuser-dev/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/query_permute.pth")
print(q.dtype)# torch.Size([2, 48, 17776, 64]) torch.bfloat16
q = q[:1,:,:,:]
# new_q = torch.zeros((1, 48, 17806, 64), dtype=torch.bfloat16).cuda()
# new_q[:, :, 30:, :] = q
k = torch.load("/home/xieruiqi/diffuser-dev-225exp/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/key_permute.pth")
print(k.dtype)# torch.Size([2, 48, 17776, 64]) torch.bfloat16
k = k[:1,:,:,:]
# new_k = torch.zeros((1, 48, 17806, 64), dtype=torch.bfloat16).cuda()
# new_k[:, :, 30:, :] = k
v = torch.load("/home/xieruiqi/diffuser-dev-225exp/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/value_permute.pth")
print(v.dtype)# torch.Size([2, 48, 17776, 64]) torch.bfloat16
v = v.to(torch.float16)
v = v[:1,:,:,:]
# new_v = torch.zeros((1, 48, 17806, 64), dtype=torch.float16).cuda()
# new_v[:, :, 30:, :] = v
out = torch.load("/home/xieruiqi/diffuser-dev-225exp/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/attn_out_test.pth")
print(out.dtype) # torch.Size([2, 48, 17776, 64]) torch.bfloat16
out = out[:1,:,:,:]
# new_out = torch.zeros((1, 48, 17806, 64), dtype=torch.float16).cuda()
# new_out[:, :, 30:, :] = out

sm_scale = 1 / (64 ** 0.5)
tensor_layout = "HND"
q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, BLKQ=64, WARPQ=32, BLKK=64, tensor_layout=tensor_layout)
# q_int8, q_scale, k_int8, k_scale = per_block_int8_cuda(q, k, BLKQ=64, BLKK=64, tensor_layout=tensor_layout)

print(q_int8.shape) # torch.Size([2, 48, 17776, 64])
print(k_int8.shape) # torch.Size([2, 48, 17776, 64])
print(q_scale.shape) # torch.Size([2, 48, 278]) torch.Size([2, 48, 556])
print(k_scale.shape) # torch.Size([2, 48, 278]) torch.Size([2, 48, 278])
# print(q_int8.dtype) # torch.int8
# print(k_int8.dtype) # torch.int8
# print(q_scale.dtype) # torch.float32
# print(k_scale.dtype) # torch.float32
batch = q.shape[0]
seq_len = q.shape[2]
head = q.shape[1]
headdim = q.shape[3]
o_int8 = torch.empty(batch, head, seq_len, headdim, dtype=torch.float16).cuda()
dense_o_int8 = torch.empty(batch, head, seq_len, headdim, dtype=torch.float16).cuda()
sm_scale = 1 / (headdim ** 0.5)
is_causal = False
_is_causal = 1 if is_causal else 0
for i in range(2): 
    kernel_int8(q_int8, k_int8, v, o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, sparse)
    torch.cuda.synchronize()
for i in range(2): 
    kernel_int8(q_int8, k_int8, v, dense_o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, all_one_sparse)
    torch.cuda.synchronize()
# _, time_int8 = benchmark_forward(kernel_int8, q_int8, k_int8, v, o_int8, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, sparse, repeats=100, verbose=False, desc='Triton')
print(o_int8) # torch.Size([1, 48, 17776, 64])
print(dense_o_int8) # torch.Size([1, 48, 17776, 64])
diff = torch.abs(o_int8 - dense_o_int8)
mean_diff = torch.mean(diff)
print(f"Mean difference: {mean_diff.item()}")

# 提取第 12 个 head 的差值切片
# head_index = 12
# diff_head_12 = diff[0, head_index, :, :].cpu().numpy()  # 提取第 12 个 head 的数据并转为 NumPy 数组

# # 绘制差值的二维图像
# plt.figure(figsize=(12, 6))
# plt.imshow(diff_head_12, cmap="viridis", interpolation="nearest", aspect="auto")
# plt.title(f"Absolute Difference for Head {head_index}")
# plt.colorbar(label="Absolute Difference Value")
# plt.xlabel("Head Dimension (64)")
# plt.ylabel("Sequence Length (17776)")

# # 保存图片到当前目录
# output_path = f"absolute_difference_head_{head_index}.png"
# plt.savefig(output_path)
# print(f"Visualization saved to {output_path}")


# 在 batch 和 head 维度（第 0 和第 1 维）取最大值
# max_diff_across_batch_head = torch.max(diff, dim=0).values  # 先对 batch 维度取最大值
# max_diff_across_batch_head = torch.max(max_diff_across_batch_head, dim=0).values  # 再对 head 维度取最大值

# # max_diff_across_batch_head 的形状为 [17776, 64]
# max_diff_data = max_diff_across_batch_head.cpu().numpy()

# # 绘制二维图像
# plt.figure(figsize=(12, 6))
# plt.imshow(max_diff_data, cmap="viridis", interpolation="nearest", aspect="auto")
# plt.title("Maximum Difference Across Batch and Head Dimensions")
# plt.colorbar(label="Maximum Difference Value")
# plt.xlabel("Head Dimension (64)")
# plt.ylabel("Sequence Length (17776)")

# # 保存图片到当前目录
# output_path = "max_diff_across_batch_head.png"
# plt.savefig(output_path)
# print(f"Visualization saved to {output_path}")


print(torch.cosine_similarity(o_int8[:, :, :, :], dense_o_int8[:, :, :, :], dim=3))
print(torch.cosine_similarity(o_int8[:, :, :, :], dense_o_int8[:, :, :, :], dim=3).shape)
print(torch.cosine_similarity(o_int8[:, :, :, :], dense_o_int8[:, :, :, :], dim=3).mean())

# 假设 cosine_similarity 已经计算完成
cos_sim = torch.cosine_similarity(o_int8[:, :, :, :], dense_o_int8[:, :, :, :], dim=3)  # 结果大小为 [1, 48, 17762]
print(torch.mean(cos_sim))
# 去掉 batch 维度，形状变为 [48, 17762]
# cos_sim_data = cos_sim.squeeze(0).cpu().numpy()

# # 绘制二维图像
# plt.figure(figsize=(12, 6))
# plt.imshow(cos_sim_data, cmap="viridis", interpolation="nearest", aspect="auto")
# plt.title("Cosine Similarity Across Heads and Sequence Length")
# plt.colorbar(label="Cosine Similarity")
# plt.xlabel("Sequence Length (17762)")
# plt.ylabel("Head Index (48)")

# # 保存图片到当前目录
# output_path = "cosine_similarity_visualization.png"
# plt.savefig(output_path)
# print(f"Visualization saved to {output_path}")
# # print(f'seq len: {seq_len}, sparse ratio: {sparse_ratio}, flops: {flops*1e-12/time_int8.mean}, latency:{time_int8.mean*1e3}')
       

