import torch
from quant import per_warp_int8 as per_warp_int8_cuda


rows = torch.arange(-256, 256, dtype=torch.bfloat16).unsqueeze(1)
q = rows.repeat(1, 64)
k = rows.repeat(1, 64)
q = q.unsqueeze(0).unsqueeze(0).cuda()  
k = k.unsqueeze(0).unsqueeze(0).cuda()
tensor_layout = "HND"
q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, BLKQ=64, WARPQ=32, BLKK=64, tensor_layout=tensor_layout)
print(f"q: {q}, k: {k}")
print(f"q_int8: {q_int8[0,0,:,0]}, q_scale: {q_scale}")
print(f"k_int8: {k_int8[0,0,:,0]}, k_scale: {k_scale}")