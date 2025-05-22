import torch
import diffusers
from diffusers import CogVideoXPipeline
sparse_plan = torch.load('/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/sparse_plan.pth', map_location = 'cuda')
permute_plan = torch.load('/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/permute_plan.pth', map_location = 'cuda')
sparse = torch.load("/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/kernel_sparse_plan.pth")
ckpt_path = "/home/models/models--THUDM--CogVideoX-5b/snapshots/8d6ea3f817438460b25595a120f109b88d5fdfad"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = CogVideoXPipeline.from_pretrained(
    ckpt_path,
    torch_dtype=torch.bfloat16
).to(device)

print(dir(pipe.transformer.transformer_blocks[0].attn1))
# print(sparse.shape)
# print(sparse_plan['sparse'].shape)
print(permute_plan['permute'].shape)