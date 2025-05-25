import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import diffusers
import time
import shutil
import argparse
import logging
from functools import wraps

import diffusers
# -- INFO: replace the attn processor with paro ones, then import the pipeline. --
from paroattention import PARO_CogVideoXAttnProcessor2_0
diffusers.models.attention_processor.CogVideoXAttnProcessor2_0 = PARO_CogVideoXAttnProcessor2_0
# --- replace definition. ---
from paroattention.utils import TimestepCallback
# from paroattention.utils import LayerNormWithPermute, LayerNormWithInversePermute
# F.scaled_dot_product_attention = paroattn
# diffusers.apply_rotary_embedding = apply_rotary_embedding_with_permute
# -----------------------
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
<<<<<<< HEAD
# from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging

import numpy as np
def seed_everything(seed):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
=======
from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging
>>>>>>> fef02d9092a8f611c0bb2191a66d5525c9a927f8

'''
input: F,H,W: fixed as standard.
permute_plan:
    - empty_plan: [N_block(42), N_head(48)]
    - permute_plan: [N_block(42), N_head(48)]
sparse_plan:
    - sparse: [N_timestep(10), N_block(42), N_head(48), N_downsamplertd(278)]
        - 17550 img_token
        - 226   text_token + 30 -> (256//64=4)
        - (17550-30)/64 = 273.75 -> 274
        - 278 = 274 + 4
    - dense_rate: [N_timestep(10), N_block(42), N_head(48)]
---
PAROAttention
    - self.prefetch = False
    - self.sparse_mask. (in python wrapper., updated each iter.)
--- 
transformer_block.norm1 -> LayerNormWithPermute
    - self.FHW
    - self.permute_order (update each iter.)
transformer_block.norm2 -> LayerNormWithInversePermute
    - self.FHW
    - self.inv_permute_order
apply_rotary_embedding 
    - self.FHW
---
'''

def paroattn_convert(pipe):
    
    # load the util files and config.
    F = 13
    H = 30
    W = 45
<<<<<<< HEAD

    # sparse_plan = torch.load('/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/kernel_sparse_plan.pth', map_location='cpu')  # load on cpu to avoid large GPU memory cost.
    sparse_plan = torch.zeros((10, 42, 48, 278, 278)).cpu()  # [10, 42, 48, 278, 278] 
    numel = sparse_plan.numel()
    num_ones = int(numel * 0.3)
    indices = torch.randperm(numel)[:num_ones]
    sparse_plan.view(-1)[indices] = 1
    sparse_plan[..., -1] = 1
    sparse_ratio = sparse_plan.sum() / numel
    print(f"Sparse plan loaded with ratio: {sparse_ratio:.2f}")

    permute_plan = torch.load('/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/permute_plan.pth', map_location='cuda')
    
    # init the sparse_plan & permute plan.
    sparse_mask = sparse_plan  # [10, 42, 48, 278, 278] # torch.ones((10, 42, 48, 278, 278)).cpu() 
    empty_head = (~permute_plan['empty'].bool()).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cpu()  # [42, 48]
    # sparse_mask = sparse_mask * empty_head    # assign empty head to sparse mask
    permute_order = permute_plan['permute']  # [42, 48]

=======
    sparse_plan = torch.load('./sparse_plan.pth', map_location='cpu')  # load on cpu to avoid large GPU memory cost.
    permute_plan = torch.load('./permute_plan.pth', map_location='cuda')
    
    # init the sparse_plan & permute plan.
    sparse_mask = sparse_plan['sparse']  # [10, 42, 48, 278, 278]
    empty_head = (~permute_plan['empty'].bool()).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [42, 48]
    sparse_mask = sparse_mask * empty_head    # assign empty head to sparse mask
    permute_order = permute_plan['permute']  # [42, 48]
    
>>>>>>> fef02d9092a8f611c0bb2191a66d5525c9a927f8
    # replace the attention. init sparse mask. (on CPU.)
    # the sparse mask is loaded to each block through prefetch in PAROAttention class.
    # timestep and i_block needs to be passed to each block. (in main.py this file.)
    
    # INFO: the atteniton needs to have i_timestep & i_block(could be here.), how to feed in?
<<<<<<< HEAD
    # import ipdb; ipdb.set_trace()

    sparse_mask_gpu = sparse_mask.to(torch.bool).cuda()
    # init the double buffer to all blocks.
    
    for b in range(42):
        pipe.transformer.transformer_blocks[b].attn1.processor.sparse_mask_gpu = sparse_mask_gpu
        # init the double buffer to all blocks.

        # INFO: also init the i_timestep = 0, since the callback is on_step_end.
        pipe.transformer.transformer_blocks[b].attn1.processor.i_timestep = 0 

        pipe.transformer.transformer_blocks[b].attn1.processor.i_block = b
        # INFO: init the i_block for each block. 
        
        # replace the layernorm & rope, to add reoreder & inv reorder.
        # call the from_float() func, or just inherit (replace definition, no need for replacement here)
        pipe.transformer.transformer_blocks[b].attn1.processor.permute_plan = permute_plan

        # INFO: DIRTY, reparam a few weights for numerical stability.
        weight_rep_constant = 8.
        pipe.transformer.transformer_blocks[b].attn1.to_v.weight.div_(weight_rep_constant)
        pipe.transformer.transformer_blocks[b].attn1.to_v.bias.div_(weight_rep_constant)
        pipe.transformer.transformer_blocks[b].attn1.to_out[0].weight.mul_(weight_rep_constant)        
=======
    import ipdb; ipdb.set_trace()
    pipe.transformer.transformer_blocks[0].attn.sparse_mask_cpu = sparse_mask   
    # init the double buffer to all blocks.
    

    # INFO: also init the i_timestep = 0, since the callback is on_step_end.
    pipe.transformer.transformer_blocks[0].attn.i_timestep = 0 
    # INFO: init the i_block for each block. 
    
    # replace the layernorm & rope, to add reoreder & inv reorder.
    # call the from_float() func, or just inherit (replace definition, no need for replacement here)
    pipe.transformer.transformer_blocks[0].attn.norm1.permute_plan = permute_plan

>>>>>>> fef02d9092a8f611c0bb2191a66d5525c9a927f8
    pass
    


def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = args.ckpt if args.ckpt is not None else "/home/models/models--THUDM--CogVideoX-5b/snapshots/8d6ea3f817438460b25595a120f109b88d5fdfad"
    pipe = CogVideoXPipeline.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    paroattn_convert(pipe)

    # INFO: if memory intense
    # pipe.enable_model_cpu_offload()
    # pipe.vae.enable_tiling()

    # read the promts
    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    with open(prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.strip())

    for i, prompt in enumerate(prompts):
        # 创建TimestepCallback实例，传入num_timestep_for_sparse_mask参数
        timestep_callback = TimestepCallback(num_timestep_for_sparse_mask=10)
        
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=args.num_sampling_steps, # 50
            num_frames=49,
            guidance_scale=args.cfg_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            callback_on_step_end=timestep_callback.on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        ).frames[0]

        # total_time = 0.0
        # for b in range(42):
        #     stats = pipe.transformer.transformer_blocks[b].attn1.processor.get_time_stats()
        #     total_time += stats['total_ms']
        #     print(f"Block {b} attention时间统计:", stats)
        # print(f"Total attention time: {total_time:.2f} ms")

        
        print(f"Export video to output_{i}.mp4")
        save_path = os.path.join(args.log, "generated_videos")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        export_to_video(video, os.path.join(save_path, f"output_{i}_paro_sparse.mp4"), fps=8)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default='./log')
    parser.add_argument("--cfg-scale", type=float, default=4.0)
<<<<<<< HEAD
    parser.add_argument("--num-sampling-steps", type=int, default=30)
=======
    parser.add_argument("--num-sampling-steps", type=int, default=50)
>>>>>>> fef02d9092a8f611c0bb2191a66d5525c9a927f8
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--sage_attn", action="store_true")
    args = parser.parse_args()
    main(args)
