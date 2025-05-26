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
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
# -- INFO: replace the attn processor with paro ones, then import the pipeline. --
from paroattention import PARO_WanTransformer3DModel
# Special for wan
diffusers.models.WanTransformer3DModel = PARO_WanTransformer3DModel
# --- replace definition. ---
from paroattention.prefetch_wan import TimestepCallback, SparseMaskPrefetchStream
# from paroattention.utils import LayerNormWithPermute, LayerNormWithInversePermute
# F.scaled_dot_product_attention = paroattn
# diffusers.apply_rotary_embedding = apply_rotary_embedding_with_permute
# -----------------------
from diffusers import WanPipeline
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
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
    # F = 13
    # H = 30
    # W = 45
    
    sparse_plan = torch.load("/home/xieruiqi/diffuser-dev520/examples/wan/logs/calib_data/720p/sparse_plan_expanded.pth", map_location='cpu', weights_only=True)
    permute_plan = torch.load("/home/xieruiqi/diffuser-dev520/examples/wan/logs/calib_data/720p/permute_plan.pth", map_location='cuda', weights_only=True)
    
    sparse_mask = sparse_plan['sparse'].bool()  # [10, 42, 48, 278, 278] # torch.ones((10, 42, 48, 278, 278)).cpu() 
    empty_head = (~permute_plan['empty'].bool()).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cpu()  # [42, 48]
    # sparse_mask = sparse_mask * empty_head    # assign empty head to sparse mask

    # sparse_mask = F.pad(sparse_mask, pad=(4, 0, 4, 0), mode='constant', value=1)  # for text embed.

    # if you want to profile the speed of a given sparse ratio, you can use the code under this comment and change the sparse_ratio value. But notice that the video generated will be meaningless.

    # sparse_plan = torch.zeros((10, 42, 48, 278, 278)).cpu()  # [10, 42, 48, 278, 278] 
    # sparse_ratio = 0.3
    # numel = sparse_plan.numel()
    # num_ones = int(numel * sparse_ratio)
    # indices = torch.randperm(numel)[:num_ones]
    # sparse_plan.view(-1)[indices] = 1
    # sparse_plan[..., -1] = 1
    # sparse_ratio = sparse_plan.sum() / numel
    # print(f"Sparse plan loaded with ratio: {sparse_ratio:.2f}")
    
    # sparse_plan = torch.load('/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/kernel_sparse_plan.pth', map_location='cpu')  # load on cpu to avoid large GPU memory cost.
    # permute_plan = torch.load('/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/permute_plan.pth', map_location='cuda')
    
    # init the sparse_plan & permute plan.

    # permute_order = permute_plan['permute']  # [42, 48]

    sparse_mask[:,:,:,:,-1] = True
    sparse_mask[:,:,:,-1,:] = True
    sparse_ratio = sparse_mask.sum() / sparse_mask.numel()
    print(f"Sparse plan loaded with ratio: {sparse_ratio:.2f}")

    # replace the attention. init sparse mask. (on CPU.)
    # the sparse mask is loaded to each block through prefetch in PAROAttention class.
    # timestep and i_block needs to be passed to each block. (in main.py this file.)
    
    # INFO: the atteniton needs to have i_timestep & i_block(could be here.), how to feed in?
    # import ipdb; ipdb.set_trace()

    if not args.prefetch:
        # move to GPU.
        sparse_mask = sparse_mask.cuda()
    else:
        prefetch_stream = SparseMaskPrefetchStream(sparse_mask)
        
    # init the double buffer to all blocks.
    for i_block in range(len(pipe.transformer.blocks)):
        if not args.prefetch:
            pipe.transformer.blocks[i_block].attn1.processor.sparse_mask = sparse_mask
        else:
            pipe.transformer.blocks[i_block].attn1.processor.prefetch_stream = prefetch_stream
            
        # init the double buffer to all blocks.

        # INFO: also init the i_timestep = 0, since the callback is on_step_end.
        pipe.transformer.blocks[i_block].attn1.processor.i_timestep = 0 

        pipe.transformer.blocks[i_block].attn1.processor.i_block = i_block
        # INFO: init the i_block for each block. 
        
        # replace the layernorm & rope, to add reoreder & inv reorder.
        # call the from_float() func, or just inherit (replace definition, no need for replacement here)
        pipe.transformer.blocks[i_block].attn1.processor.permute_plan = permute_plan

        # INFO: DIRTY, reparam a few weights for numerical stability.
        weight_rep_constant = 8.
        pipe.transformer.blocks[i_block].attn1.to_v.weight.div_(weight_rep_constant)
        pipe.transformer.blocks[i_block].attn1.to_v.bias.div_(weight_rep_constant)
        pipe.transformer.blocks[i_block].attn1.to_out[0].weight.mul_(weight_rep_constant)
    pass
    


def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = args.ckpt if args.ckpt is not None else "/home/models/Wan2.1-T2V-14B-Diffusers"
    model_id = ckpt_path
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)

    pipe = WanPipeline.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        vae=vae
    )
    pipe.scheduler = scheduler
    pipe = pipe.to(device)

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
        timestep_callback = TimestepCallback(num_timestep_for_sparse_mask=30)
        
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=args.num_sampling_steps, # 50
            num_frames=81,
            guidance_scale=args.cfg_scale,
            height=720,  # 720P
            width=1280,  # 720P
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            callback_on_step_end=timestep_callback.on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        ).frames[0]
        
        save_path = os.path.join(args.log, "generated_videos")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        outpath = os.path.join(save_path, f"output_{i}_paro_sparse.mp4")
        export_to_video(video, outpath, fps=8)
        print(f"Export video to {outpath}")
        
        total_time = 0.0
        for b in range(40):
            stats = pipe.transformer.blocks[b].attn1.processor.get_time_stats()
            total_time += stats['total_ms']
        print(f"Total attention time: {total_time:.2f} ms")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default='./log')
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--num-sampling-steps", type=int, default=30)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--prefetch", action="store_true")
    args = parser.parse_args()
    main(args)
