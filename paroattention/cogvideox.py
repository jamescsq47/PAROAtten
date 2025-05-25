import inspect
import math
import logging
from typing import Optional
import omegaconf
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention
from paroattention.core import paroattn
import paroattention._qattn_sm80 as qattn
from .quant import per_block_int8 as per_block_int8_cuda
from .quant import per_warp_int8 as per_warp_int8_cuda
import sageattention._qattn_sm80 as qattn_sage
import torch.cuda as cuda

kernel_paro = qattn.qk_int8_sv_f16_accum_f16_attn
kernel_sage = qattn_sage.qk_int8_sv_f16_accum_f16_attn

def permute_qk(query, key, value, permute_plan, i_block):
    # (F,W,H) -> (Frame, With, Height) 
    # (17776-226) == 13*30*45
    n_text_tokens = 226

    BS, N_head, N_token, N_dim = query.shape
    query_image_part = query[:,:,n_text_tokens:,:]
    key_image_part = key[:,:,n_text_tokens:,:]
    value_image_part = value[:,:,n_text_tokens:,:]
    
    N_image_token = N_token - n_text_tokens
    F = 13
    H = 30
    W = 45
    assert N_image_token == F*W*H
    
    permutations = torch.tensor([
            [0, 1, 2],  # 0: FHW
            [0, 2, 1],  # 1: FWH
            [1, 2, 0],  # 2: HWF
            [1, 0, 2],  # 3: HFW
            [2, 1, 0],  # 4: WHF
            [2, 0, 1],  # 5: WFH
    ])

    permute_order_index = permute_plan['permute'][i_block]  # i_block is initialized during creating block in `transformer_3d.py`
    permute_orders = torch.stack([permutations[i.item()] for i in permute_order_index], dim=0)  # [N_head,3]
    
    for i_head in range(N_head):
        permute_dims_head = permute_orders[i_head]
        permute_dims_head_extend = tuple([0]+(permute_dims_head+1).tolist()+[4])
        
        query_image_part[:,i_head,:,:] = query_image_part[:,i_head,:,:].reshape([BS,F,H,W,N_dim]).permute(*permute_dims_head_extend).reshape([BS,N_image_token,N_dim])
        key_image_part[:,i_head,:,:] = key_image_part[:,i_head,:,:].reshape([BS,F,H,W,N_dim]).permute(*permute_dims_head_extend).reshape([BS,N_image_token,N_dim])
        value_image_part[:,i_head,:,:] = value_image_part[:,i_head,:,:].reshape([BS,F,H,W,N_dim]).permute(*permute_dims_head_extend).reshape([BS,N_image_token,N_dim])
    
    query[:,:,n_text_tokens:,:] = query_image_part
    key[:,:,n_text_tokens:,:] = key_image_part
    value[:,:,n_text_tokens:,:] = value_image_part
    
    return query, key, value
    
def permute_attn_out(attn_out, permute_plan, i_block):
    n_text_tokens = 226
    
    BS, N_head, N_token, N_dim = attn_out.shape
    attn_out_image_part = attn_out[:,:,n_text_tokens:,:]
    
    N_image_token = N_token - n_text_tokens
    F = 13
    H = 30
    W = 45
    assert N_image_token == F*W*H

    permute_order_index = permute_plan['permute'][i_block]  # i_block is initialized during creating block in `transformer_3d.py`
    permutations = torch.tensor([
        [0, 1, 2],  # 0: FHW
        [0, 2, 1],  # 1: FWH
        [1, 2, 0],  # 2: HWF
        [1, 0, 2],  # 3: HFW
        [2, 1, 0],  # 4: WHF
        [2, 0, 1],  # 5: WFH
    ])
    permutations_inv = torch.tensor([
        [0, 1, 2],  # 0: FHW
        [0, 2, 1],  # 1: FWH
        [2, 0, 1],  # 2: HWF
        [1, 0, 2],  # 3: HFW
        [2, 1, 0],  # 4: WHF
        [1, 2, 0],  # 5: WFH
    ])
    
    permute_orders = torch.stack([permutations[i.item()] for i in permute_order_index], dim=0)  # [N_head,3]
    permute_orders_inv = torch.stack([permutations_inv[i.item()] for i in permute_order_index], dim=0)  # [N_head,3]
    
    # indices = torch.zeros([N_head, N_image_token], device=self.device).long()
    for i_head in range(N_head):
        permute_dims_head = permute_orders[i_head]
        permute_dims_head_extend = tuple([0]+(permute_dims_head+1).tolist()+[4])
        permute_dims_head_inv = permute_orders_inv[i_head]
        permute_dims_head_inv_extend = tuple([0]+(permute_dims_head_inv+1).tolist()+[4])
                    
        permuted_shape = torch.tensor([BS,F,H,W,N_dim], device='cuda')[list(permute_dims_head_extend)]
        
        attn_out_image_part[:,i_head,:,:] = attn_out_image_part[:,i_head,:,:].reshape(*permuted_shape).permute(*permute_dims_head_inv_extend).reshape([BS,N_image_token,N_dim])
    
    attn_out[:,:,n_text_tokens:,:] = attn_out_image_part
            
    return attn_out

class PARO_CogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.events = {
            'total_start': cuda.Event(enable_timing=True),
            'total_end': cuda.Event(enable_timing=True),
        }
        self.time_accum = {
            'total': 0.0,
        }
        self.call_count = 0

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        sparse = self.sparse_mask_gpu
        permute_plan = self.permute_plan

        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)


        """
        INFO: the PAROAttention, replace the original scaled_dot_product_attention with PAROAttention.
        """
        
        # support prefetch and not.
        self.events['total_start'].record()

        hidden_states = torch.empty((2, 48, 17776, 64), device = 'cuda', dtype=torch.bfloat16)      
        # permute_qk(query, key, value, permute_plan, self.i_block)
        sm_scale = 1 / (head_dim ** 0.5)
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(query, key, BLKQ=64, WARPQ=32, BLKK=64, tensor_layout="HND")

        kernel_paro(q_int8, k_int8, value.to(torch.float16), hidden_states, q_scale, k_scale, 1, 0, 2, sm_scale, 0, torch.stack([sparse[self.i_timestep,self.i_block,:,:,:],sparse[self.i_timestep,self.i_block,:,:,:]],dim=0)) 

        # use the code under to replace paroattention for the profiling of the original scaled_dot_product_attention.

        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )

        self.events['total_end'].record()
        cuda.synchronize()
        total_time = self.events['total_start'].elapsed_time(self.events['total_end'])
        self.time_accum['total'] += total_time
        # permute_attn_out(hidden_states, permute_plan, self.i_block)


        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            import ipdb; ipdb.set_trace()


        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(torch.bfloat16)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) 

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )


        return hidden_states, encoder_hidden_states

    def get_time_stats(self):
        return {
            'total_ms': self.time_accum['total']
        }
