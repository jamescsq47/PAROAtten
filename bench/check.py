from contextlib import redirect_stdout
import io
import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from flash_attn.utils.benchmark import benchmark_forward
import paroattention._qattn_sm80 as qattn
import sageattention._qattn_sm80 as qattn_sage
from spas_sage_attn.utils import hyperparameter_check, get_block_map_meansim, get_block_map_meansim_fuse_quant
import spas_sage_attn._qattn as qattn_sparge
import argparse
from quant import per_block_int8 as per_block_int8_cuda
from quant import per_warp_int8 as per_warp_int8_cuda
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
import matplotlib.pyplot as plt
from torch.nn.attention.flex_attention import flex_attention
from svg.models.cog.attention import prepare_flexattention
from svg.models.cog.utils import generate_temporal_head_mask_mod, create_block_mask_cached
from svg.models.cog.utils import get_attention_mask

parser = argparse.ArgumentParser(description='Benchmark QK INT8 PV FP16')
parser.add_argument('--quant_gran', type=str, default='per_warp', choices=['per_block', 'per_warp', 'per_thread'], help='Quantization granularity')
parser.add_argument('--q_path', type=str, default='/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/query_tensor.pth')
parser.add_argument('--k_path', type=str, default='/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/key_tensor.pth')
parser.add_argument('--v_path', type=str, default='/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/value_tensor.pth')
args = parser.parse_args()
print(f"PAROAttention QK Int8 PV fp16 Benchmark with Given Data of cogvideo")

CTA_Q = 64
CTA_K = 64
WARP_Q = 32 
WARP_K = 64

kernel_paro = qattn.qk_int8_sv_f16_accum_f16_attn
kernel_sage = qattn_sage.qk_int8_sv_f16_accum_f16_attn
kernel_sparge = qattn_sparge.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold
q_path = args.q_path
k_path = args.k_path
v_path = args.v_path
_qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2 # 'per_warp'
sparse_plan = torch.load('/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/sparse_plan.pth', map_location = 'cuda')
permute_plan = torch.load('/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/permute_plan.pth', map_location = 'cuda')
sparse = torch.load("/home/xieruiqi/diffuser-dev520/examples/cogvideo_attn/logs/calib_data/0.49_0.015_1/kernel_sparse_plan.pth")

class PAROAttentionMap(torch.nn.Module):
    """
    the PARO sparse processor for attention map: block_sparse + empty
    """
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.sparse_plan = sparse_plan
        self.permute_plan = permute_plan
        self.empty_head_processor = 'zero'
        self.online = False
        self.dense_rate_accumulator = []
        self.i_block = 1
        self.i_timestep = 0

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        '''
        get the empty heads and assign uniform attn values to them. 
        '''

        BS, head_per_split_num, N_token, N_token = x.shape
        N_text_token = 256
        N_image_token = N_token - N_text_token
        attn_map_image = x[:,:,N_text_token:,N_text_token:]
        
        # DEBUG_ONLY
        x_old = x.clone()
        attn_map_image_old = attn_map_image.clone()
        
        self.block_sparse_size = 64
        
        # INFO" get the sparse mask.
        self.N_block_sparse = N_image_token // self.block_sparse_size
        N_block_sparse = self.N_block_sparse
        N_mask_token = N_block_sparse*self.block_sparse_size  # when not divisible, could be smaller than N_image_token
        indices = torch.arange(self.split_range[0], self.split_range[1], device='cuda')
        
        attn_map_image_ = attn_map_image[:,:,:N_mask_token,:N_mask_token]
        if False:
            block_sparse_masks = self.get_sparse_mask(attn_map_image_)
        else:
            N_timestep_in_calib_data = self.sparse_plan['sparse'].shape[0]
            i_timestep_in_calib_data = int(self.i_timestep // (1/N_timestep_in_calib_data))
            block_sparse_masks = self.sparse_plan['sparse'][i_timestep_in_calib_data, self.i_block,indices].unsqueeze(0).repeat([BS,1,1,1])  # [BS, N_block, N_block]
            
        for i_ in indices:
            block_sparse_mask = block_sparse_masks[:,i_%head_per_split_num,:,:]
            # import ipdb; ipdb.set_trace()
            block_sparse_mask = block_sparse_mask.reshape([BS,self.N_block_sparse,1,self.N_block_sparse,1])
            
            attn_map_image_head = attn_map_image_[:,i_ % head_per_split_num,:,:].reshape([
                BS,
                self.N_block_sparse,
                self.block_sparse_size,
                self.N_block_sparse,
                self.block_sparse_size,    
            ])
            
            attn_map_image[:,i_%head_per_split_num,:N_mask_token,:N_mask_token] = (attn_map_image_head*block_sparse_mask).reshape([
                BS,N_mask_token,N_mask_token
            ])
        
        indices = torch.arange(self.split_range[0], self.split_range[1], device='cuda')
        if_head_empty = self.permute_plan['empty'][self.i_block][indices]
        empty_indices = torch.nonzero(if_head_empty).squeeze(-1)
        
        if len(empty_indices) > 0:
            # print('Block {}, has empty heads {}'.format(self.i_block, empty_indices.tolist()))
            empty_heads = attn_map_image.index_select(dim=1, index=empty_indices)
            if self.empty_head_processor == 'uniform':
                x[:,empty_indices,N_text_token:,N_text_token:] = empty_heads.mean(-1).unsqueeze(-1).repeat(1,1,1,N_image_token)
            elif self.empty_head_processor == 'zero':
                x[:,empty_indices,N_text_token:,N_text_token:].fill_(0.)
            else:
                raise NotImplementedError
        
        block_sparse_mask_with_empty = block_sparse_masks*(1-if_head_empty).reshape([1,head_per_split_num,1,1])
        dense_rate = (block_sparse_masks.sum() / block_sparse_masks.numel()).item()
        dense_rate_with_empty = (block_sparse_mask_with_empty.sum() / block_sparse_mask_with_empty.numel()).item()
        # print(f'dense rate:{dense_rate:.4f}, dense_rate_with_empty:{dense_rate_with_empty:.4f}')
        self.dense_rate_accumulator.append(dense_rate_with_empty)

        return x

def customize_scaled_dot_product_attention(q_path, k_path, v_path, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, head_split_num=12,\
        ) -> torch.Tensor:
        
        device = 'cuda'
        query = torch.load(q_path,map_location = 'cuda')
        key = torch.load(k_path,map_location = 'cuda')
        value = torch.load(v_path,map_location = 'cuda')
            
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(device)
        # if is_causal:
        #     assert attn_mask is None
        #     temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        #     attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        #     attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # if enable_gqa:
        #     key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        #     value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        assert L == S

        head_num = query.shape[1]
        head_split_num = 12
        head_per_split_num = head_num // head_split_num
        assert head_num % head_split_num == 0
        attn_output = torch.zeros_like(query)
        softmax_func = torch.softmax
        
        '''
        INFO: the qk permute (token-level) based on existing permuteing order.
        '''
        query_old = query.clone()
        key_old = key.clone()
        value_old = value.clone()
        query, key, value = permute_qk(query, key, value)
        
        # import ipdb; ipdb.set_trace()
        
        for i in range(head_split_num):
            slice_range = (slice(None), slice(i * head_per_split_num, (i + 1) * head_per_split_num), slice(None), slice(None))
            query_part = query[slice_range]
            key_part = key[slice_range]
            value_part = value[slice_range]
        
            '''Quantization: qkv'''   
            # INFO: if config.attn.skip_text_quant = True, quantize image_part only.
            n_text_tokens = 256
            n_token = query.shape[2]
            n_image_tokens = n_token - n_text_tokens
        
            query_part_ = query_part[:,:,:n_image_tokens,:]
            key_part_ = key_part[:,:,:n_image_tokens,:]
            value_part_ = value_part[:,:,:n_image_tokens,:]
                
            BS, head_per_split_num, N_token, N_dim = query_part_.shape

            q_quantizer = nn.Identity()
            k_quantizer = nn.Identity()
            v_quantizer = nn.Identity()
            pre_softmax_attn_map_quantizer = nn.Identity()
            attn_map_quantizer = nn.Identity()
            attn_map_sparse_processor = PAROAttentionMap()

            query_part[:,:,:N_token,:] = q_quantizer(query_part_.reshape([-1,N_dim])).reshape([BS, head_per_split_num, N_token, N_dim])
            key_part[:,:,:N_token,:] = k_quantizer(key_part_.reshape([-1,N_dim])).reshape([BS, head_per_split_num, N_token, N_dim])
            
            n_group = 1  # default

            value_part_ = value_part_.reshape([BS,head_per_split_num,n_group,N_token//n_group,N_dim]).permute([0,1,4,2,3]).reshape([-1,N_token//n_group])
            value_part[:,:,:N_token,:] = v_quantizer(
                    value_part_
                ).reshape([BS, head_per_split_num, N_dim, N_token]).permute([0,1,3,2])
            attn_map_pre_softmax_part = query_part @ key_part.transpose(-2, -1) * scale_factor
            attn_map_pre_softmax_part += attn_bias

            attn_map_sparse_processor.split_range = [i * head_per_split_num, (i + 1) * head_per_split_num]
            
            BS, head_per_split_num, N_token, N_token = attn_map_pre_softmax_part.shape

            attn_map_post_softmax_part = softmax_func(attn_map_pre_softmax_part, dim=-1)
            attn_map_post_softmax_part = torch.dropout(attn_map_post_softmax_part, dropout_p, train=True)

            attn_map_post_softmax_part = attn_map_sparse_processor(attn_map_post_softmax_part)
            
            attn_output[slice_range] = attn_map_post_softmax_part @ value_part
        
        # import ipdb; ipdb.set_trace()
        # attn_output = permute_attn_out(attn_output)
            
        """ the same as the following code, only split to lower the memory footprint
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        attn_score = attn_weight @ value
        """
        assert attn_output.dtype == torch.bfloat16
        return attn_output
    
def permute_qk(query, key, value):
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

    i_block = 1

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
    
def permute_attn_out(attn_out):
    n_text_tokens = 226
    
    BS, N_head, N_token, N_dim = attn_out.shape
    attn_out_image_part = attn_out[:,:,n_text_tokens:,:]
    
    N_image_token = N_token - n_text_tokens
    F = 13
    H = 30
    W = 45
    assert N_image_token == F*W*H
    
    i_block = 1

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

out = customize_scaled_dot_product_attention(q_path, k_path, v_path)
out = out[:1,:,:,:]
# out = torch.cat([out, out], dim=-1)

flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3

def sample_mse(query, key, value, attention_masks):

    assert len(attention_masks) == 2

    cfg, num_heads, seq_len, dim = query.size()
    num_sampled_rows = min(32, seq_len)
    sampled_rows = torch.randint(low=0, high=seq_len, size=(num_sampled_rows,))
    sampled_q = query[:, :, sampled_rows, :]
    sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)
    
        
    sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
    sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)  # (1, seq_len, dim)

    sampled_mses = torch.zeros(len(attention_masks), cfg, num_heads, device=query.device, dtype=query.dtype)
    
    # Only have Tri-diagonal and Striped

    for mask_idx, attn_mask in enumerate(attention_masks):
        sampled_attention_mask = attn_mask[sampled_rows, :]
        sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float('-inf'))
        sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
        sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
        mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
        sampled_mses[mask_idx] = mse

    return sampled_mses

def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len ** 2
    
    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements
      
    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size
    
    return width_frame

sparse = sparse[0,1,:,:,:] # torch.Size([48, 278, 278]) one: calculate; zero: skip
sparse = sparse.to(torch.bool).cuda() 
all_one_sparse = torch.ones((48, 278, 278), dtype=torch.bool).cuda() # no sparse for comparison
q = torch.load(q_path).cuda()# torch.Size([2, 48, 17776, 64]) torch.bfloat16
k = torch.load(k_path).cuda()# torch.Size([2, 48, 17776, 64]) torch.bfloat16
v = torch.load(v_path).cuda()# torch.Size([2, 48, 17776, 64]) torch.bfloat16
v = v.to(torch.float16)
q,k,v = permute_qk(q,k,v)
q = q[:1,:,:,:]
k = k[:1,:,:,:]
v = v[:1,:,:,:]
# # if head_dim = 128
# q = torch.cat([q, q], dim=-1)
# k = torch.cat([k, k], dim=-1)
# v = torch.cat([v, v], dim=-1)
# print(q.shape)

batch = q.shape[0]
head = q.shape[1]
seq_len = q.shape[2]
headdim = q.shape[3]
sm_scale = 1 / (headdim ** 0.5)
tensor_layout = "HND"
is_causal = False
_is_causal = 1 if is_causal else 0
flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)
sparse_ratio = torch.sum(sparse).item() / (batch*head*seq_len*seq_len/CTA_Q/CTA_Q)

for i in range(5):
    if(args.quant_gran == 'per_warp'):
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, BLKQ=64, WARPQ=32, BLKK=64, tensor_layout=tensor_layout)
    elif(args.quant_gran == 'per_block'):
        q_int8, q_scale, k_int8, k_scale = per_block_int8_cuda(q, k, BLKQ=64, BLKK=64, tensor_layout=tensor_layout)
_, time_paro_quant = benchmark_forward(per_block_int8_cuda, q, k, BLKQ=64, BLKK=64, tensor_layout=tensor_layout, repeats=100, verbose=False, desc='Triton')
kernel_out = torch.empty(batch, head, seq_len, headdim, dtype=torch.float16).cuda()
dense_kernel_out = torch.empty(batch, head, seq_len, headdim, dtype=torch.float16).cuda()
sage_out = torch.empty(batch, head, seq_len, headdim, dtype=torch.float16).cuda()


print("All of the sparse ratio means the ratio of calculated elements to all elements in the attention map")
print("The speed-up ratio is compared with FA2")

for i in range(5): sdpa(q.to(torch.float16), k.to(torch.float16), v, is_causal=is_causal)
torch.cuda.synchronize()
_, time_fa = benchmark_forward(sdpa, q.to(torch.float16), k.to(torch.float16), v, is_causal=is_causal, repeats=100, verbose=False, desc='Triton')
print(f'\nFA2: latency:{time_fa.mean*1e3}ms, flops: {flops/time_fa.mean*1e-12}TFLOPs/s, speed-up ratio: {time_fa.mean/time_fa.mean}x')

for i in range(5): kernel_sage(q_int8, k_int8, v, sage_out, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0)
torch.cuda.synchronize()
_, time_sage = benchmark_forward(kernel_sage, q_int8, k_int8, v, sage_out, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
print(f'Sage: latency:{time_sage.mean*1e3}ms, flops: {flops/time_sage.mean*1e-12}TFLOPs/s, speed-up ratio: {time_sage.mean/time_fa.mean}x\n')

simthreshd1 = 0
cdfthreshd = 0
for simthreshd1, cdfthreshd in {(0.3,0.85), (0.5,0.1), (0.6,0.77)}:#(0.6,0.55), (0.5,0.1), (0.62,0.66)
    for i in range(5): lut, valid_block_num, q_sparge, q_scale_sparge, k_sparge, k_scale_sparge = get_block_map_meansim_fuse_quant(q, k, None, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=False)  
    torch.cuda.synchronize()
    _, time_sparge_overhead = benchmark_forward(get_block_map_meansim_fuse_quant, q, k, None, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=False, repeats=100, verbose=False, desc='Triton')
    pvthreshd = 100000
    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    qk_sparsity = 0
    out_sparge = torch.empty_like(out).cuda()
    for i in range(5):kernel_sparge(q_sparge, k_sparge, v, out_sparge, lut, valid_block_num, pvthreshd, q_scale_sparge, k_scale_sparge, 1, _is_causal, 1, sm_scale, 0)
    torch.cuda.synchronize()
    _, time_sparge = benchmark_forward(kernel_sparge, q_sparge, k_sparge, v, out_sparge, lut, valid_block_num, pvthreshd, q_scale_sparge, k_scale_sparge, 1, _is_causal, 1, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
    if is_causal is False:
        qk_sparsity = (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
    else:
        qk_sparsity = (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
    print(f'sparge: qk sparse ratio:{qk_sparsity.item()}, latency: {time_sparge.mean*1e3}ms, flops:{flops/time_sparge.mean*1e-12}TFLOPs/s, speed-up ratio: {time_sparge.mean/time_fa.mean}x')
    print(f'overhead of sparge attention: {time_sparge_overhead.mean*1e3}, which equals to {time_sparge_overhead.mean/time_sparge.mean} of the GEMM time\n')

context_length = 226
num_frame = 50
frame_size = 351
cfg_size = 2
num_head = 48
seq_len = context_length + num_frame * frame_size
for sparsity in {0.2, 0.3, 0.5}:
    diag_width = multiplier = sparsity_to_width(sparsity, context_length, num_frame, frame_size)
    block_mask = prepare_flexattention(2, head, headdim, torch.bfloat16, "cuda", context_length, num_frame, frame_size, diag_width, multiplier)
    spatial_mask = get_attention_mask("spatial", context_length, num_frame, frame_size)
    temporal_mask = get_attention_mask("temporal", context_length, num_frame, frame_size)
    attention_masks = [spatial_mask, temporal_mask]
    for i in range(5): sample_mse(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attention_masks)
    torch.cuda.synchronize()
    _, time_svg_overhead = benchmark_forward(sample_mse, q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attention_masks, repeats=100, verbose=False, desc='Triton')
    with redirect_stdout(io.StringIO()):
        for i in range(5): flex_attention(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), block_mask=block_mask)
        torch.cuda.synchronize()
        _, time_flex = benchmark_forward(flex_attention, q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), block_mask=block_mask, repeats=100, verbose=False, desc='Triton')
    print(f'SVG: sparse ratio: {sparsity}, latency：{time_flex.mean*1e3} ms, flops: {flops/time_flex.mean*1e-12} TFLOPs/s')
    print(f'overhead of SVG: {time_svg_overhead.mean*1e3}ms, which equals to {time_svg_overhead.mean/time_flex.mean} of the GEMM time\n')

for i in range(5): kernel_paro(q_int8, k_int8, v, dense_kernel_out, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, all_one_sparse)
torch.cuda.synchronize()
_, time_all_dense_int8 = benchmark_forward(kernel_paro, q_int8, k_int8, v, dense_kernel_out, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, all_one_sparse, repeats=100, verbose=False, desc='Triton')
print(f'PARO: sparse ratio:1, latency:{time_all_dense_int8.mean*1e3}ms, flops: {flops/time_all_dense_int8.mean*1e-12}TFLOPs/s')      

for i in range(5): kernel_paro(q_int8, k_int8, v, kernel_out, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, sparse)
torch.cuda.synchronize()
_, time_paro = benchmark_forward(kernel_paro, q_int8, k_int8, v, kernel_out, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, sparse, repeats=100, verbose=False, desc='Triton')
print(f'PARO: sparse ratio:{sparse_ratio}, latency:{time_paro.mean*1e3}ms, flops: {flops/time_paro.mean*1e-12}TFLOPs/s')      
print(f'overhead of PARO attention: {time_paro_quant.mean*1e3}, which equals to {time_paro_quant.mean/time_paro.mean} of the GEMM time')

diff = torch.abs(kernel_out - out)
mean_diff = torch.mean(diff)
print(f"Mean difference: {mean_diff.item()}")
cos_sim = torch.cosine_similarity(kernel_out, out, dim=3)  #[1, 48, 17762]
<<<<<<< HEAD
print(f"Mean cosin similarity: {torch.mean(cos_sim)}\n")
# dense_kernel_out = sage_out

for sparse_ratio in {0.2, 0.3, 0.5}:
    sparse = torch.zeros((batch*head*((63+seq_len)//64)*((63+seq_len)//64)), dtype=bool).cuda()
    random_tensor = torch.rand(sparse.shape).cuda()
    sparse[random_tensor < sparse_ratio] = True
    for i in range(5): kernel_paro(q_int8, k_int8, v, kernel_out, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, sparse)
    torch.cuda.synchronize()
    _, time_paro = benchmark_forward(kernel_paro, q_int8, k_int8, v, kernel_out, q_scale, k_scale, 1, _is_causal, _qk_quant_gran, sm_scale, 0, sparse, repeats=100, verbose=False, desc='Triton')
    print(f'PARO: sparse ratio:{sparse_ratio}, latency:{time_paro.mean*1e3}ms, flops: {flops/time_paro.mean*1e-12}TFLOPs/s')      
    print("   *sparse ratio means the ratio of calculated elements to all elements in the attention map")
    print(f'overhead of PARO attention: {time_paro_quant.mean*1e3}, which equals to {time_paro_quant.mean/time_paro.mean} of the GEMM time\n')
=======
print(f"Mean cosin similarity: {torch.mean(cos_sim)}")
>>>>>>> fef02d9092a8f611c0bb2191a66d5525c9a927f8
