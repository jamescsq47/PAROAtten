import torch
from diffusers.callbacks import PipelineCallback

class TimestepCallback(PipelineCallback):
    """
    Callback to update the timestep for the sparse mask indexing inside CustomizedAttnProcessor.
    """
    def __init__(self, num_timestep_for_sparse_mask=10):
        super().__init__()
        self.timestep_history = []
        self.num_timestep_for_sparse_mask = num_timestep_for_sparse_mask
        
    def on_step_end(self, pipeline, step, timestep, callback_kwargs):
        # 记录timestep信息
        self.timestep_history.append({
            'step': step,
            'timestep': timestep.item() if torch.is_tensor(timestep) else timestep,
            'latents': callback_kwargs.get('latents', None)
        })
        
        total_steps = len(pipeline.scheduler.timesteps)
        step = step % total_steps  
        normalized_timestep = (step+1) / total_steps  # INFO: get the next step. 
        self.sparse_mask_timestep = int(normalized_timestep * self.num_timestep_for_sparse_mask)
        # print(f'at step:{step} T={normalized_timestep:.3f}, mapped to sparse_mask_timestep={self.sparse_mask_timestep}') 
        pipeline.transformer.i_timestep = self.sparse_mask_timestep
        for i_block in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i_block].attn1.processor.i_timestep = self.sparse_mask_timestep
            
        return callback_kwargs
    
class SparseMaskPrefetchStream:
    def __init__(self, sparse_mask):
        """
        prefetch the sparse mask 
        """
        self.sparse_mask = sparse_mask  # the sparse mask on CPU, should be in shape [num_timeste, mask_size]
        self.n_timestep, self.n_transformer_block, self.n_head, self.mask_len, _= self.sparse_mask.shape
        self.num_iterations = self.n_timestep * self.n_transformer_block
        self.sparse_mask = self.sparse_mask.reshape([self.num_iterations, self.n_head, self.mask_len, self.mask_len])
        self.i_iter = 0
        self.batch_size = 2  # 2 is the CFG BS
        
        # create the double buffer on GPU.
        self.double_buffer = [
            torch.empty([self.n_head, self.mask_len, self.mask_len], dtype=torch.bool, device='cuda'),
            torch.empty([self.n_head, self.mask_len, self.mask_len], dtype=torch.bool, device='cuda'), 
        ]
        
        # create the prefetch stream.
        self.stream = torch.cuda.Stream()
        # init the first mask to buffer[0]
        self.double_buffer[0].copy_(self.sparse_mask[0], non_blocking=False)
        
    def reset(self):
        # the n_iter should be reset after each pipeline run.
        self.i_iter = 0

    # def get_sparse_mask(self):
    #     """
    #     get the sparse mask and prefetch the next mask.
    #     """
    #     load_buf_id = self.i_iter % 2
    #     write_buf_id = (self.i_iter + 1) % 2

    #     if self.i_iter < self.num_iterations - 1:
    #         with torch.cuda.stream(self.stream):
    #             self.double_buffer[write_buf_id].copy_(
    #                 self.sparse_mask[self.i_iter + 1], non_blocking=True
    #             )

    #     # 等待 prefetch 完成（也可考虑延迟 sync）
    #     self.stream.synchronize()
    #     self.i_iter += 1

    #     return self.double_buffer[load_buf_id]

