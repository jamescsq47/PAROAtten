from diffusers.callbacks import PipelineCallback
import torch

class TimestepCallback(PipelineCallback):
    """
    Callback to update the timestep for the sparse mask.
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
        normalized_timestep = (step+1) / total_steps  # INFO: get the next step. 
        
        sparse_mask_timestep = int(normalized_timestep * self.num_timestep_for_sparse_mask)
        
        print(f'at T={normalized_timestep:.3f}, mapped to sparse_mask_timestep={sparse_mask_timestep}')
        
        pipeline.transformer.i_timestep = normalized_timestep
        for i_block in range(len(pipeline.transformer.transformer_blocks)):
            pipeline.transformer.transformer_blocks[i_block].attn1.processor.i_timestep = sparse_mask_timestep
            
        return callback_kwargs
