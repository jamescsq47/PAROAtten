import torch

class PipelinedAttentionRunner:
    def __init__(self, attn_tensor_cpu: torch.Tensor, device='cuda'):
        self.device = device
        self.cpu_tensor = attn_tensor_cpu
        self.N_layer, self.N_block, self.N_head, self.N, _ = attn_tensor_cpu.shape

        self.buffers = [
            torch.empty(self.N_head, self.N, self.N, device=self.device),
            torch.empty(self.N_head, self.N, self.N, device=self.device)
        ]
        self.stream = torch.cuda.Stream()

    def dummy_attention(self, attn_block):
        COMPUTE_MEMORY_RATE = 100   # to simulate the compute takes more time than loading 
        for _ in range(COMPUTE_MEMORY_RATE):
            Q = K = V = attn_block
            scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.N ** 0.5)
            weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    def run(self, mode='prefetch'):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        if mode == 'full_load':
            gpu_tensor = self.cpu_tensor.to(self.device, non_blocking=False)
            for layer in range(self.N_layer):
                for block in range(self.N_block):
                    _ = self.dummy_attention(gpu_tensor[layer, block])

        elif mode == 'prefetch':
            for layer in range(self.N_layer):
                for block in range(self.N_block):
                    buf_id = (layer * self.N_block + block) % 2
                    curr_buf = self.buffers[buf_id]
                    with torch.cuda.stream(self.stream):
                        curr_buf.copy_(self.cpu_tensor[layer, block], non_blocking=True)
                    self.stream.synchronize()
                    _ = self.dummy_attention(curr_buf)

        elif mode == 'no_prefetch':
            for layer in range(self.N_layer):
                for block in range(self.N_block):
                    buf_id = (layer * self.N_block + block) % 2
                    curr_buf = self.buffers[buf_id]
                    curr_buf.copy_(self.cpu_tensor[layer, block], non_blocking=False)
                    _ = self.dummy_attention(curr_buf)

        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / 1000.0  # convert ms to sec


if __name__ == "__main__":
    # warmup.
    for i_ in range(3):
        torch.manual_seed(0)

        # Parameters
        N_layer = 4
        N_block = 4
        N_head = 48
        N = 256

        # Create pinned memory tensor and transfer it to GPU for baseline
        attn_tensor_cpu = torch.randn(N_layer, N_block, N_head, N, N, pin_memory=True)
        runner = PipelinedAttentionRunner(attn_tensor_cpu)
        
        # === Baseline (pure compute, exclude tensor creation time) ===
        # Directly move tensor to GPU before the timing
        gpu_tensor = attn_tensor_cpu.to('cuda', non_blocking=True)  # Only move tensor, no timing included
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for layer in range(N_layer):
            for block in range(N_block):
                _ = runner.dummy_attention(gpu_tensor[layer, block])
        end_event.record()
        torch.cuda.synchronize()
        baseline_time = start_event.elapsed_time(end_event) / 1000.0  # in sec
        if i_ == 2:
            print(f"[baseline     ]: {baseline_time:.4f} sec")
        
        # === Other modes ===
        modes = ['no_prefetch', 'prefetch', 'full_load']
        times = {}
        for mode in modes:
            t = runner.run(mode=mode)
            times[mode] = t
            if i_ == 2:
                print(f"[{mode:<12}]: {t:.4f} sec,  {t/baseline_time:.4f} times with baseline")



