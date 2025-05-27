import torch

import paroattention._rope as rope_op

Z = 4
D = 4096
H = 128
W = 64
HS = 128

Q = torch.empty((Z, H, W, D), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
K = torch.empty((Z, H, W, D), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

freq = torch.empty((HS), dtype=torch.float, device="cuda").normal_(mean=0., std=0.5)

WARM_UP = 25
REP = 100

for _ in range(WARM_UP):
    _ = rope_op.rope_permutation(Q, K, freq, False)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = rope_op.rope_permutation(Q, K, freq, False)
    end_event[i].record()
torch.cuda.synchronize()
no_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

for _ in range(WARM_UP):
    _ = rope_op.rope_permutation(Q, K, freq, True)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    _ = rope_op.rope_permutation(Q, K, freq, True)
    end_event[i].record()
torch.cuda.synchronize()
yes_dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

print("Rope (without and with permutation): %.4f ms - %.4f ms" % (torch.mean(no_dur).item(), torch.mean(yes_dur).item()))