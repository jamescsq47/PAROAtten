import torch
from inficom import rope_permutation

Z = 4
D = 4096
H = 128
W = 128
HS = 128

Q = torch.empty((Z, H, W, D), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
K = torch.empty((Z, H, W, D), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

freq = torch.empty((HS), dtype=torch.float, device="cuda").normal_(mean=0., std=0.5)

rope_permutation(Q, K, freq)

WARM_UP = 25
REP = 100

for _ in range(WARM_UP):
    rope_permutation(Q, K, freq)

start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
for i in range(REP):
    start_event[i].record()
    rope_permutation(Q, K, freq)
    end_event[i].record()
torch.cuda.synchronize()
dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

print(torch.mean(dur).item())