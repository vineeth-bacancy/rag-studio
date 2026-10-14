"""Proxy CPU latency benchmark: pure-PyTorch decoder layers with RANDOM weights and
Qwen3-0.6B tensor shapes (hidden 1024, 16 q heads / 8 kv heads, head_dim 128, MLP 3072,
SwiGLU, RMSNorm incl. q/k-norm, causal SDPA). No weights are downloaded.
We time LAYERS layers and scale linearly to 28 layers (embedding lookup and the
last-position yes/no logits are negligible next to the 28 blocks).
Usage: python3 proxy_bench.py <threads> <layers_measured>
"""
import sys
import time

import torch
import torch.nn.functional as F

threads = int(sys.argv[1])
LAYERS = int(sys.argv[2])
SCALE = 28 / LAYERS
torch.set_num_threads(threads)
torch.manual_seed(0)

H, NQ, NKV, HD, FF = 1024, 16, 8, 128, 3072


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        L = torch.nn.Linear
        self.q, self.k, self.v = L(H, NQ * HD, bias=False), L(H, NKV * HD, bias=False), L(H, NKV * HD, bias=False)
        self.o = L(NQ * HD, H, bias=False)
        self.gate, self.up, self.down = L(H, FF, bias=False), L(H, FF, bias=False), L(FF, H, bias=False)
        self.n1, self.n2 = torch.nn.RMSNorm(H), torch.nn.RMSNorm(H)
        self.qn, self.kn = torch.nn.RMSNorm(HD), torch.nn.RMSNorm(HD)

    def forward(self, x):
        b, s, _ = x.shape
        h = self.n1(x)
        q = self.qn(self.q(h).view(b, s, NQ, HD)).transpose(1, 2)
        k = self.kn(self.k(h).view(b, s, NKV, HD)).transpose(1, 2)
        v = self.v(h).view(b, s, NKV, HD).transpose(1, 2)
        k = k.repeat_interleave(NQ // NKV, dim=1)
        v = v.repeat_interleave(NQ // NKV, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, s, NQ * HD))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


model = torch.nn.Sequential(*[Block() for _ in range(LAYERS)]).eval()
print(f'threads={threads} layers_measured={LAYERS} (scaled x{SCALE:.0f} to 28) torch={torch.__version__}')


def run(batch, seq, reps=2):
    x = torch.randn(batch, seq, H)
    times = []
    with torch.inference_mode():
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    best = min(times) * SCALE
    print(f'  batch={batch:3d} seq={seq:4d} tokens={batch*seq:6d}  est_28L={best:6.2f}s '
          f'({batch*seq/best:6.0f} tok/s)  raw={[round(t, 3) for t in times]}')
    return best


print('Reranker proxy: 20 pairs x ~390 tokens (300 doc + ~15 query + 73 template)')
run(1, 390, reps=2)
run(20, 390, reps=2)
t4 = sum(run(4, 390, reps=1) for _ in range(5))
print(f'  20 pairs as 5 batches of 4: est_28L={t4:.2f}s total')
print('Embedding proxy: 16 chunks x 256 tokens (Qwen3-Embedding-0.6B has identical shapes)')
te = run(16, 256, reps=2)
print(f'  extrapolated 1190 chunks x 256 tok: {te/16*1190/60:.1f} min')
