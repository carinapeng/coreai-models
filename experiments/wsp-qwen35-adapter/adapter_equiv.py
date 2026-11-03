"""Is the published edit exactly a rank-1 adapter on two down_proj layers?  Compare logits."""

import glob
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


H = os.path.expanduser("~/.cache/huggingface/hub")
base_dir = glob.glob(f"{H}/models--Qwen--Qwen3.5-0.8B/snapshots/*")[0]
edit_dir = glob.glob(
    f"{H}/models--RikiyaT--Qwen3.5-0.8B-weight-space-programming/snapshots/*"
)[0]
print("transformers", transformers.__version__)
tok = AutoTokenizer.from_pretrained(base_dir)
dtype = torch.float32
base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=dtype).eval()
edited = AutoModelForCausalLM.from_pretrained(edit_dir, dtype=dtype).eval()
edit = torch.load(HERE + "/rank1_edit.pt")


def find_layers(m):
    for name, mod in m.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList):
            return mod


layers = find_layers(base)


class Rank1Adapter(torch.nn.Module):
    """down_proj(h) + u * (c . h): the published edit as a rank-1 adapter."""

    def __init__(self, inner, u, c):
        super().__init__()
        self.inner = inner
        self.register_buffer("u", u.to(dtype))
        self.register_buffer("c", c.to(dtype))
        self.scale = 1.0

    def forward(self, h):
        return self.inner(h) + self.scale * (h @ self.c)[..., None] * self.u


adapters = {}
for L, e in edit.items():
    c = torch.zeros(layers[L].mlp.down_proj.in_features)
    c[e["cols"]] = e["coef"]
    adapters[L] = Rank1Adapter(layers[L].mlp.down_proj, e["u"], c)
    layers[L].mlp.down_proj = adapters[L]

prompts = [
    "Solve: what is 17 * 23? Think step by step.",
    "Find all real x such that x^2 - 5x + 6 = 0.",
    "Write a short poem about the ocean.",
]
rows = []
with torch.no_grad():
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        for a in adapters.values():
            a.scale = 1.0
        la = base(ids).logits
        le = edited(ids).logits
        for a in adapters.values():
            a.scale = 0.0
        lb = base(ids).logits
        rows.append(
            (
                p[:28],
                (la - le).abs().max().item(),
                (lb - le).abs().max().item(),
                (la.argmax(-1) == le.argmax(-1)).float().mean().item(),
            )
        )
print(
    f"{'prompt':30s} {'max|adapter-edited|':>20s} {'max|base-edited|':>18s} {'top1 agree':>10s}"
)
for r in rows:
    print(f"{r[0]:30s} {r[1]:20.4g} {r[2]:18.4g} {r[3]:10.3f}")
n_adapter = sum(int((a.c != 0).sum()) + a.u.numel() for a in adapters.values())
print(
    f"adapter parameters: {n_adapter:,} (dense u + 64 nonzero c per layer); "
    f"as dense rank-1 vectors: {sum(a.c.numel() + a.u.numel() for a in adapters.values()):,} floats"
)
