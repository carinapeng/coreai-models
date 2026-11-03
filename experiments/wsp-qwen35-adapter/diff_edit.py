"""Which weights did the published Qwen3.5-0.8B edit change, and what is the delta's structure?"""

import json
import os
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
import glob

HERE = os.path.dirname(os.path.abspath(__file__))


base_dir = snapshot_download(
    "Qwen/Qwen3.5-0.8B", allow_patterns=["*.safetensors", "*.json"]
)
edit_dir = snapshot_download(
    "RikiyaT/Qwen3.5-0.8B-weight-space-programming",
    allow_patterns=["*.safetensors", "*.json", "eval/*"],
)

bf = glob.glob(os.path.join(base_dir, "*.safetensors"))[0]
ef = glob.glob(os.path.join(edit_dir, "*.safetensors"))[0]
changed, total = [], 0
with safe_open(bf, "pt") as b, safe_open(ef, "pt") as e:
    bk, ek = set(b.keys()), set(e.keys())
    print(
        "tensors: base",
        len(bk),
        "edited",
        len(ek),
        "| only in base",
        len(bk - ek),
        "| only in edited",
        len(ek - bk),
    )
    for k in sorted(bk & ek):
        tb, te = b.get_tensor(k), e.get_tensor(k)
        total += tb.numel()
        if tb.shape != te.shape:
            print("SHAPE CHANGE", k, tb.shape, te.shape)
            continue
        d = te.float() - tb.float()
        nz = int((d != 0).sum())
        if nz:
            changed.append((k, tb.shape, tb.dtype, nz, d))
print(f"total params compared: {total:,}")
for k, shape, dtype, nz, d in changed:
    cols = torch.nonzero(d.abs().sum(0) > 0).flatten()
    rows = torch.nonzero(d.abs().sum(1) > 0).flatten()
    s = torch.linalg.svdvals(d)
    rank_eff = int((s > s[0] * 1e-3).sum())
    print(
        f"\n{k}  shape={tuple(shape)} dtype={dtype}  changed={nz:,} ({nz / d.numel():.4%} of tensor)"
    )
    print(
        f"  rows touched={len(rows)}  cols touched={len(cols)}  effective rank(1e-3)={rank_eff}  top sv={s[:3].tolist()}"
    )
    print(
        f"  |delta| max={d.abs().max():.4g} mean(nonzero)={d[d != 0].abs().mean():.4g}  ||delta||/||W||={d.norm() / torch.tensor(0.0 + 1e-12).add(1).mul(0).add(1).mul(0) if False else (d.norm() / (d.norm() * 0 + 1)):.4g}"
    )
n = sum(c[3] for c in changed)
print(
    f"\nTOTAL changed weights: {n:,} of {total:,} ({n / total:.4%}) across {len(changed)} tensors"
)
json.dump(
    {"changed": [(k, list(s), nz) for k, s, _, nz, _ in changed], "total": total},
    open(HERE + "/diff_summary.json", "w"),
    indent=1,
)
print("edit dir:", edit_dir)
