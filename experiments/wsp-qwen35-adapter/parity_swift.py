"""Step 1 via the Swift driver (wspgen logits; GPU preferred + expectFrequentReshapes, the
LLM runner's load options): teacher-forced logits at positions 1..T-1 for the parity prompts,
adapter off (scale 0) and on (scale 1), vs PyTorch fp32 base / edited checkpoint and, for a
4bit export, vs the matched coreai-opt 4-bit model in eager PyTorch (results/pt4bit_logits.npz).

usage: python parity_swift.py <export dir> <tag>
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from coreai_llm import BASE_DIR, EDIT_DIR, EXP  # noqa: E402

export, tag = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(BASE_DIR)
raw_prompts = [
    "Solve: what is 17 * 23? Think step by step.",
    "Find all real x such that x^2 - 5x + 6 = 0.",
    "Write a short poem about the ocean.",
    "The capital of France is",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
]
math_q = json.loads(open(EXP / "data/math500_test.jsonl").readline())["problem"]
chat = tok.apply_chat_template(
    [{"role": "user", "content": math_q}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
prompts = raw_prompts + [chat]
ids_list = [tok(p).input_ids for p in prompts]

jobs = EXP / f"jobs/parity_{tag}.jsonl"
with open(jobs, "w") as f:
    for s in (0.0, 1.0):
        for k, ids in enumerate(ids_list):
            f.write(
                json.dumps({"key": f"p{k}_s{s:g}", "prompt": ids, "scale": s}) + "\n"
            )
binp = EXP / f"results/swift_logits_{tag}.bin"
subprocess.run(
    [
        str(EXP / "swiftgen/.build3/release/wspgen"),
        "logits",
        str(EXP / export / "model.aimodel"),
        str(EXP / "adapter.bin"),
        str(jobs),
        str(binp),
    ],
    check=True,
    capture_output=True,
)
V = 248320
flat = np.fromfile(binp, dtype=np.float16).astype(np.float32).reshape(-1, V)
ca, o = {}, 0
for s in (0.0, 1.0):
    for k, ids in enumerate(ids_list):
        n = len(ids) - 1
        ca[(k, s)] = flat[o : o + n]
        o += n
assert o == flat.shape[0]
binp.unlink()  # ~120 MB; the comparisons below are what is kept

with torch.no_grad():
    base = AutoModelForCausalLM.from_pretrained(BASE_DIR, dtype=torch.float32).eval()
    edited = AutoModelForCausalLM.from_pretrained(EDIT_DIR, dtype=torch.float32).eval()
    pt_base = [base(torch.tensor([i])).logits[0, 1:].numpy() for i in ids_list]
    pt_edit = [edited(torch.tensor([i])).logits[0, 1:].numpy() for i in ids_list]
pt4 = np.load(EXP / "results/pt4bit_logits.npz") if "4bit" in tag else None


def cmp(a, b):
    return float(np.abs(a - b).max()), float((a.argmax(-1) == b.argmax(-1)).mean())


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


rows = []
for k, p in enumerate(prompts):
    off, on = ca[(k, 0.0)], ca[(k, 1.0)]
    r = {
        "prompt": p[:28].replace("\n", " "),
        "positions": len(ids_list[k]) - 1,
        "ca_off_vs_pt_base": cmp(off, pt_base[k]),
        "ca_on_vs_pt_edited": cmp(on, pt_edit[k]),
        "ca_on_vs_pt_base": cmp(on, pt_base[k]),
        "pt_edited_vs_pt_base": cmp(pt_edit[k], pt_base[k]),
        "delta_corr_ca_vs_pt_fp32": corr(on - off, pt_edit[k] - pt_base[k]),
    }
    if pt4 is not None:
        p4off, p4on = (
            pt4[f"{k}_off"].astype(np.float32)[1:],
            pt4[f"{k}_on"].astype(np.float32)[1:],
        )
        r |= {
            "ca_off_vs_pt4bit_off": cmp(off, p4off),
            "ca_on_vs_pt4bit_on": cmp(on, p4on),
            "delta_corr_ca_vs_pt4bit": corr(on - off, p4on - p4off),
        }
    rows.append(r)


def f(t):
    return f"{t[0]:7.4f}/{t[1]:5.3f}"


print(
    f"[{tag}] Swift driver, teacher-forced, positions 1..T-1; cells: max|dlogit| / top-1 agreement"
)
print(
    f"{'prompt':28s} {'n':>3s} {'off~PTbase':>14s} {'on~PTedit':>14s} {'on~PTbase':>14s} {'PTedit~PTbase':>14s} "
    f"{'corr(dCA,dPT32)':>15s}"
    + (
        f" {'off~PT4off':>14s} {'on~PT4on':>14s} {'corr(dCA,dPT4)':>14s}"
        if pt4 is not None
        else ""
    )
)
for r in rows:
    line = (
        f"{r['prompt']:28s} {r['positions']:3d} {f(r['ca_off_vs_pt_base']):>14s} {f(r['ca_on_vs_pt_edited']):>14s} "
        f"{f(r['ca_on_vs_pt_base']):>14s} {f(r['pt_edited_vs_pt_base']):>14s} {r['delta_corr_ca_vs_pt_fp32']:15.4f}"
    )
    if pt4 is not None:
        line += f" {f(r['ca_off_vs_pt4bit_off']):>14s} {f(r['ca_on_vs_pt4bit_on']):>14s} {r['delta_corr_ca_vs_pt4bit']:14.4f}"
    print(line)
(EXP / f"results/parity_swift_{tag}.json").write_text(
    json.dumps({"export": export, "rows": rows}, indent=1)
)
