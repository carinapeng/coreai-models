"""Step 1: Core AI parity of the mutable-weight adapter export vs PyTorch base / edited.

PyTorch reference: HF Qwen3.5-0.8B in float32, base (adapter off) and the published edited
checkpoint. Core AI: the adapter export (fp16, preset none) with the adapter states set to
zeros (off) and to the published edit (on), plus the plain export without adapter states.
Full-sequence logits from one ``main`` call per prompt, compared at every position.
Also checks the incremental path: ``prefill`` on a prefix, then single-token ``main`` steps.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from coreai_llm import BASE_DIR, EDIT_DIR, EXP, CoreAIQwen35  # noqa: E402

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--adapter-export", default="exports/qwen35_adapter")
ap.add_argument("--plain-export", default=None)
ap.add_argument("--tag", default="fp16")
ap.add_argument(
    "--teacher-forced",
    action="store_true",
    help="prefill token 0, then main one token at a time (positions 1..T-1 compared)",
)
args = ap.parse_args()

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
print("prompt token counts:", [len(i) for i in ids_list])

# ---- PyTorch references (float32) ----
with torch.no_grad():
    base = AutoModelForCausalLM.from_pretrained(BASE_DIR, dtype=torch.float32).eval()
    edited = AutoModelForCausalLM.from_pretrained(EDIT_DIR, dtype=torch.float32).eval()
    pt_base = [base(torch.tensor([i])).logits[0].float().numpy() for i in ids_list]
    pt_edit = [edited(torch.tensor([i])).logits[0].float().numpy() for i in ids_list]
del base, edited


def cmp(a, b):
    return float(np.abs(a - b).max()), float((a.argmax(-1) == b.argmax(-1)).mean())


# ---- Core AI ----
results = {
    "prompts": [p[:60] for p in prompts],
    "n_tokens": [len(i) for i in ids_list],
    "rows": [],
}
S0 = 1 if args.teacher_forced else 0
if S0:
    pt_base = [x[S0:] for x in pt_base]
    pt_edit = [x[S0:] for x in pt_edit]


def run(m, ids):
    print(f"run {len(ids)} tokens", flush=True)
    m.reset()
    if not args.teacher_forced:
        return m.forward(ids)
    m.prefill(ids[:1])
    return np.stack([m.forward([t])[0] for t in ids[1:]])


# One loaded model at a time: each compile needs GBs of temp disk.
ca_plain = None
results["load_seconds"] = {}
if args.plain_export:
    plain = CoreAIQwen35(EXP / args.plain_export / "model.aimodel", adapter=False)
    ca_plain = []
    for ids in ids_list:
        ca_plain.append(run(plain, ids))
    plain.close()
    results["load_seconds"]["plain"] = plain.load_seconds
ca = CoreAIQwen35(EXP / args.adapter_export / "model.aimodel")
results["load_seconds"]["adapter"] = ca.load_seconds
print("load seconds (first load compiles):", results["load_seconds"])

ca_off, ca_on = [], []
for ids in ids_list:
    ca.set_adapter(0.0)
    ca_off.append(run(ca, ids))
    ca.set_adapter(1.0)
    ca_on.append(run(ca, ids))

hdr = (
    f"{'prompt':28s} {'T':>4s} | {'CA off vs PT base':>18s} | {'CA on vs PT edited':>19s} | "
    f"{'CA on vs PT base':>17s} | {'CA off vs CA plain':>18s} | {'PT edit vs base':>16s}"
)
print(hdr)
pt4_path = EXP / "results/pt4bit_logits.npz"
pt4 = np.load(pt4_path) if (pt4_path.exists() and "4bit" in args.tag) else None


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


for k, p in enumerate(prompts):
    r = {
        "prompt": p[:28].replace("\n", " "),
        "T": len(ids_list[k]),
        "ca_off_vs_pt_base": cmp(ca_off[k], pt_base[k]),
        "ca_on_vs_pt_edited": cmp(ca_on[k], pt_edit[k]),
        "ca_on_vs_pt_base": cmp(ca_on[k], pt_base[k]),
        "ca_off_vs_ca_plain": cmp(ca_off[k], ca_plain[k])
        if ca_plain
        else (float("nan"), float("nan")),
        "pt_edited_vs_pt_base": cmp(pt_edit[k], pt_base[k]),
        "ca_on_minus_off_vs_pt_edit_minus_base_maxabs": float(
            np.abs((ca_on[k] - ca_off[k]) - (pt_edit[k] - pt_base[k])).max()
        ),
    }
    if pt4 is not None:
        p4off, p4on = (
            pt4[f"{k}_off"].astype(np.float32)[S0:],
            pt4[f"{k}_on"].astype(np.float32)[S0:],
        )
        r["ca_off_vs_pt4bit_off"] = cmp(ca_off[k], p4off)
        r["ca_on_vs_pt4bit_on"] = cmp(ca_on[k], p4on)
        r["ca_on_vs_pt4bit_off"] = cmp(ca_on[k], p4off)
        r["pt4bit_on_vs_pt4bit_off"] = cmp(p4on, p4off)
        r["delta_corr_ca_vs_pt4bit"] = corr(ca_on[k] - ca_off[k], p4on - p4off)
        r["delta_corr_ca_vs_pt_fp32"] = corr(
            ca_on[k] - ca_off[k], pt_edit[k] - pt_base[k]
        )
        r["delta_corr_pt4bit_vs_pt_fp32"] = corr(p4on - p4off, pt_edit[k] - pt_base[k])
    results["rows"].append(r)

    def f(t):
        return f"{t[0]:8.4f} / {t[1]:6.3f}"

    print(
        f"{r['prompt']:28s} {r['T']:4d} | {f(r['ca_off_vs_pt_base']):>18s} | "
        f"{f(r['ca_on_vs_pt_edited']):>19s} | {f(r['ca_on_vs_pt_base']):>17s} | "
        f"{f(r['ca_off_vs_ca_plain']):>18s} | {f(r['pt_edited_vs_pt_base']):>16s}"
    )
print(
    "(each cell: max |dlogit| over all positions x vocab / top-1 agreement over positions)"
)
if pt4 is not None:
    print(
        f"{'prompt':28s} | {'CA off vs PT4 off':>18s} | {'CA on vs PT4 on':>17s} | {'CA on vs PT4 off':>17s} | "
        f"{'PT4 on vs off':>15s} | corr(dCA,dPT4) corr(dCA,dPT32) corr(dPT4,dPT32)"
    )
    for r in results["rows"]:

        def f(t):
            return f"{t[0]:8.4f} / {t[1]:6.3f}"

        print(
            f"{r['prompt']:28s} | {f(r['ca_off_vs_pt4bit_off']):>18s} | {f(r['ca_on_vs_pt4bit_on']):>17s} | "
            f"{f(r['ca_on_vs_pt4bit_off']):>17s} | {f(r['pt4bit_on_vs_pt4bit_off']):>15s} | "
            f"{r['delta_corr_ca_vs_pt4bit']:14.4f} {r['delta_corr_ca_vs_pt_fp32']:15.4f} {r['delta_corr_pt4bit_vs_pt_fp32']:16.4f}"
        )
    print(
        "(d = adapter on minus off, all positions x vocab; PT4 = coreai-opt 4bit model in eager PyTorch)"
    )

# ---- Incremental path: prefill(prefix) then main one token at a time, adapter on ----
ids = ids_list[-1]
split = len(ids) - 8
ca.set_adapter(1.0)
ca.reset()
ca.prefill(ids[:split])
steps = [ca.forward([t])[0] for t in ids[split:]]
inc = np.stack(steps)
results["incremental_on_vs_pt_edited_last8"] = cmp(inc, pt_edit[-1][split - S0 :])
if pt4 is not None:
    results["incremental_on_vs_pt4bit_on_last8"] = cmp(
        inc, pt4[f"{len(prompts) - 1}_on"].astype(np.float32)[split:]
    )
    print("incremental vs PT4 on:", results["incremental_on_vs_pt4bit_on_last8"])
results["incremental_on_vs_ca_full_last8"] = cmp(inc, ca_on[-1][split - S0 :])
print(
    "incremental (prefill + 8 decode steps, adapter on) vs PT edited:",
    results["incremental_on_vs_pt_edited_last8"],
    " vs CA full-seq:",
    results["incremental_on_vs_ca_full_last8"],
)

# ---- Toggle mid-stream without reload: off -> on -> off on the same loaded model ----
tog = []
for s in (0.0, 1.0, 0.0):
    ca.set_adapter(s)
    tog.append(run(ca, ids_list[0]))
results["toggle_off_on_off_first_vs_last_maxabs"] = float(np.abs(tog[0] - tog[2]).max())
results["toggle_on_vs_pt_edited"] = cmp(tog[1], pt_edit[0])
results["teacher_forced"] = bool(S0)
print(
    "toggle off->on->off: |off1-off2| max",
    results["toggle_off_on_off_first_vs_last_maxabs"],
    " on vs PT edited",
    results["toggle_on_vs_pt_edited"],
)

(EXP / "results").mkdir(exist_ok=True)
(EXP / f"results/parity_{args.tag}.json").write_text(json.dumps(results, indent=1))
ca.close()
