"""Forward-only reward vs adapter strength: does a loop-likelihood proxy find the published fix?"""

import glob
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


spec = importlib.util.spec_from_file_location("loopstats", HERE + "/loopstats.py")
ls = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
H = os.path.expanduser("~/.cache/huggingface/hub")
base_dir = glob.glob(f"{H}/models--Qwen--Qwen3.5-0.8B/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(base_dir)
model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32).eval()
edit = torch.load(HERE + "/rank1_edit.pt")
layers = next(
    m
    for n, m in model.named_modules()
    if n.endswith("layers") and isinstance(m, torch.nn.ModuleList)
)


class Rank1Adapter(torch.nn.Module):
    def __init__(self, inner, u, c):
        super().__init__()
        self.inner = inner
        self.u = u
        self.c = c
        self.scale = 0.0

    def forward(self, h):
        return self.inner(h) + self.scale * (h @ self.c)[..., None] * self.u


ads = []
for L, e in edit.items():
    c = torch.zeros(layers[L].mlp.down_proj.in_features)
    c[e["cols"]] = e["coef"]
    a = Rank1Adapter(layers[L].mlp.down_proj, e["u"].float(), c)
    layers[L].mlp.down_proj = a
    ads.append(a)


def tail_rep(t):
    t = t[-4000:]
    g = [t[i : i + 30] for i in range(0, len(t) - 30, 10)]
    s = {}
    for x in g:
        s[x] = s.get(x, 0) + 1
    return sum(c for c in s.values() if c > 1) / max(1, len(g))


base_rows = [json.loads(line) for line in open(HERE + "/samples_base_math500.jsonl")]
edit_rows = [json.loads(line) for line in open(HERE + "/samples_edited_math500.jsonl")]
looping = [r["text"] for r in base_rows if tail_rep(r["text"]) > 0.5][:8]
clean = [r["text"] for r in edit_rows if tail_rep(r["text"]) < 0.3][:8]
CTX, TAIL = 384, 256


def seqs(texts, where):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids[0]
        out.append(ids[-(CTX + TAIL) :] if where == "tail" else ids[: CTX + TAIL])
    return out


loop_seqs, clean_seqs = seqs(looping, "tail"), seqs(clean, "head")


@torch.no_grad()
def mean_logprob(seq_list):
    tot, n = 0.0, 0
    for ids in seq_list:
        logits = model(ids[None]).logits[0, :-1]
        lp = torch.log_softmax(logits.float(), -1).gather(-1, ids[1:, None])[:, 0]
        tot += lp[-TAIL:].sum().item()
        n += TAIL
    return tot / n


print(
    f"looping tails: {len(loop_seqs)}  clean reasoning: {len(clean_seqs)}  (last {TAIL} tokens scored, {CTX} context)"
)
print(
    f"{'strength s':>10s} {'a=6s':>6s} {'loop tail logprob/tok':>22s} {'clean ppl':>10s}"
)
for s in [0.0, 1 / 6, 0.5, 1.0, 1.5, 2.0]:
    for a in ads:
        a.scale = s
    lt = mean_logprob(loop_seqs)
    cl = mean_logprob(clean_seqs)
    print(
        f"{s:10.3f} {6 * s:6.2f} {lt:22.3f} {torch.exp(torch.tensor(-cl)).item():10.3f}"
    )
