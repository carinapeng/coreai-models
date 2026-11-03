"""Greedy 48-token decode, adapter on and off: PyTorch-4bit eager vs the Swift driver (wspgen) on Core AI."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from coreai_export import ADAPTER_LAYERS, HF_ID, Rank1MutableAdapter
from coreai_llm import EXP, dense_adapter
from coreai_models.export.pipeline import ExportConfig, build_model, resolve_model

job0 = json.loads(open(EXP / "jobs/main.jsonl").readline())
prompt, N = job0["prompt"], 48
jobs = EXP / "jobs/sanity.jsonl"
with open(jobs, "w") as f:
    for s in (0.0, 1.0):
        f.write(
            json.dumps(
                {
                    **job0,
                    "key": f"greedy_s{s:g}",
                    "scale": s,
                    "cap": N,
                    "top_k": 1,
                    "presence_penalty": 0.0,
                }
            )
            + "\n"
        )
out = EXP / "results/sanity_swift.jsonl"
out.unlink(missing_ok=True)
subprocess.run(
    [
        str(EXP / "swiftgen/.build/release/wspgen"),
        "gen",
        str(EXP / "exports/qwen35_4bit_adapter2d/model.aimodel"),
        str(EXP / "adapter.bin"),
        str(jobs),
        str(out),
        "248046,248044",
    ],
    check=True,
    capture_output=True,
)
swift = {json.loads(line)["key"]: json.loads(line)["tokens"] for line in open(out)}

cfg = ExportConfig(
    hf_model_id=HF_ID,
    compression="4bit",
    compute_precision="float16",
    max_context_length=9216,
)
with tempfile.TemporaryDirectory() as tmp:
    model = build_model(resolve_model(cfg), tmp).model.eval()
for L in ADAPTER_LAYERS:
    mlp = model.model.layers[L].mlp
    mlp.down_proj = Rank1MutableAdapter(mlp.down_proj, f"l{L}", torch.float16)
res = {}
with torch.no_grad():
    for s in (0.0, 1.0):
        for name, v in dense_adapter(s).items():
            L, w = name[1:3], name[-1]
            getattr(model.model.layers[int(L)].mlp.down_proj, w).update(v)
        st = model.new_eager_state(
            dtype=torch.float16, device="cpu", max_seq_len=len(prompt) + N + 8
        )
        logits = model.eager_step(torch.tensor([prompt], dtype=torch.int32), st)
        toks = []
        for _ in range(N):
            t = int(logits[0, -1].argmax())
            toks.append(t)
            logits = model.eager_step(torch.tensor([[t]], dtype=torch.int32), st)
        sw = swift[f"greedy_s{s:g}"]
        agree = next(
            (i for i, (a, b) in enumerate(zip(toks, sw)) if a != b),
            min(len(toks), len(sw)),
        )
        res[f"scale{s:g}"] = {
            "first_divergence": agree,
            "n": N,
            "pt4bit": toks,
            "swift_coreai": sw,
        }
        print(f"scale {s:g}: identical prefix {agree}/{N} tokens")
print(
    "base vs adapter (PT4) first divergence:",
    next(
        (
            i
            for i, (a, b) in enumerate(
                zip(res["scale0"]["pt4bit"], res["scale1"]["pt4bit"])
            )
            if a != b
        ),
        N,
    ),
)
(EXP / "results/sanity_greedy.json").write_text(json.dumps(res, indent=1))
