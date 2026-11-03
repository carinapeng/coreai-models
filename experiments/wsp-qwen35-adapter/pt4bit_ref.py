"""PyTorch reference for the 4bit export: the same coreai-opt 4-bit quantized model the
exporter traces (eager, CPU, fp16), with the same rank-1 MutableWeight adapter, off and on.

Writes results/pt4bit_logits.npz: {f"{k}_off", f"{k}_on"} for each parity prompt k.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from coreai_export import ADAPTER_LAYERS, HF_ID, Rank1MutableAdapter  # noqa: E402
from coreai_llm import BASE_DIR, EXP, dense_adapter  # noqa: E402
from coreai_models.export.pipeline import ExportConfig, build_model, resolve_model  # noqa: E402

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
ids_list = [tok(p).input_ids for p in raw_prompts + [chat]]

cfg = ExportConfig(
    hf_model_id=HF_ID,
    compression="4bit",
    compute_precision="float16",
    max_context_length=9216,
)
resolved = resolve_model(cfg)
with tempfile.TemporaryDirectory(prefix="wsp_pt4bit_") as tmp:
    model = build_model(resolved, tmp).model.eval()
    for L in ADAPTER_LAYERS:
        mlp = model.model.layers[L].mlp
        mlp.down_proj = Rank1MutableAdapter(mlp.down_proj, f"l{L}", torch.float16)
    adapters = {
        f"l{L}_{v}": getattr(model.model.layers[L].mlp.down_proj, v)
        for L in ADAPTER_LAYERS
        for v in ("u", "c")
    }

    out = {}
    t0 = time.perf_counter()
    with torch.no_grad():
        for scale, tag in ((0.0, "off"), (1.0, "on")):
            for name, value in dense_adapter(scale).items():
                adapters[name].update(value)
            for k, ids in enumerate(ids_list):
                state = model.new_eager_state(
                    dtype=torch.float16, device="cpu", max_seq_len=128
                )
                args = model.eager_forward_args(
                    torch.tensor([ids], dtype=torch.int32), state
                )
                logits = model(*args)
                out[f"{k}_{tag}"] = logits[0].float().numpy().astype(np.float16)
                print(tag, k, len(ids), f"{time.perf_counter() - t0:.1f}s", flush=True)
np.savez(EXP / "results/pt4bit_logits.npz", **out)
print("saved", EXP / "results/pt4bit_logits.npz")
