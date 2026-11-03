"""Write wspgen jobs for MATH-500: N seeded problems x samples x adapter scales.

Prompt: Qwen chat template, thinking on, problem + the standard boxed-answer instruction.
Per-sample seed depends on (problem, sample) only, so every adapter scale sees the same seed.

usage: python make_jobs.py --scales 0 1 --n 50 --cap 8192 --out jobs/main.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from coreai_llm import BASE_DIR, EXP  # noqa: E402

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
ap = argparse.ArgumentParser()
ap.add_argument("--scales", type=float, nargs="+", required=True)
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--samples", type=int, default=1)
ap.add_argument("--cap", type=int, default=8192)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(BASE_DIR)
rows = [json.loads(line) for line in open(EXP / "data/math500_test.jsonl")]
pids = sorted(np.random.default_rng(a.seed).choice(500, a.n, replace=False).tolist())
out = EXP / a.out
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for i in pids:
        text = tok.apply_chat_template(
            [{"role": "user", "content": rows[i]["problem"] + SUFFIX}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        ids = tok(text).input_ids
        for j in range(a.samples):
            for s in a.scales:
                f.write(
                    json.dumps(
                        {
                            "key": f"s{s:g}_i{i}_j{j}",
                            "scale": s,
                            "idx": i,
                            "sample": j,
                            "id": rows[i]["unique_id"],
                            "prompt": ids,
                            "seed": a.seed * 1_000_003 + i * 1009 + j,
                            "cap": a.cap,
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "top_k": 20,
                            "presence_penalty": 1.5,
                        }
                    )
                    + "\n"
                )
print(f"{len(pids) * a.samples * len(a.scales)} jobs -> {out}; problems: {pids}")
print("stop ids:", tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
