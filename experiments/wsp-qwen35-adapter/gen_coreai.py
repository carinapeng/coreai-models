"""Step 3/4: MATH-500 generation on Core AI with the mutable-weight adapter at a given scale.

One loaded model serves every condition: the adapter scale is set with set_weight between
samples, no re-export or reload. Host-side sampling (numpy) on the logits ``main`` returns:
temperature, top_k, top_p, presence_penalty (vLLM convention: subtract the penalty from the
logit of every token already present in the generated output). Per-sample RNG is seeded
from (seed, problem index, sample index), identical across adapter scales.

Appends one JSON row per finished sample to the output jsonl and skips rows already present,
so an interrupted run resumes where it stopped.

usage: python gen_coreai.py --scales 0 1 --n 50 --cap 8192 --out results/gen_coreai.jsonl
       (scale s = s x the published edit; the blog's a = 6 s)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from coreai_llm import BASE_DIR, EXP, CoreAIQwen35  # noqa: E402

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def problem_ids(n: int, seed: int) -> list[int]:
    return sorted(np.random.default_rng(seed).choice(500, n, replace=False).tolist())


def build_prompt(tok, problem: str) -> list[int]:
    text = tok.apply_chat_template(
        [{"role": "user", "content": problem + SUFFIX}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return tok(text).input_ids


def sample(
    logits: np.ndarray, rng, generated_set, temperature, top_k, top_p, presence_penalty
):
    x = logits.astype(np.float32)
    if presence_penalty and generated_set:
        idx = np.fromiter(generated_set, dtype=np.int64)
        x[idx] -= presence_penalty
    x = x / temperature
    k_idx = np.argpartition(x, -top_k)[-top_k:]
    k_log = x[k_idx]
    order = np.argsort(-k_log)
    k_idx, k_log = k_idx[order], k_log[order]
    p = np.exp(k_log - k_log.max())
    p /= p.sum()
    keep = np.cumsum(p) - p < top_p  # smallest prefix with mass >= top_p
    p = p[keep]
    p /= p.sum()
    return int(k_idx[keep][rng.choice(len(p), p=p)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="exports/qwen35_4bit_adapter")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--presence-penalty", type=float, default=1.5)
    ap.add_argument("--compute", default="default")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--shard",
        type=int,
        nargs=2,
        default=(0, 1),
        help="i n: take every n-th job from i",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    rows = [json.loads(line) for line in open(EXP / "data/math500_test.jsonl")]
    pids = problem_ids(args.n, args.seed)

    out = EXP / args.out
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["scale"], r["idx"], r["sample"]))
    jobs = [(s, i, j) for i in pids for j in range(args.samples) for s in args.scales]
    jobs = jobs[args.shard[0] :: args.shard[1]]
    todo = [j for j in jobs if j not in done]
    print(
        f"{len(jobs)} jobs, {len(todo)} to do; runtime={'bundled' if os.environ.get('USE_LOCAL_COREAI') else 'os'}",
        flush=True,
    )
    max_ctx = 9216
    m = CoreAIQwen35(
        EXP / args.export / "model.aimodel", max_ctx=max_ctx, compute=args.compute
    )
    settings = {
        k: getattr(args, k)
        for k in ("cap", "seed", "temperature", "top_p", "top_k", "presence_penalty")
    }
    t_all, n_all = time.perf_counter(), 0
    for scale, i, j in todo:
        r = rows[i]
        m.set_adapter(scale)
        m.reset()
        ids = build_prompt(tok, r["problem"])
        cap = min(args.cap, max_ctx - len(ids) - 1)
        rng = np.random.default_rng([args.seed, i, j])
        t0 = time.perf_counter()
        logits = m.forward(ids)[-1]
        gen, gen_set, finish = [], set(), "length"
        for _ in range(cap):
            t = sample(
                logits,
                rng,
                gen_set,
                args.temperature,
                args.top_k,
                args.top_p,
                args.presence_penalty,
            )
            gen.append(t)
            gen_set.add(t)
            if t in stop:
                finish = "stop"
                break
            logits = m.forward([t])[-1]
        dt = time.perf_counter() - t0
        n_all += len(gen)
        rec = {
            "scale": scale,
            "a": 6 * scale,
            "idx": i,
            "sample": j,
            "id": r["unique_id"],
            "answer": r["answer"],
            "prompt_tokens": len(ids),
            "gen_tokens": len(gen),
            "finish": finish,
            "seconds": round(dt, 1),
            "settings": settings,
            "text": tok.decode(gen, skip_special_tokens=False),
        }
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"scale={scale:g} idx={i} s={j} tokens={len(gen)} finish={finish} {len(gen) / dt:.1f} tok/s "
            f"(overall {n_all / (time.perf_counter() - t_all):.1f} tok/s)",
            flush=True,
        )
    m.close()


if __name__ == "__main__":
    main()
