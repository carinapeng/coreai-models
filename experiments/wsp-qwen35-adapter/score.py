"""Score wspgen outputs: decode, loop metrics (loopstats.py's own functions), boxed share,
and accuracy (math-verify, run in the separate .gradeenv interpreter).

usage: python score.py results/gen_main.jsonl jobs/main.jsonl [more result/job pairs...]
Writes <result>.scored.jsonl (per sample) and <result>.summary.json (per scale).
"""

import ast
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from coreai_llm import BASE_DIR, EXP  # noqa: E402

# Pull tail_repetition / has_boxed out of loopstats.py without running its module-level loop.
_src = (EXP / "loopstats.py").read_text()
_ns: dict = {}
for node in ast.parse(_src).body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "tail_repetition",
        "has_boxed",
    ):
        exec(compile(ast.Module([node], []), "loopstats.py", "exec"), _ns)
tail_repetition, has_boxed = _ns["tail_repetition"], _ns["has_boxed"]

GRADER = r'''
import json, sys
from math_verify import parse, verify


def last_boxed(text):
    """Content of the last \\boxed{...} in text (brace-matched), or None."""
    i = text.rfind("\\boxed")
    while i >= 0:
        j = text.find("{", i)
        if j < 0:
            return None
        depth, k = 0, j
        while k < len(text):
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            if depth == 0:
                return text[j + 1:k]
            k += 1
        i = text.rfind("\\boxed", 0, i)  # unterminated: try the previous one
    return None


def check(answer, text):
    b = last_boxed(text) if text else None
    if b is None:
        return False, None
    try:
        return bool(verify(parse("$" + answer + "$"), parse("$" + b + "$"))), b
    except Exception:
        return False, b


for line in sys.stdin:
    r = json.loads(line)
    strict, bs = check(r["answer"], r["after_think"])
    lenient, bl = check(r["answer"], r["text"])
    print(json.dumps({"key": r["key"], "correct": strict, "correct_lenient": lenient,
                      "pred_strict": bs, "pred_lenient": bl}))
'''


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar (binomial on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def main(pairs):
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    rows_ds = [json.loads(line) for line in open(EXP / "data/math500_test.jsonl")]
    for res_path, jobs_path in pairs:
        jobs = {
            json.loads(line)["key"]: json.loads(line) for line in open(EXP / jobs_path)
        }
        recs = []
        for line in open(EXP / res_path):
            r = json.loads(line)
            j = jobs[r["key"]]
            text = tok.decode(r["tokens"], skip_special_tokens=True)
            final = text.split("</think>")[-1] if "</think>" in text else text
            recs.append(
                {
                    "key": r["key"],
                    "scale": j["scale"],
                    "a": 6 * j["scale"],
                    "idx": j["idx"],
                    "sample": j["sample"],
                    "answer": rows_ds[j["idx"]]["answer"],
                    "gen_tokens": len(r["tokens"]),
                    "hit_cap": r["finish"] == "length",
                    "closed_think": "</think>" in text,
                    "tail_rep": tail_repetition(text),
                    "boxed_tail2000": has_boxed(text),
                    "boxed_after_think": "\\boxed" in final and "</think>" in text,
                    "decode_tok_per_s": len(r["tokens"])
                    / max(r["decode_seconds"], 1e-9),
                    "final": final[-4000:],
                    "text": text,
                }
            )
        g = subprocess.run(
            [str(EXP / ".gradeenv/bin/python"), "-c", GRADER],
            capture_output=True,
            text=True,
            input="".join(
                json.dumps(
                    {
                        "key": x["key"],
                        "answer": x["answer"],
                        "text": x["text"],
                        "after_think": x["text"].split("</think>")[-1]
                        if x["closed_think"]
                        else "",
                    }
                )
                + "\n"
                for x in recs
            ),
            check=True,
        )
        graded = {
            json.loads(line)["key"]: json.loads(line) for line in g.stdout.splitlines()
        }
        for x in recs:
            gr = graded[x["key"]]
            x["correct"], x["correct_lenient"] = gr["correct"], gr["correct_lenient"]
            x["pred_strict"], x["pred_lenient"] = gr["pred_strict"], gr["pred_lenient"]
            x["loop_tail_rep_gt_0.5"] = x["tail_rep"] > 0.5
        out = EXP / res_path
        with open(out.with_suffix(".scored.jsonl"), "w") as f:
            for x in recs:
                f.write(json.dumps({k: v for k, v in x.items() if k != "final"}) + "\n")

        by = defaultdict(list)
        for x in recs:
            by[x["scale"]].append(x)
        summary = {}
        for s, xs in sorted(by.items()):
            n = len(xs)
            summary[f"{s:g}"] = {
                "a": 6 * s,
                "n": n,
                "hit_cap": sum(x["hit_cap"] for x in xs) / n,
                "tail_rep_gt_0.5": sum(x["loop_tail_rep_gt_0.5"] for x in xs) / n,
                "hit_cap_or_tail_rep": sum(
                    x["hit_cap"] or x["loop_tail_rep_gt_0.5"] for x in xs
                )
                / n,
                "closed_think": sum(x["closed_think"] for x in xs) / n,
                "boxed_tail2000": sum(x["boxed_tail2000"] for x in xs) / n,
                "boxed_after_think": sum(x["boxed_after_think"] for x in xs) / n,
                "accuracy": sum(x["correct"] for x in xs) / n,
                "accuracy_lenient": sum(x["correct_lenient"] for x in xs) / n,
                "mean_gen_tokens": sum(x["gen_tokens"] for x in xs) / n,
                "mean_tail_rep": sum(x["tail_rep"] for x in xs) / n,
                "decode_tok_per_s_mean": sum(x["decode_tok_per_s"] for x in xs) / n,
            }
        # Paired comparisons of every scale against scale 0 on the problems both have.
        if 0.0 in by:
            base = {(x["idx"], x["sample"]): x for x in by[0.0]}
            for s, xs in by.items():
                if s == 0.0:
                    continue
                pairs_ = [
                    (base[(x["idx"], x["sample"])], x)
                    for x in xs
                    if (x["idx"], x["sample"]) in base
                ]
                for m in (
                    "correct",
                    "correct_lenient",
                    "hit_cap",
                    "loop_tail_rep_gt_0.5",
                    "closed_think",
                ):
                    b = sum(p[m] and not q[m] for p, q in pairs_)
                    c = sum(q[m] and not p[m] for p, q in pairs_)
                    summary[f"{s:g}"][f"paired_vs_base_{m}"] = {
                        "n_pairs": len(pairs_),
                        "base_only": b,
                        "this_only": c,
                        "mcnemar_exact_p": mcnemar_p(b, c),
                    }
        out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
        hdr = (
            f"{'a':>5s} {'n':>3s} {'hit cap':>8s} {'tailrep>.5':>10s} {'either':>7s} {'</think>':>8s} {'boxed':>6s} "
            f"{'boxed>':>6s} {'acc':>6s} {'acc_len':>7s} {'mean tok':>8s}"
        )
        print(res_path)
        print(hdr)
        for k, v in summary.items():
            print(
                f"{v['a']:5g} {v['n']:3d} {v['hit_cap']:8.2f} {v['tail_rep_gt_0.5']:10.2f} {v['hit_cap_or_tail_rep']:7.2f} "
                f"{v['closed_think']:8.2f} {v['boxed_tail2000']:6.2f} {v['boxed_after_think']:6.2f} {v['accuracy']:6.2f} "
                f"{v['accuracy_lenient']:7.2f} {v['mean_gen_tokens']:8.0f}"
            )
            for m in (
                "correct",
                "correct_lenient",
                "hit_cap",
                "loop_tail_rep_gt_0.5",
                "closed_think",
            ):
                pv = v.get(f"paired_vs_base_{m}")
                if pv:
                    print(
                        f"      paired vs a=0 [{m}]: base-only {pv['base_only']}, this-only {pv['this_only']}, "
                        f"exact McNemar p={pv['mcnemar_exact_p']:.3g} (n={pv['n_pairs']})"
                    )


if __name__ == "__main__":
    a = sys.argv[1:]
    main(list(zip(a[0::2], a[1::2])))
